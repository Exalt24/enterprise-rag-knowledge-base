"""The cause of a hybrid fallback must survive to the CI log, once, and must not be echoed by a second guard.

hybrid_search swallows every exception and answers with plain vector search. The count says THAT it happened; without the
cause a reader hunts through a library bump for a reason the run already had. And the hybrid rows of such a run are
vector numbers, so judging them against the baseline would print three more lines about the same single fault.
Everything below is generated from the product of exception type x where in hybrid_search it broke x how the message reads.
No live service, no model. Run (its own invocation, never together with test_hybrid_ranking.py):
    pytest tests/test_eval_fallback_cause.py
"""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import eval.harness as harness  # noqa: E402
from app.services.advanced_retrieval import AdvancedRetrieval  # noqa: E402
from eval.harness import Retrievers  # noqa: E402
from eval_fixtures import canned_baseline, canned_report  # noqa: E402

# CLASS-TABLE: exception type x the stage of hybrid_search that broke x how the message reads; the first cause is kept, one line
EXCEPTIONS = (RuntimeError, ValueError, KeyError, TypeError, OSError, ImportError)
STAGES = ("vector-search", "bm25-build", "bm25-query")
MESSAGES = {
    "plain": "bm25 index could not be built",
    "two-lines": "first line\nsecond line",
    "very-long": "x" * 500,
}
UNIT_CASES = list(itertools.product(EXCEPTIONS, STAGES, MESSAGES))


class _Store:
    def __init__(self, stage, exceptions):
        self.stage, self.exceptions = stage, iter(exceptions)

    def search_with_scores(self, query, k):
        if self.stage == "vector-search":
            raise next(self.exceptions)
        return [(Document(page_content="a"), 0.9), (Document(page_content="b"), 0.4)][:k]

    def search(self, query, k):
        return [Document(page_content="a"), Document(page_content="b")][:k]


class _Bm25:
    def __init__(self, stage, exceptions):
        self.stage, self.exceptions = stage, exceptions

    def invoke(self, query):
        if self.stage == "bm25-query":
            raise next(self.exceptions)
        return [Document(page_content="b")]


def _engine(stage, exceptions):
    eng = AdvancedRetrieval.__new__(AdvancedRetrieval)  # skip __init__: it wires an LLM and the live store
    exceptions = iter(exceptions)
    eng.vector_store = _Store(stage, exceptions)

    def bm25(k):
        if stage == "bm25-build":
            raise next(exceptions)
        return _Bm25(stage, exceptions)

    eng._get_bm25_retriever = bm25
    return eng


def _retrievers(engine):
    r = Retrievers.__new__(Retrievers)
    r.engine = engine
    return r


@pytest.mark.parametrize("exc_type,stage,message", UNIT_CASES, ids=lambda v: v if isinstance(v, str) else v.__name__)
def test_the_first_cause_is_kept_with_its_type_and_the_later_ones_do_not_replace_it(exc_type, stage, message):
    first = exc_type(MESSAGES[message])
    later = LookupError("a different fault later")
    eng = _engine(stage, [first, later])
    assert eng.hybrid_fallback_cause is None
    eng.hybrid_search("q", k=2)
    expected = f"{type(first).__name__}: {first}"
    assert eng.hybrid_fallback_count == 1 and eng.hybrid_fallback_cause == expected
    eng.hybrid_search("q", k=2)
    assert eng.hybrid_fallback_count == 2
    assert eng.hybrid_fallback_cause == expected, "the second fault replaced the first cause"
    shown = _retrievers(eng).fallback_cause
    assert shown == " ".join(expected.split())[:300]
    assert "\n" not in shown and len(shown) <= 300 and shown.startswith(exc_type.__name__ + ":")


def test_a_clean_engine_has_no_cause_and_the_class_default_is_not_shared():
    eng = _engine("none", [])
    eng.hybrid_search("q", k=2)
    assert eng.hybrid_fallback_cause is None and _retrievers(eng).fallback_cause == ""
    broken = _engine("bm25-build", [RuntimeError("boom")])
    broken.hybrid_search("q", k=2)
    assert AdvancedRetrieval.hybrid_fallback_cause is None and eng.hybrid_fallback_cause is None


# ---------------------------------------------------------------- the printed run

METHODS_THAT_FELL = [c for c in itertools.product((False, True), repeat=3)]  # (vector, bm25, hybrid) numbers below the pin
CAUSES = ("RuntimeError: BM25Retriever signature changed", "ImportError: cannot import name 'rank_bm25' from 'rank_bm25'")
CLI_CASES = list(itertools.product(METHODS_THAT_FELL, CAUSES))


@pytest.mark.parametrize("fell,cause", CLI_CASES, ids=lambda v: str(v))
def test_the_run_prints_the_cause_once_and_speaks_only_for_the_methods_that_did_not_fall_back(fell, cause, tmp_path, monkeypatch, capsys):
    import eval.run_eval as run_eval

    baseline = canned_baseline()
    for method, lower in zip(("vector", "bm25", "hybrid"), fell):
        if lower:
            baseline["numbers"][method]["direct"]["hit@1"] = 1.0  # pinned above this run's 0.5
    report = canned_report(fallbacks=58, cause=cause, fallback_methods=["hybrid"])
    pinned = tmp_path / "baseline.json"
    pinned.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_PATH", pinned)
    monkeypatch.setattr(run_eval, "run_evaluation", lambda *args, **kwargs: report)
    assert run_eval.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert out.count(cause) == 1, "the cause must be printed exactly once, beside the count"
    assert f"hybrid fallbacks (hybrid_search answered with plain vector search): 58 (first cause: {cause})" in out
    assert out.count("FAIL: hybrid_search fell back to vector-only 58 time(s)") == 1
    assert "NOT COMPARED: hybrid" in out
    assert "REGRESSION: hybrid/" not in out, "hybrid's numbers are vector numbers after a fallback, the fallback line already speaks"
    for method, lower in zip(("vector", "bm25"), fell):
        assert (f"REGRESSION: {method}/direct/hit@1" in out) == lower, method


def test_a_run_without_a_fallback_prints_no_cause_and_no_not_compared_line(capsys):
    text = harness.format_report(canned_report(), "T")
    assert "first cause" not in text
    assert "hybrid fallbacks (hybrid_search answered with plain vector search): 0" in text


# ---------------------------------------------------------------- the whole run, with the real engine


def test_a_bm25_failure_inside_hybrid_alone_gives_one_cause_line_and_no_regression_lines(monkeypatch, capsys):
    import eval.run_eval as run_eval

    original = AdvancedRetrieval._get_bm25_retriever

    def only_hybrid_breaks(self, k):
        if k == harness.RETRIEVAL_DEPTH * 2:  # hybrid_search asks for twice the depth, the BM25-only mode for the depth itself
            raise RuntimeError("BM25Retriever signature changed in the new langchain-community")
        return original(self, k)

    monkeypatch.setattr(AdvancedRetrieval, "_get_bm25_retriever", only_hybrid_breaks)
    assert run_eval.main(["--check"]) == 1
    out = capsys.readouterr().out
    cause = "RuntimeError: BM25Retriever signature changed in the new langchain-community"
    assert out.count(cause) == 1
    assert "REGRESSION" not in out and "CHUNKING CHANGED" not in out
    assert "NOT COMPARED: hybrid" in out
