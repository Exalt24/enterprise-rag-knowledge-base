"""Every kind of change against every pinned field, generated.

The baseline is one file that several different edits can disagree with: an improvement (a number rises), a small wobble
(a fall inside the tolerance), a real fall, a chunk change (a splitter bump), and an edit to a question or a label. Each
needs its own outcome and its own readable message, and a rise must never fail. The cases below are the product of every
pinned field and every kind of change, so a field or a kind nobody thought of is covered too.
Run: pytest tests/test_eval_baseline_changes.py
"""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from eval.harness import BASELINE_PATH, compare_to_baseline, verdict_lines  # noqa: E402
from eval_fixtures import canned_baseline, canned_report  # noqa: E402

BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
TOLERANCE = BASELINE["tolerance"]

# CLASS-TABLE: pinned field (method, category, metric) x kind of change; the expected outcome and message come from the kind
FIELDS = [
    (method, category, metric)
    for method, per_category in BASELINE["numbers"].items()
    for category, metrics in per_category.items()
    for metric in metrics
]
KINDS = (
    "same",
    "rise",
    "fall_inside_tolerance",
    "fall_exactly_the_tolerance",
    "fall_beyond_tolerance",
    "chunk_count_changed",
    "chunk_hash_changed",
    "questions_changed",
    "calibration_changed",
)
CASES = list(itertools.product(FIELDS, KINDS))

DELTA = {"same": 0.0, "rise": 0.1, "fall_inside_tolerance": -0.01, "fall_exactly_the_tolerance": -TOLERANCE, "fall_beyond_tolerance": -0.03}


def _report_like(baseline):
    """A report whose every number equals the pinned one, so exactly one field can be moved per case."""
    results = {}
    for method, per_category in baseline["numbers"].items():
        results[method] = {}
        for category, metrics in per_category.items():
            results[method][category] = {"n": 5, **{m: {"value": v, "lo": 0.0, "hi": 1.0} for m, v in metrics.items()}}
    return {
        "chunk_count": baseline["chunk_count"],
        "chunk_hash": baseline["chunk_hash"],
        "questions_hash": baseline["questions_hash"],
        "calibration_hash": baseline["calibration_hash"],
        "results": results,
        "fallback_methods": [],
    }


def _case(field, kind):
    method, category, metric = field
    report = _report_like(BASELINE)
    pinned = BASELINE["numbers"][method][category][metric]
    entry = report["results"][method][category][metric]
    if kind in DELTA:
        entry["value"] = round(pinned + DELTA[kind], 4)
    else:
        entry["value"] = round(pinned - 0.5, 4)  # a hard fall, which must NOT be judged when the cause is a change in what is measured
    if kind == "chunk_count_changed":
        report["chunk_count"] += 1
    elif kind == "chunk_hash_changed":
        report["chunk_hash"] = "0" * 64
    elif kind == "questions_changed":
        report["questions_hash"] = "0" * 64
    elif kind == "calibration_changed":
        report["calibration_hash"] = "0" * 64
    return report, pinned, entry["value"]


@pytest.mark.parametrize("field,kind", CASES, ids=lambda v: "/".join(v) if isinstance(v, tuple) else v)
def test_each_change_to_each_pinned_field_has_its_own_outcome_and_message(field, kind):
    method, category, metric = field
    report, pinned, now = _case(field, kind)
    verdict = compare_to_baseline(report, BASELINE)
    lines = verdict_lines(verdict)
    text = "\n".join(lines)
    if kind in ("same", "rise", "fall_inside_tolerance", "fall_exactly_the_tolerance"):
        assert verdict.ok and lines == [], f"{kind} must pass without a word: {lines}"
    elif kind == "fall_beyond_tolerance":
        assert not verdict.ok and verdict.chunking == [] and verdict.questions == []
        assert len(verdict.regressions) == 1, verdict.regressions
        assert f"REGRESSION: {method}/{category}/{metric} fell from {pinned} to {now}, by 0.03, more than the tolerance of 0.02" in text
        assert "CHUNKING CHANGED" not in text and "QUESTIONS CHANGED" not in text
    elif kind.startswith("chunk"):
        assert not verdict.ok and verdict.regressions == [] and verdict.questions == []
        assert text.startswith("CHUNKING CHANGED: ") and "--write-baseline" in text and "REGRESSION" not in text
    else:
        name = "questions.jsonl" if kind == "questions_changed" else "calibration.jsonl"
        assert not verdict.ok and verdict.regressions == [] and verdict.chunking == []
        assert text.startswith(f"QUESTIONS CHANGED: {name}") and "--write-baseline" in text and "REGRESSION" not in text


# The same kinds through the command line, so the text a person reads in CI is the text asserted here.
CLI_KINDS = ("same", "rise", "fall_inside_tolerance", "fall_beyond_tolerance", "chunk_hash_changed", "questions_changed", "calibration_changed")
CLI_CASES = list(itertools.product(("vector", "bm25", "hybrid"), CLI_KINDS))


@pytest.mark.parametrize("method,kind", CLI_CASES, ids=lambda v: v)
def test_the_command_line_prints_the_message_of_each_kind(method, kind, tmp_path, monkeypatch, capsys):
    import eval.run_eval as run_eval

    base = canned_baseline()
    report = canned_report()
    delta = {"same": 0.0, "rise": 0.1, "fall_inside_tolerance": -0.01, "fall_beyond_tolerance": -0.1}.get(kind, 0.0)
    report["results"][method]["direct"]["hit@1"]["value"] = round(0.5 + delta, 4)
    if kind == "chunk_hash_changed":
        report["chunk_hash"] = "d" * 64
    elif kind == "questions_changed":
        report["questions_hash"] = "d" * 64
    elif kind == "calibration_changed":
        report["calibration_hash"] = "d" * 64
    pinned = tmp_path / "baseline.json"
    pinned.write_text(json.dumps(base), encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_PATH", pinned)
    monkeypatch.setattr(run_eval, "run_evaluation", lambda *args, **kwargs: report)
    status = run_eval.main(["--check"])
    out = capsys.readouterr().out
    if kind in ("same", "rise", "fall_inside_tolerance"):
        assert status == 0 and "baseline check: OK" in out
    elif kind == "fall_beyond_tolerance":
        assert status == 1 and f"REGRESSION: {method}/direct/hit@1 fell from 0.5 to 0.4, by 0.1" in out
    elif kind == "chunk_hash_changed":
        assert status == 1 and "CHUNKING CHANGED" in out and "REGRESSION" not in out
    else:
        assert status == 1 and "QUESTIONS CHANGED" in out and "--write-baseline" in out and "REGRESSION" not in out


def test_a_baseline_that_predates_the_question_hashes_asks_for_a_re_pin_instead_of_failing_unreadably():
    old = {k: v for k, v in BASELINE.items() if k not in ("questions_hash", "calibration_hash")}
    verdict = compare_to_baseline(_report_like(BASELINE), old)
    assert [line.split(":")[0] for line in verdict_lines(verdict)] == ["QUESTIONS CHANGED", "QUESTIONS CHANGED"]
