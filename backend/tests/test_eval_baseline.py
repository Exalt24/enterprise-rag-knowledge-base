"""The pinned baseline for the deterministic run.

Three different outcomes, three different messages. "Chunking changed" means the text splitter (or the corpus) produced
other chunks, for example after a Dependabot bump. "Questions changed" means a question or a label was edited. Either way
the old numbers can no longer be compared. "Numbers fell" means retrieval got worse on the same chunks and the same
questions. Reading one as another sends a person looking in the wrong place. A number may rise freely.
Every change kind against every pinned field is generated in test_eval_baseline_changes.py.
Run: pytest tests/test_eval_baseline.py
"""
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import (  # noqa: E402
    BASELINE_PATH,
    BASELINE_TOLERANCE,
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    chunk_corpus,
    compare_to_baseline,
    load_corpus,
    load_questions,
    report_numbers,
    run_evaluation,
    verdict_lines,
)
from eval_fixtures import canned_baseline, canned_report  # noqa: E402

BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report():
    chunks = chunk_corpus(load_corpus())
    return run_evaluation(HashingEmbeddings(), load_questions(QUESTIONS_PATH), load_questions(CALIBRATION_PATH), chunks)


def _shape(numbers):
    return {(method, category, metric) for method, per in numbers.items() for category, ms in per.items() for metric in ms}


# ---------------------------------------------------------------- the real baseline against the real run


def test_chunking_has_not_changed_since_the_baseline_was_pinned(report):
    verdict = compare_to_baseline(report, BASELINE)
    assert not verdict.chunking, "\n".join(line for line in verdict_lines(verdict) if line.startswith("CHUNKING CHANGED"))


def test_the_questions_have_not_changed_since_the_baseline_was_pinned(report):
    verdict = compare_to_baseline(report, BASELINE)
    assert not verdict.questions, "\n".join(line for line in verdict_lines(verdict) if line.startswith("QUESTIONS CHANGED"))


def test_retrieval_numbers_have_not_fallen_below_the_baseline(report):
    verdict = compare_to_baseline(report, BASELINE)
    assert not verdict.regressions, "NUMBERS FELL BELOW THE BASELINE on unchanged chunks and questions: " + "; ".join(verdict.regressions)


def test_the_run_had_no_hybrid_fallback(report):
    assert report["fallbacks"] == 0, f"hybrid_search silently fell back to vector-only, so the hybrid rows are not hybrid: {report['fallback_cause']}"


def test_the_baseline_file_pins_what_the_spec_says_it_pins(report):
    assert BASELINE["tolerance"] == BASELINE_TOLERANCE == 0.02
    assert BASELINE["chunk_count"] == report["chunk_count"]
    assert set(BASELINE["numbers"]) == {"vector", "bm25", "hybrid"}
    # The same fields are pinned as the run produces. The values are NOT compared for equality: the docs say a number may
    # rise freely, so an improvement must not turn this test red (a fall is judged above, with the field and the amount).
    pinned, produced = _shape(BASELINE["numbers"]), _shape(report_numbers(report))
    assert pinned == produced, f"pinned but not produced: {sorted(pinned - produced)}; produced but not pinned: {sorted(produced - pinned)}"
    for method, per in BASELINE["numbers"].items():
        for category, metrics in per.items():
            for metric, value in metrics.items():
                assert 0.0 <= value <= 1.0, f"{method}/{category}/{metric} is pinned at {value}"
    for key in ("chunk_hash", "questions_hash", "calibration_hash"):
        assert len(BASELINE.get(key, "")) == 64, f"the baseline does not record {key}"


# ---------------------------------------------------------------- the comparison itself, on made-up reports


def _fake_report(count=10, chunk_hash="aa", value=0.8, **hashes):
    cell = {"n": 5, "hit@1": {"value": value, "lo": 0.0, "hi": 1.0}}
    return {
        "chunk_count": count,
        "chunk_hash": chunk_hash,
        "questions_hash": hashes.get("questions_hash", "qq"),
        "calibration_hash": hashes.get("calibration_hash", "cc"),
        "results": {"vector": {"direct": cell}},
    }


def _fake_baseline(count=10, chunk_hash="aa", value=0.8, tolerance=0.02):
    return {
        "chunk_count": count,
        "chunk_hash": chunk_hash,
        "questions_hash": "qq",
        "calibration_hash": "cc",
        "tolerance": tolerance,
        "numbers": {"vector": {"direct": {"hit@1": value}}},
    }


def test_equal_numbers_pass():
    assert compare_to_baseline(_fake_report(), _fake_baseline()).ok


def test_a_rise_passes():
    assert compare_to_baseline(_fake_report(value=0.95), _fake_baseline()).ok


def test_a_fall_inside_the_tolerance_passes():
    assert compare_to_baseline(_fake_report(value=0.781), _fake_baseline()).ok


def test_a_fall_beyond_the_tolerance_is_a_regression_with_its_own_message():
    verdict = compare_to_baseline(_fake_report(value=0.77), _fake_baseline())
    assert verdict.chunking == []
    assert len(verdict.regressions) == 1
    assert "vector/direct/hit@1 fell from 0.8 to 0.77, by 0.03, more than the tolerance of 0.02" in verdict.regressions[0]


def test_a_changed_chunk_count_is_reported_as_chunking_and_not_as_a_regression():
    verdict = compare_to_baseline(_fake_report(count=11, value=0.1), _fake_baseline())
    assert verdict.regressions == []  # the numbers are not comparable, so none are judged
    assert any("chunk count is 11" in line for line in verdict.chunking)


def test_a_changed_chunk_hash_alone_is_reported_as_chunking():
    verdict = compare_to_baseline(_fake_report(chunk_hash="bb", value=0.1), _fake_baseline())
    assert verdict.regressions == []
    assert any("hash of the chunk texts" in line for line in verdict.chunking)


def test_a_changed_question_file_is_reported_as_questions_before_any_number_is_judged():
    verdict = compare_to_baseline(_fake_report(questions_hash="zz", value=0.1), _fake_baseline())
    assert verdict.regressions == [] and verdict.chunking == []
    assert any(line.startswith("QUESTIONS CHANGED: questions.jsonl") for line in verdict_lines(verdict))


def test_a_changed_calibration_file_is_reported_as_questions_too():
    verdict = compare_to_baseline(_fake_report(calibration_hash="zz", value=0.1), _fake_baseline())
    assert verdict.regressions == []
    assert any(line.startswith("QUESTIONS CHANGED: calibration.jsonl") for line in verdict_lines(verdict))


def test_a_metric_missing_from_the_run_is_flagged():
    report = _fake_report()
    del report["results"]["vector"]["direct"]["hit@1"]
    verdict = compare_to_baseline(report, _fake_baseline())
    assert any("missing" in line for line in verdict.regressions)


def test_the_failure_kinds_never_share_a_message():
    both = compare_to_baseline(_fake_report(count=12, value=0.1), _fake_baseline())
    only_fall = compare_to_baseline(_fake_report(value=0.1), _fake_baseline())
    assert both.chunking and not both.regressions
    assert only_fall.regressions and not only_fall.chunking


# ---------------------------------------------------------------- the command line


def _pin(tmp_path, monkeypatch, baseline):
    import eval.run_eval as run_eval

    pinned = tmp_path / "baseline.json"
    pinned.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_PATH", pinned)
    return run_eval


def test_the_deterministic_run_is_checked_end_to_end_by_the_cli(capsys):
    import eval.run_eval as run_eval

    assert run_eval.main(["--check"]) == 0
    assert "baseline check: OK" in capsys.readouterr().out


def test_the_cli_fails_with_the_chunking_message_when_the_baseline_hash_differs(tmp_path, monkeypatch, capsys):
    changed = copy.deepcopy(BASELINE)
    changed["chunk_hash"] = "0" * 64
    run_eval = _pin(tmp_path, monkeypatch, changed)
    assert run_eval.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "CHUNKING CHANGED" in out and "REGRESSION" not in out


def test_the_cli_fails_with_the_questions_message_when_the_baseline_question_hash_differs(tmp_path, monkeypatch, capsys):
    changed = copy.deepcopy(BASELINE)
    changed["questions_hash"] = "0" * 64
    run_eval = _pin(tmp_path, monkeypatch, changed)
    assert run_eval.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "QUESTIONS CHANGED" in out and "re-pin with `python eval/run_eval.py --write-baseline`" in out
    assert "REGRESSION" not in out and "CHUNKING CHANGED" not in out


def test_the_cli_fails_with_the_regression_message_when_a_pinned_number_is_higher_than_the_run(tmp_path, monkeypatch, capsys):
    # A fixed fixture on purpose: this test is about how the message reads, so a punctuation edit in a corpus file
    # (which moves the live chunk hash) must not be able to break it.
    import eval.run_eval as run_eval

    report = canned_report()
    baseline = canned_baseline()
    baseline["numbers"]["hybrid"]["direct"]["hit@1"] = 1.0
    monkeypatch.setattr(run_eval, "run_evaluation", lambda *args, **kwargs: report)
    pinned = tmp_path / "baseline.json"
    pinned.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(run_eval, "BASELINE_PATH", pinned)
    assert run_eval.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION: hybrid/direct/hit@1 fell from 1.0 to 0.5, by 0.5, more than the tolerance of 0.02" in out
    assert "CHUNKING CHANGED" not in out and "QUESTIONS CHANGED" not in out
