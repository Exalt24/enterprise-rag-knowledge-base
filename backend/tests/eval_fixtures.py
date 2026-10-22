"""Fixed fixtures for the evaluation tests: a complete report with literal numbers and the baseline pinned from it.

Tests that check how a verdict or a message READS use these, never the live corpus, so editing a corpus sentence cannot
break a test whose subject is wording. Not collected as a test file (no test_ prefix).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.harness import ANSWERABLE_METRICS, CATEGORIES, METHODS, baseline_view  # noqa: E402

CANNED_VALUE = 0.5


def _entry(value):
    return {"value": value, "lo": max(0.0, value - 0.25), "hi": min(1.0, value + 0.25)}


def canned_report(value=CANNED_VALUE, fallbacks=0, cause="", fallback_methods=()):
    """A report shaped exactly like run_evaluation's, every number `value`, so the verdict and the printing can be driven."""
    results = {}
    for method in METHODS:
        results[method] = {}
        for category in CATEGORIES:
            if category == "unanswerable":
                results[method][category] = {"n": 4, "abstain": _entry(value)}
            else:
                results[method][category] = {"n": 4, **{metric: _entry(value) for metric in ANSWERABLE_METRICS}}
    chance = {c: {m: 0.1 for m in ("hit@1", "hit@3", "mrr", "recall@3", "recall@5")} for c in CATEGORIES if c != "unanswerable"}
    return {
        "chunk_count": 20,
        "chunk_hash": "c" * 64,
        "questions_hash": "a" * 64,
        "calibration_hash": "b" * 64,
        "store_points": 20,
        "question_counts": {c: 4 for c in CATEGORIES},
        "calibration_count": 8,
        "depth": 5,
        "fallbacks": fallbacks,
        "fallback_cause": cause,
        "fallback_methods": list(fallback_methods),
        "thresholds": {m: 0.3 for m in METHODS},
        "results": results,
        "chance": chance,
    }


def canned_baseline():
    return baseline_view(canned_report())
