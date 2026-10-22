"""The mutation check finds its targets in the real source.

eval/mutation_check.py breaks the guarded code in a temporary copy and demands a test failure. Running it takes about
a minute or more, so this test only proves the cheap half on every run: each mutation still finds exactly one place in the
current source and produces valid Python. If someone renames hybrid_search or moves the counter, this fails and says
which mutant is stale, instead of the check quietly guarding nothing.
Run: pytest tests/test_eval_mutation_targets.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from eval.mutation_check import MUTANTS, TargetNotFound, mutated_source  # noqa: E402


@pytest.mark.parametrize("mutant", MUTANTS, ids=lambda m: m.name)
def test_each_mutant_finds_one_target_and_yields_valid_python(mutant):
    try:
        source = mutated_source(mutant)
    except TargetNotFound as exc:
        pytest.fail(f"mutant {mutant.name} is stale: {exc}")
    compile(source, mutant.file, "exec")


def test_every_guarded_area_has_a_mutant():
    names = " ".join(m.name for m in MUTANTS)
    areas = ("bm25-branch", "bonus", "threshold", "must-contain", "fallback-counter", "fallback-cause", "baseline-rise",
             "baseline-question-hash", "baseline-fallen-back", "readme-table", "bootstrap-upper", "bootstrap-lower",
             "eval-chunking", "label-validation")
    for area in areas:
        assert area in names, f"no mutant covers {area}"
