"""What the README says must be what the evaluation measures.

A metric defined in code and not in the table that quotes it lets a reader take hit@3 = 1.00 as "found everything it
needs", when for a multi-document question it only means one of the needed chunks was found. The README tables therefore
carry recall for every row and say what a hit means, Table 1 is checked cell by cell against a fresh deterministic run,
and Table 2 (a local run with the real model that CI cannot repeat) is checked for shape and for the relations between
its own columns. The cases are generated from tables x categories x methods x columns.
Run: pytest tests/test_eval_readme_claims.py
"""
import itertools
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import (  # noqa: E402
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    TABLE_METRICS,
    chunk_corpus,
    load_corpus,
    load_questions,
    markdown_table,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
EVALUATION_DOC = (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
CELL = re.compile(r"^(\d\.\d\d) \((\d\.\d\d) to (\d\.\d\d)\)$")


def _section(start, end):
    return README.split(start, 1)[1].split(end, 1)[0]


def _parse(markdown):
    """header list and {(method, category): {column: cell text}} from a markdown table."""
    header, rows = None, {}
    for line in markdown.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None and cells[0] == "method":
            header = cells
        elif header is not None and len(cells) == len(header):
            rows[(cells[0], cells[1])] = dict(zip(header, cells))
    return header, rows


TABLES = {"Table 1": _parse(_section("**Table 1", "**Table 2")), "Table 2": _parse(_section("**Table 2", "**Unanswerable"))}


@pytest.fixture(scope="module")
def generated():
    chunks = chunk_corpus(load_corpus())
    report = run_evaluation(HashingEmbeddings(), load_questions(QUESTIONS_PATH), load_questions(CALIBRATION_PATH), chunks)
    return _parse(markdown_table(report, with_chance=True))


# CLASS-TABLE: README table x answerable category x method x column; every cell exists, reads as value (low to high), and Table 1 equals the run
README_TABLES = ("Table 1", "Table 2")
CATEGORIES = ("direct", "paraphrase", "multi-document")
METHODS = ("vector", "bm25", "hybrid")
COLUMNS = ("hit@1", "hit@3", "MRR", "recall@3", "recall@5")  # written out, not read from the code under test
CELL_CASES = list(itertools.product(README_TABLES, CATEGORIES, METHODS, COLUMNS))


@pytest.mark.parametrize("table,category,method,column", CELL_CASES, ids=lambda v: v)
def test_every_readme_cell_exists_and_table_1_equals_a_fresh_run(table, category, method, column, generated):
    header, rows = TABLES[table]
    assert column in header, f"{table} has no {column} column (columns: {header})"
    assert (method, category) in rows, f"{table} has no {method} / {category} row"
    text = rows[(method, category)][column]
    match = CELL.match(text)
    assert match, f"{table} {method} / {category} / {column} reads {text!r}, not 'value (low to high)'"
    value, low, high = (float(g) for g in match.groups())
    assert low <= value <= high
    if table == "Table 1":
        assert text == generated[1][(method, category)][column], f"Table 1 is out of date against a fresh run: {method} / {category} / {column}"


RELATION_CASES = list(itertools.product(README_TABLES, CATEGORIES, METHODS))


@pytest.mark.parametrize("table,category,method", RELATION_CASES, ids=lambda v: v)
def test_the_columns_of_one_row_agree_with_each_other(table, category, method):
    row = TABLES[table][1][(method, category)]
    value = {col: float(CELL.match(row[col]).group(1)) for col in COLUMNS}
    assert value["hit@1"] <= value["hit@3"] + 1e-9
    assert value["recall@3"] <= value["recall@5"] + 1e-9
    assert value["recall@3"] <= value["hit@3"] + 1e-9, "recall counts needed chunks found, so it can never exceed the share of questions with a hit"
    if category != "multi-document":
        # one needed chunk per question: a hit and a recall of one chunk are the same thing
        assert value["recall@3"] == value["hit@3"], f"{table} {method} / {category}: one need per question, so recall@3 must equal hit@3"


def test_table_1_has_a_chance_row_with_every_column_for_each_category(generated):
    header, rows = TABLES["Table 1"]
    for category in CATEGORIES:
        for column in COLUMNS:
            assert re.fullmatch(r"\d\.\d\d", rows[("chance", category)][column]), (category, column)
            assert rows[("chance", category)][column] == generated[1][("chance", category)][column]


def test_both_tables_have_the_same_columns():
    assert TABLES["Table 1"][0] == TABLES["Table 2"][0] == ["method", "category", "n"] + list(COLUMNS)


def test_the_code_prints_the_columns_the_readme_carries():
    assert [title for _, title in TABLE_METRICS] == list(COLUMNS)


# ---------------------------------------------------------------- the sentences the tables need


def test_the_readme_defines_a_multi_document_hit_next_to_the_tables():
    text = " ".join(_section("### Rerunnable evaluation", "**Unanswerable").split())
    assert "multi-document" in text and "any one needed chunk" in text
    assert "recall@3" in text and "how many of the needed chunks" in text


def test_the_readme_says_a_ceiling_cell_has_no_spread():
    text = " ".join(README.split())
    assert "ceiling" in text and "no spread" in text and "every resample" in text


NO_COUNTS = (r"\b\d+ (?:tests?|mutants?)\b", r"\b(?:ten|nine|eleven|twelve|thirteen|fourteen|fifteen) (?:tests|mutants|things)\b", r"\b\d+ of \d+ mutants\b", r"\b\d+ passed\b")
COUNT_CASES = list(itertools.product(("README.md", "docs/evaluation.md", "docs/mcp-server.md"), NO_COUNTS))


@pytest.mark.parametrize("name,pattern", COUNT_CASES, ids=lambda v: v)
def test_no_test_or_mutant_count_is_written_into_prose(name, pattern):
    text = (ROOT / name).read_text(encoding="utf-8")
    # a count in prose goes stale at the next test; the docs say how to print it instead
    assert not re.search(pattern, text), f"{name} writes a count that matches {pattern}: {re.search(pattern, text).group(0)!r}"


# ---------------------------------------------------------------- numbers the prose quotes from the tables

PROSE = " ".join(README.split())


def _value(table, method, category, column):
    return float(CELL.match(TABLES[table][1][(method, category)][column]).group(1))


def test_the_prose_recall_figures_are_the_table_2_cells():
    said = re.search(r"recall@3 is (\d\.\d\d) for vector, (\d\.\d\d) for BM25 and (\d\.\d\d) for hybrid", PROSE)
    assert said, "the sentence quoting recall@3 for the three methods moved; update this test with it"
    cells = [_value("Table 2", method, "multi-document", "recall@3") for method in ("vector", "bm25", "hybrid")]
    assert [float(g) for g in said.groups()] == cells, f"the prose says {said.groups()} but Table 2 has {cells}"


@pytest.mark.parametrize("table,method", list(itertools.product(README_TABLES, METHODS)), ids=lambda v: v)
def test_the_prose_hit_3_of_1_00_on_multi_document_is_in_both_tables(table, method):
    assert "hit@3 is 1.00 for all three methods in both tables" in PROSE, "the hit@3 sentence moved; update this test with it"
    assert _value(table, method, "multi-document", "hit@3") == 1.0, f"{table} {method} multi-document hit@3 is no longer 1.00, so the prose is wrong"


def test_the_declined_fraction_in_the_prose_is_its_own_ratio():
    said = re.search(r"(\d+) of the (\d+) unanswerable questions were declined \((\d\.\d\d), interval (\d\.\d\d) to (\d\.\d\d)\)", PROSE)
    assert said, "the sentence quoting the declined unanswerable questions moved; update this test with it"
    declined, total, value, low, high = int(said.group(1)), int(said.group(2)), *(float(g) for g in said.groups()[2:])
    assert round(declined / total, 2) == value and low <= value <= high
