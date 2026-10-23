"""A count written in the docs must be the count a fresh run produces.

The README says how many chunks the corpus gives and how many questions there are, in total and per category. Those
numbers were typed by hand, so adding a question or editing a corpus file and re-pinning the baseline left the sentence
wrong with every test green. Now every counted quantity (chunks, documents, questions in total and per category,
calibration questions, the `n` column of both tables, the hybrid fallback count, the bootstrap resamples, the skill
scenarios) is derived from a fresh deterministic run (the hashing embedder: no model download, no network) and compared
with every sentence that states it in every markdown page of the repo (tests/doc_pages.py finds them: the root, docs/,
skills/ and anything they link), so a new page is read the day it is added.

Three things keep it from passing by accident:
1. A sentence that carries a number next to a counted noun this file does not read fails (`unread count`), so a new
   number cannot be added without a pattern and a row.
2. Each pattern is proved on a sample sentence, so a reworded pattern cannot quietly stop matching.
3. A number only a real-model run (or a CI run) can produce cannot be rerun here, so the sentence that quotes it must
   carry the date of that run.
Run: pytest tests/test_eval_readme_counts.py
"""
import itertools
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from doc_pages import OTHER_RUN_DATES, markdown_pages, sentences  # noqa: E402
from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    chunk_corpus,
    load_corpus,
    load_questions,
    run_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = tuple(markdown_pages())  # found by looking at the repo (doc_pages.py), so a page added tomorrow is scanned tomorrow
TEXT ={name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS}
SCENARIOS = ROOT / "skills" / "querying-knowledge-base" / "evaluations"

WORDS = "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
NUMBER = rf"(?<![\w.])(\d[\d,]*|{WORDS})"
WORD_VALUE = dict(zip(WORDS.split("|"), range(2, 21)))
CATEGORY_NAMES = ("direct", "paraphrase", "multi-document", "unanswerable")

PATTERNS = {
    "chunks": rf"(?<!top ){NUMBER}[ -]chunks\b",
    "documents": rf"{NUMBER} documents\b",
    "questions": rf"{NUMBER} (?:labelled )?questions\b",
    "calibration": rf"{NUMBER}(?:-question set| calibration questions)\b",
    "direct": rf"{NUMBER} direct\b",
    "paraphrase": rf"{NUMBER} paraphrase\b",
    "multi-document": rf"{NUMBER} multi-document\b",
    "unanswerable": rf"{NUMBER} unanswerable\b",
    "fallbacks": r"fallbacks in this run: (\d+)",
    "resamples": rf"{NUMBER} resamples\b",
    "scenarios": rf"{NUMBER} (?:evaluation )?scenarios\b",
}
# one sample sentence per pattern and the number it must read: a pattern that stops matching fails its own row
SAMPLES = {
    "chunks": ("They give 27 chunks through the chunker.", 27),
    "documents": ("The corpus is four documents invented here.", 4),
    "questions": ("The 42 questions in the file are labelled.", 42),
    "calibration": ("Calibrated on a separate 16-question set.", 16),
    "direct": ("They are 12 direct, 12 paraphrase and 8 multi-document.", 12),
    "paraphrase": ("They are 12 direct, 12 paraphrase and 8 multi-document.", 12),
    "multi-document": ("They are 12 direct, 12 paraphrase and 8 multi-document.", 8),
    "unanswerable": ("5 of the 10 unanswerable questions were declined.", 10),
    "fallbacks": ("Hybrid fallbacks in this run: 0.", 0),
    "resamples": ("Intervals use 2,000 resamples of the questions.", 2000),
    "scenarios": ("Six evaluation scenarios were written first.", 6),
}
# a number next to one of these nouns is a count this file must read
COUNTED_NOUN = r"(?:chunks|questions|documents|scenarios|resamples)"
UNREAD = re.compile(rf"(?<!top ){NUMBER}\b(?:[ -][\w-]+){{0,2}}[ -]{COUNTED_NOUN}\b")


def to_int(text):
    text = text.replace(",", "").lower()
    return int(text) if text.isdigit() else WORD_VALUE[text]


def counted_sentences(name):
    return [s for s in sentences(TEXT[name]) if not any(d in s for d in OTHER_RUN_DATES)]


def table_n_cells(name, table_heading, next_heading):
    """{category: [n values]} from the markdown table that sits between two bold headings of a file."""
    section = TEXT[name].split(table_heading, 1)[1].split(next_heading, 1)[0]
    found = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) > 3 and cells[2].isdigit() and cells[0] in ("vector", "bm25", "hybrid"):
            found.setdefault(cells[1], []).append(int(cells[2]))
    return found


@pytest.fixture(scope="module")
def truth():
    docs = load_corpus()
    chunks = chunk_corpus(docs)
    report = run_evaluation(HashingEmbeddings(), load_questions(QUESTIONS_PATH), load_questions(CALIBRATION_PATH), chunks)
    counts = report["question_counts"]
    return {
        "chunks": report["chunk_count"],
        "documents": len(docs),
        "questions": sum(counts.values()),
        "calibration": report["calibration_count"],
        **{category: counts[category] for category in CATEGORY_NAMES},
        "fallbacks": report["fallbacks"],
        "resamples": BOOTSTRAP_RESAMPLES,
        "scenarios": len(list(SCENARIOS.glob("*.md"))),
    }


def stated(source, quantity):
    """(where, value) for every time `source` states `quantity`."""
    if source.endswith("n column"):
        if quantity not in CATEGORY_NAMES:
            return []
        table = "**Table 1" if "Table 1" in source else "**Table 2"
        nxt = "**Table 2" if "Table 1" in source else "**Unanswerable"
        return [(f"{source} row {i + 1}", n) for i, n in enumerate(table_n_cells("README.md", table, nxt).get(quantity, []))]
    return [
        (sentence, to_int(m.group(1)))
        for sentence in counted_sentences(source.removesuffix(" prose"))
        for m in re.finditer(PATTERNS[quantity], sentence, flags=re.I)
    ]


# CLASS-TABLE: counted quantity x source that states it (prose of each doc, the n column of each README table); every statement equals the fresh run
QUANTITIES = ("chunks", "documents", "questions", "calibration", "direct", "paraphrase", "multi-document", "unanswerable", "fallbacks", "resamples", "scenarios")
SOURCES = (*(f"{name} prose" for name in DOCS), "README Table 1 n column", "README Table 2 n column")
COUNT_CASES = list(itertools.product(QUANTITIES, SOURCES))
# pairs that MUST state the quantity today, so a reworded sentence cannot make the check pass by saying nothing
MUST_STATE = {
    ("chunks", "README.md prose"), ("documents", "README.md prose"), ("questions", "README.md prose"),
    ("calibration", "README.md prose"), ("direct", "README.md prose"), ("paraphrase", "README.md prose"),
    ("multi-document", "README.md prose"), ("unanswerable", "README.md prose"), ("fallbacks", "README.md prose"),
    ("calibration", "docs/evaluation.md prose"), ("documents", "docs/evaluation.md prose"), ("resamples", "docs/evaluation.md prose"),
    ("direct", "README Table 1 n column"), ("paraphrase", "README Table 1 n column"), ("multi-document", "README Table 1 n column"),
    ("direct", "README Table 2 n column"), ("paraphrase", "README Table 2 n column"), ("multi-document", "README Table 2 n column"),
}


@pytest.mark.parametrize("quantity,source", COUNT_CASES, ids=lambda v: v)
def test_every_stated_count_is_the_fresh_run_count(quantity, source, truth):
    found = stated(source, quantity)
    if (quantity, source) in MUST_STATE:
        assert found, f"{source} no longer states the {quantity} count in a form this test reads: add the sentence back or update PATTERNS"
    for where, value in found:
        assert value == truth[quantity], (
            f"{source} says {value} for {quantity}, a fresh run gives {truth[quantity]} ({where!r}): "
            "re-pin, paste the --markdown table into Table 1, and fix the sentence"
        )


@pytest.mark.parametrize("quantity", QUANTITIES, ids=lambda v: v)
def test_each_pattern_reads_its_sample(quantity):
    sentence, expected = SAMPLES[quantity]
    assert [to_int(m.group(1)) for m in re.finditer(PATTERNS[quantity], sentence, flags=re.I)][:1] == [expected]


@pytest.mark.parametrize("name", DOCS, ids=lambda v: v)
def test_no_count_is_written_that_this_file_does_not_read(name):
    read = {}
    for quantity in QUANTITIES:
        for sentence in counted_sentences(name):
            for m in re.finditer(PATTERNS[quantity], sentence, flags=re.I):
                read.setdefault(sentence, set()).add(m.start(1))
    for sentence in counted_sentences(name):
        for m in UNREAD.finditer(sentence):
            assert m.start(1) in read.get(sentence, set()), (
                f"{name} states a count this test does not read: {m.group(0)!r} in {sentence[:120]!r}. "
                "Delete the number (say 'see the table' or point at the command) or add a pattern and a row to the CLASS-TABLE."
            )


# ---------------------------------------------------------------- a number only a real run can produce carries its date

DATE = re.compile(r"\b20\d\d-\d\d-\d\d\b")
REAL_RUN = re.compile(r"real model|real-model|Table 2", re.I)
CI_RUN = re.compile(r"\brun \d{8,}")
DATED_CASES = [("README.md", "a real-model figure", REAL_RUN, 3), ("README.md", "a CI run id", CI_RUN, 0), ("docs/evaluation.md", "a CI run id", CI_RUN, 1)]


@pytest.mark.parametrize("name,what,marker,at_least", DATED_CASES, ids=[f"{n}-{w}" for n, w, _, _ in DATED_CASES])
def test_a_sentence_quoting_a_run_that_cannot_be_repeated_here_carries_its_date(name, what, marker, at_least):
    hits = [s for s in sentences(TEXT[name]) if marker.search(s)]
    assert len(hits) >= at_least, f"{name} has {len(hits)} sentence(s) about {what}, expected at least {at_least}: the sentences moved, update DATED_CASES"
    for sentence in hits:
        assert DATE.search(sentence), f"{name}: this sentence quotes {what} and carries no run date (YYYY-MM-DD): {sentence[:140]!r}"
