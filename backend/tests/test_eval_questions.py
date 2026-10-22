"""The labelled question set and the synthetic corpus, checked against each other.

A label is only worth anything if the phrase it points at really is in the document it names, so these tests are what
make `must_contain` trustworthy. No live service, no model.
Run: pytest tests/test_eval_questions.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from eval.harness import (  # noqa: E402
    CATEGORIES,
    CALIBRATION_PATH,
    CORPUS_DIR,
    QUESTIONS_PATH,
    chunk_corpus,
    chunk_satisfies,
    load_corpus,
    load_questions,
    phrase_in_text,
)

QUESTIONS = load_questions(QUESTIONS_PATH, validate=False)  # loading must not hide the label tests; validation has its own tests
CALIBRATION = load_questions(CALIBRATION_PATH, validate=False)  # loading must not hide the label tests; validation has its own tests
DOCS = {d.metadata["file_name"]: d.page_content for d in load_corpus()}
CHUNKS = chunk_corpus(load_corpus())


def test_the_corpus_is_three_or_four_documents_of_250_to_450_words():
    assert 3 <= len(DOCS) <= 4, sorted(DOCS)
    for name, text in DOCS.items():
        words = len(text.split())
        assert 250 <= words <= 450, f"{name} has {words} words"


def test_every_corpus_document_says_it_is_synthetic():
    for name, text in DOCS.items():
        assert "Synthetic example document" in text, name


def test_the_corpus_names_no_link_or_address():
    # a cheap guard that nothing real slipped in; the documents are invented
    for name, text in DOCS.items():
        assert "http" not in text and "@" not in text and "www." not in text, name


def test_there_are_at_least_40_questions_across_the_four_categories():
    assert len(QUESTIONS) >= 40
    seen = {c: sum(1 for q in QUESTIONS if q.category == c) for c in CATEGORIES}
    assert set(q.category for q in QUESTIONS) == set(CATEGORIES)
    assert all(n >= 8 for n in seen.values()), seen


def test_question_ids_and_texts_are_unique():
    ids = [q.id for q in QUESTIONS]
    texts = [q.text.lower() for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert len(texts) == len(set(texts))


@pytest.mark.parametrize("q", [q for q in QUESTIONS + CALIBRATION if q.needs], ids=lambda q: q.id)
def test_every_must_contain_appears_in_exactly_the_document_named(q):
    for need in q.needs:
        assert need.document in DOCS, f"{q.id}: no such document {need.document}"
        holders = [name for name, text in DOCS.items() if need.phrase in text]
        assert holders == [need.document], f"{q.id}: phrase {need.phrase!r} found in {holders}, named {need.document}"
        assert DOCS[need.document].count(need.phrase) == 1, f"{q.id}: phrase is not unique inside its document"


@pytest.mark.parametrize("q", [q for q in QUESTIONS + CALIBRATION if q.needs], ids=lambda q: q.id)
def test_every_must_contain_survives_chunking_inside_a_chunk(q):
    # a phrase cut in half by the splitter would make the question unanswerable at chunk level
    for need in q.needs:
        assert any(chunk_satisfies(need, c) for c in CHUNKS), f"{q.id}: no chunk holds {need.phrase!r}"


def test_unanswerable_questions_carry_no_label_and_the_rest_carry_one():
    for q in QUESTIONS:
        if q.category == "unanswerable":
            assert q.needs == (), q.id
        else:
            assert len(q.needs) >= 1, q.id


def test_multi_document_questions_need_two_different_documents():
    multi = [q for q in QUESTIONS if q.category == "multi-document"]
    assert multi
    for q in multi:
        assert len({n.document for n in q.needs}) >= 2, q.id


def test_the_calibration_set_is_separate_from_the_eval_questions():
    assert not {q.text.lower() for q in QUESTIONS} & {q.text.lower() for q in CALIBRATION}
    assert not {q.id for q in QUESTIONS} & {q.id for q in CALIBRATION}
    assert any(q.category == "unanswerable" for q in CALIBRATION)
    assert any(q.category != "unanswerable" for q in CALIBRATION)


def test_the_corpus_folder_holds_only_the_documents_that_were_loaded():
    assert sorted(p.name for p in CORPUS_DIR.glob("*.md")) == sorted(DOCS)


def test_phrase_matching_accepts_a_present_phrase_and_rejects_the_rest():
    assert phrase_in_text("red when", "a package is red when it is late")
    assert not phrase_in_text("amber when", "a package is red when it is late")
    assert not phrase_in_text("", "any text at all")  # an empty label must never count as a hit
    assert not phrase_in_text("Red When", "a package is red when it is late")  # verbatim means case sensitive
