"""Labels: a bad one is named when the questions load, and an unanswerable question stays unanswerable.

A label that points at the wrong document used to survive loading and die later as `ValueError: k must be a
non-negative integer` from deep inside the chance baseline. Now every label is checked against the corpus as the file
loads and one LabelError names each bad question. Separately, an unanswerable question is only a fair test while the
corpus really cannot answer it, so the words that would answer each one are checked against every document.
No live service, no model. Run: pytest tests/test_eval_labels.py
"""
import itertools
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import (  # noqa: E402
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    LabelError,
    Need,
    Question,
    check_labels_against_chunks,
    chunk_corpus,
    load_corpus,
    load_questions,
    run_evaluation,
)

DOCS = {d.metadata["file_name"]: d.page_content for d in load_corpus()}
QUESTIONS = load_questions(QUESTIONS_PATH, validate=False)  # loading must not hide the label tests; validation has its own tests
CALIBRATION = load_questions(CALIBRATION_PATH, validate=False)  # loading must not hide the label tests; validation has its own tests
CHUNKS = chunk_corpus(load_corpus())

# ---------------------------------------------------------------- a bad label is named when the questions load

FAKE_DOCS = {"a.md": "Alpha holds the needle phrase. Another sentence.", "b.md": "Beta holds the thread phrase."}
ANSWERABLE = ("direct", "paraphrase", "multi-document", "answerable")


def _good_row(category):
    row = {"id": "x01", "category": category, "question": "What holds the needle?", "expected_document": "a.md", "must_contain": "needle phrase"}
    if category == "multi-document":
        row["also"] = [{"expected_document": "b.md", "must_contain": "thread phrase"}]
    return row


def _set(**changes):
    def apply(row):
        row.update(changes)
        return row

    return apply


FAULTS = {
    "wrong-document": (_set(expected_document="b.md"), "x01: the phrase 'needle phrase' is not in b.md (it is in a.md)"),
    "no-such-document": (_set(expected_document="zzz.md"), "x01: expected_document 'zzz.md' is not a file in the corpus"),
    "phrase-not-in-corpus": (_set(must_contain="not there at all"), "x01: the phrase 'not there at all' is not in a.md"),
    "empty-phrase": (_set(must_contain=""), "x01: must_contain is empty or missing for a.md"),
    "empty-document": (_set(expected_document=""), "x01: expected_document is empty or missing"),
    "unknown-category": (_set(category="easy"), "x01: category 'easy' is not one of"),
    "no-question-text": (_set(question=" "), "x01: question text is empty or missing"),
    "also-not-a-list": (_set(also="b.md"), "x01: also must be a list of objects"),
    "also-wrong-document": (_set(also=[{"expected_document": "a.md", "must_contain": "thread phrase"}]), "x01: the phrase 'thread phrase' is not in a.md (it is in b.md)"),
}

# CLASS-TABLE: category x kind of bad label; each load must raise LabelError naming the question id and what is wrong
# answerable is a legal category in the calibration set, so renaming it is not a fault
BAD_CASES = [c for c in itertools.product(ANSWERABLE, sorted(FAULTS)) if c != ("answerable", "unknown-category")]


def _load(tmp_path, rows):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for name, text in FAKE_DOCS.items():
        (corpus / name).write_text(text, encoding="utf-8")
    path = tmp_path / "q.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return load_questions(path, corpus_dir=corpus)


@pytest.mark.parametrize("category,fault", BAD_CASES, ids=lambda v: v)
def test_a_bad_label_raises_one_error_naming_the_question_and_the_fault(category, fault, tmp_path):
    mutate, expected = FAULTS[fault]
    with pytest.raises(LabelError) as caught:
        _load(tmp_path, [mutate(_good_row(category))])
    assert expected in str(caught.value), str(caught.value)


@pytest.mark.parametrize("category", ANSWERABLE)
def test_a_good_label_of_each_answerable_category_loads(category, tmp_path):
    (q,) = _load(tmp_path, [_good_row(category)])
    assert q.id == "x01" and len(q.needs) == (2 if category == "multi-document" else 1)


def test_an_unanswerable_question_with_a_label_is_refused_and_a_clean_one_loads(tmp_path):
    clean = {"id": "u01", "category": "unanswerable", "question": "Capital of Australia?", "expected_document": "", "must_contain": ""}
    assert _load(tmp_path, [clean])[0].needs == ()
    with pytest.raises(LabelError, match="u01: an unanswerable question must have no"):
        _load(tmp_path, [dict(clean, must_contain="needle phrase")])


def test_a_multi_document_question_without_a_second_need_is_refused(tmp_path):
    row = _good_row("multi-document")
    del row["also"]
    with pytest.raises(LabelError, match="x01: a multi-document question needs at least one entry in also"):
        _load(tmp_path, [row])


def test_a_repeated_id_a_missing_id_and_a_broken_line_are_each_named(tmp_path):
    good = _good_row("direct")
    with pytest.raises(LabelError) as caught:
        _load(tmp_path, [good, good, "{not json", dict(good, id="")])
    text = str(caught.value)
    assert "x01: id is used twice" in text and "line 3: not valid JSON" in text and "line 4: id is empty or missing" in text


def test_every_problem_is_listed_not_only_the_first(tmp_path):
    rows = [dict(_good_row("direct"), expected_document="b.md"), dict(_good_row("direct"), id="x02", must_contain="nope")]
    with pytest.raises(LabelError) as caught:
        _load(tmp_path, rows)
    assert "2 bad label(s)" in str(caught.value) and "x01:" in str(caught.value) and "x02:" in str(caught.value)


def test_the_shipped_question_files_pass_validation():
    assert load_questions(QUESTIONS_PATH) and load_questions(CALIBRATION_PATH)


def test_run_eval_names_a_wrong_label_instead_of_dying_inside_the_metrics(tmp_path, monkeypatch, capsys):
    import eval.run_eval as run_eval

    text = QUESTIONS_PATH.read_text(encoding="utf-8")
    right = '"expected_document": "risk-register-procedure.md", "must_contain": "20 to 25 is critical"'
    wrong = text.replace(right, '"expected_document": "monthly-cost-report-guide.md", "must_contain": "20 to 25 is critical"')
    assert wrong != text
    broken = tmp_path / "questions.jsonl"
    broken.write_text(wrong, encoding="utf-8")
    monkeypatch.setattr(run_eval, "QUESTIONS_PATH", broken)
    assert run_eval.main(["--check"]) == 2
    out = capsys.readouterr().out
    assert "LABEL ERROR" in out and "d02: the phrase '20 to 25 is critical' is not in monthly-cost-report-guide.md" in out


def test_a_phrase_that_no_chunk_holds_is_named_before_the_run_starts():
    cut = Question("z01", "direct", "q", (Need("a.md", "needle phrase"),))
    pieces = [Document(page_content="needle", metadata={"file_name": "a.md"}), Document(page_content="phrase", metadata={"file_name": "a.md"})]
    with pytest.raises(LabelError, match="z01: no chunk of a.md holds the phrase"):
        check_labels_against_chunks([cut], pieces)
    with pytest.raises(LabelError, match="z01"):
        run_evaluation(HashingEmbeddings(), [cut], CALIBRATION, CHUNKS)


# ---------------------------------------------------------------- an unanswerable question stays unanswerable

KEY_PATTERNS = {
    "u01": [r"canberra"],
    "u02": [r"annual leave"],
    "u03": [r"software"],
    "u04": [r"penalt"],
    "u05": [r"cedar.{0,200}sponsor", r"sponsor.{0,200}cedar"],
    "u06": [r"cedar.{0,200}budget", r"budget.{0,200}cedar"],
    "u07": [r"safety"],
    "u08": [r"sourdough"],
    "u09": [r"programme board.{0,80}\bmeets\b", r"\bmeets\b.{0,80}programme board"],
    "u10": [r"discount rate", r"net present value"],
    "c09": [r"airline"],
    "c10": [r"weather"],
    "c11": [r"password"],
    "c12": [r"holiday"],
    "c13": [r"football"],
    "c14": [r"vendor"],
    "c15": [r"floors"],
    "c16": [r"canteen"],
}
# CLASS-TABLE: unanswerable question x the words that would answer it x every corpus document; none may hold them
KEY_CASES = [(qid, pattern, doc) for (qid, patterns), doc in itertools.product(KEY_PATTERNS.items(), sorted(DOCS)) for pattern in patterns]


@pytest.mark.parametrize("qid,pattern,doc", KEY_CASES, ids=lambda v: str(v)[:40])
def test_no_document_holds_the_words_that_would_answer_an_unanswerable_question(qid, pattern, doc):
    assert not re.search(pattern, DOCS[doc].lower(), flags=re.S), f"{doc} now holds {pattern!r}, which would answer {qid}"


def test_every_unanswerable_question_has_key_words_to_guard_it():
    ids = {q.id for q in QUESTIONS + CALIBRATION if q.category == "unanswerable"}
    assert ids == set(KEY_PATTERNS), f"unanswerable questions without key words: {sorted(ids - set(KEY_PATTERNS))}; key words for no question: {sorted(set(KEY_PATTERNS) - ids)}"
