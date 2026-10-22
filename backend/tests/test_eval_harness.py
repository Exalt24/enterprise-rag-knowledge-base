"""The evaluation harness: embedder, wiring into the repo's real store, retrieval modes, metrics, thresholds.

These tests exist to go red when the harness or the ranking code is broken (eval/mutation_check.py proves they do).
No network, no model download, no running Qdrant: the store is QdrantClient(":memory:") and the embedder is hashing.
Run in its own invocation, never together with test_hybrid_ranking.py (its stubs are process wide):
    pytest tests/test_eval_harness.py
"""
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

import eval.harness as harness  # noqa: E402
from eval.embedders import HashingEmbeddings, tokenize  # noqa: E402
from eval.harness import (  # noqa: E402
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    Need,
    Question,
    Retrieved,
    Retrievers,
    abstains,
    bootstrap_interval,
    calibrate_threshold,
    chance_hit,
    chance_mrr,
    chunk_corpus,
    in_memory_store,
    load_corpus,
    load_questions,
    question_metrics,
    run_evaluation,
)
from app.services.vector_store import vector_store  # noqa: E402

EMB = HashingEmbeddings()
CHUNKS = chunk_corpus(load_corpus())
QUESTIONS = load_questions(QUESTIONS_PATH)
CALIBRATION = load_questions(CALIBRATION_PATH)


# ---------------------------------------------------------------- embedder


def test_the_hashing_embedder_subclasses_langchain_embeddings():
    # QdrantVectorStore rejects a duck type with a TypeError, so this is the property that matters
    assert isinstance(EMB, Embeddings)


def test_the_hashing_embedder_is_deterministic_normalised_and_fixed_size():
    a, b = EMB.embed_query("the risk register"), HashingEmbeddings().embed_query("the risk register")
    assert a == b
    assert len(a) == 384
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_the_hashing_embedder_ranks_shared_words_above_unrelated_ones():
    q = EMB.embed_query("critical risks reviewed every week")
    near = EMB.embed_documents(["Critical risks are reviewed every week."])[0]
    far = EMB.embed_documents(["Handover took six weeks of asset register work."])[0]
    cosine = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    assert cosine(q, near) > cosine(q, far)


def test_a_text_of_only_stopwords_still_gets_a_valid_unit_vector():
    v = EMB.embed_query("what is the")
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9)


def test_tokenizer_lowercases_drops_stopwords_and_trims_plurals():
    assert tokenize("The Risks are Reviewed") == ["risk", "reviewed"]


# ---------------------------------------------------------------- wiring into the repo's real store


def test_ingestion_goes_through_the_real_vector_store_service():
    before = (vector_store._client, vector_store._vectorstore)
    with in_memory_store(EMB, CHUNKS) as service:
        assert service is vector_store
        assert service.get_stats()["total_documents"] == len(CHUNKS)  # real get_stats over the in-memory client
        assert len(service.get_all_documents()) == len(CHUNKS)  # real scroll code
        scored = service.search_with_scores("Who owns the programme risk register?", k=3)
        assert len(scored) == 3 and scored[0][1] >= scored[1][1] >= scored[2][1]
    assert (vector_store._client, vector_store._vectorstore) == before  # the singleton is put back


def test_chunk_ids_are_stable_between_runs():
    again = chunk_corpus(load_corpus())
    assert [c.id for c in CHUNKS] == [c.id for c in again]
    assert len({c.id for c in CHUNKS}) == len(CHUNKS)


# ---------------------------------------------------------------- the three retrieval modes


@pytest.fixture(scope="module")
def retrievers():
    with in_memory_store(EMB, CHUNKS):
        yield Retrievers()


def test_each_mode_puts_the_chunk_that_quotes_the_question_first(retrievers):
    query = "Critical risks are reviewed every week."
    for mode in ("vector", "bm25", "hybrid"):
        got = retrievers.by_name(mode)(query)
        assert "Critical risks are reviewed every week." in got.chunks[0].page_content, mode
        assert len(got.chunks) <= harness.RETRIEVAL_DEPTH


def test_hybrid_scores_are_the_weighted_cosine_plus_the_bm25_bonus(retrievers):
    # Independent restatement of the documented formula: 0.7 * cosine, plus a flat 0.15 when BM25 also returns the chunk.
    query = "Who owns the programme risk register?"
    k = harness.RETRIEVAL_DEPTH
    docs, scores = retrievers.engine.hybrid_search(query, k=k, with_scores=True)
    cosine = {d.page_content: s for d, s in vector_store.search_with_scores(query, k=k * 2)}
    bm25 = {d.page_content for d in retrievers.engine._get_bm25_retriever(k * 2).invoke(query)}
    assert bm25, "BM25 returned nothing, so this test would prove nothing"
    for doc, score in zip(docs, scores):
        expected = 0.7 * max(0.0, min(1.0, cosine.get(doc.page_content, 0.0))) + (0.15 if doc.page_content in bm25 else 0.0)
        assert score == pytest.approx(expected, abs=1e-9)
    assert scores == sorted(scores, reverse=True)


def test_bm25_can_pull_in_a_chunk_that_vector_search_did_not_rank(retrievers):
    # With k=1 each side sees few chunks; the BM25 bonus must be able to change what hybrid returns.
    query = "Who owns the programme risk register?"
    bm25_docs = retrievers.engine._get_bm25_retriever(2).invoke(query)
    hybrid_docs, scores = retrievers.engine.hybrid_search(query, k=1, with_scores=True)
    assert hybrid_docs[0].page_content in {d.page_content for d in bm25_docs}
    assert scores[0] >= 0.15  # nothing reaches the top without at least the keyword bonus or a strong cosine


def test_the_clean_run_reports_zero_hybrid_fallbacks(retrievers):
    for q in QUESTIONS[:5]:
        retrievers.hybrid(q.text)
    assert retrievers.fallbacks == 0


# ---------------------------------------------------------------- fallback makes the run fail


def _break_bm25_for_hybrid_only(monkeypatch):
    """Break the BM25 index builder, but let the BM25-only mode skip it: only hybrid_search swallows such an error."""

    def boom(self, k):
        raise RuntimeError("bm25 index could not be built")

    monkeypatch.setattr("app.services.advanced_retrieval.AdvancedRetrieval._get_bm25_retriever", boom)
    monkeypatch.setattr(Retrievers, "bm25", lambda self, query: Retrieved([], 0.0))


def test_a_broken_bm25_branch_is_counted_by_the_harness(monkeypatch):
    _break_bm25_for_hybrid_only(monkeypatch)
    report = run_evaluation(EMB, QUESTIONS[:6], CALIBRATION, CHUNKS)
    assert report["fallbacks"] > 0


def test_run_eval_exits_nonzero_when_a_fallback_happened(monkeypatch, capsys):
    import eval.run_eval as run_eval

    _break_bm25_for_hybrid_only(monkeypatch)
    assert run_eval.main([]) == 1
    assert "FAIL: hybrid_search fell back" in capsys.readouterr().out


def test_run_eval_refuses_to_pin_a_baseline_from_a_run_with_fallbacks(monkeypatch, tmp_path):
    import eval.run_eval as run_eval

    target = tmp_path / "baseline.json"
    monkeypatch.setattr(run_eval, "BASELINE_PATH", target)
    _break_bm25_for_hybrid_only(monkeypatch)
    assert run_eval.main(["--write-baseline"]) == 1
    assert not target.exists()


# ---------------------------------------------------------------- metrics


def _chunk(file_name, text):
    return Document(page_content=text, metadata={"file_name": file_name})


def test_a_first_rank_hit_scores_one_on_every_rank_metric():
    q = Question("x", "direct", "q", (Need("a.md", "needle"),))
    got = Retrieved([_chunk("a.md", "a needle here"), _chunk("a.md", "hay")], top_score=0.9)
    m = question_metrics(q, got, threshold=0.5)
    assert (m["hit@1"], m["hit@3"], m["mrr"], m["recall@3"], m["recall@5"], m["kept"]) == (1, 1, 1, 1, 1, 1)


def test_a_rank_four_hit_misses_hit_at_three_and_scores_a_quarter():
    q = Question("x", "direct", "q", (Need("a.md", "needle"),))
    got = Retrieved([_chunk("a.md", "h")] * 3 + [_chunk("a.md", "needle")], top_score=0.9)
    m = question_metrics(q, got, threshold=0.5)
    assert (m["hit@1"], m["hit@3"], m["recall@3"], m["recall@5"]) == (0, 0, 0, 1)
    assert m["mrr"] == 0.25


def test_a_phrase_in_the_wrong_document_does_not_count():
    q = Question("x", "direct", "q", (Need("a.md", "needle"),))
    got = Retrieved([_chunk("b.md", "a needle here")], top_score=0.9)
    assert question_metrics(q, got, threshold=0.5)["hit@1"] == 0


def test_a_multi_document_question_scores_recall_by_the_share_of_needs_found():
    q = Question("x", "multi-document", "q", (Need("a.md", "one"), Need("b.md", "two")))
    got = Retrieved([_chunk("a.md", "one"), _chunk("c.md", "x"), _chunk("c.md", "y"), _chunk("b.md", "two")], top_score=0.9)
    m = question_metrics(q, got, threshold=0.5)
    assert m["hit@1"] == 1 and m["recall@3"] == 0.5 and m["recall@5"] == 1.0


def test_an_unanswerable_question_is_declined_only_below_the_threshold():
    q = Question("x", "unanswerable", "q", ())
    assert question_metrics(q, Retrieved([], 0.1), 0.3) == {"abstain": 1.0}
    assert question_metrics(q, Retrieved([], 0.6), 0.3) == {"abstain": 0.0}


def test_abstains_is_a_strict_less_than():
    assert abstains(0.2, 0.3)
    assert not abstains(0.3, 0.3)
    assert not abstains(0.4, 0.3)


def test_a_declined_answerable_question_is_not_kept():
    q = Question("x", "direct", "q", (Need("a.md", "needle"),))
    got = Retrieved([_chunk("a.md", "needle")], top_score=0.1)
    assert question_metrics(q, got, threshold=0.3)["kept"] == 0.0


def test_chance_baselines_match_hand_computed_values():
    assert chance_hit(10, 1, 1) == pytest.approx(0.1)
    assert chance_hit(10, 1, 3) == pytest.approx(0.3)
    assert chance_hit(10, 10, 1) == pytest.approx(1.0)
    assert chance_mrr(2, 1, 5) == pytest.approx(0.75)  # half the time rank 1, half rank 2


def test_bootstrap_is_seeded_contains_the_mean_and_is_degenerate_when_all_equal():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    lo, hi = bootstrap_interval(values, random.Random(1))
    assert (lo, hi) == bootstrap_interval(values, random.Random(1))
    assert lo <= sum(values) / len(values) <= hi
    assert lo < hi
    assert bootstrap_interval([1.0] * 6, random.Random(1)) == (1.0, 1.0)


# ---------------------------------------------------------------- threshold


def test_calibration_puts_the_threshold_between_separable_scores():
    t = calibrate_threshold([0.8, 0.9, 0.7], [0.1, 0.2, 0.3])
    assert 0.3 < t < 0.7
    assert all(not abstains(s, t) for s in (0.8, 0.9, 0.7))
    assert all(abstains(s, t) for s in (0.1, 0.2, 0.3))


def test_calibration_with_overlapping_scores_picks_the_best_balanced_split():
    answerable = [0.9, 0.8, 0.4]
    unanswerable = [0.5, 0.2, 0.1]
    t = calibrate_threshold(answerable, unanswerable)
    kept = sum(1 for s in answerable if not abstains(s, t)) / 3
    declined = sum(1 for s in unanswerable if abstains(s, t)) / 3
    assert (kept + declined) / 2 == pytest.approx(5 / 6)  # no threshold on these scores does better


def test_the_reported_thresholds_come_from_the_separate_calibration_set():
    report = run_evaluation(EMB, QUESTIONS[:4], CALIBRATION, CHUNKS)
    assert report["calibration_count"] == len(CALIBRATION)
    assert set(report["thresholds"]) == {"vector", "bm25", "hybrid"}
    assert all(t > 0 for t in report["thresholds"].values())


# ---------------------------------------------------------------- whole run


@pytest.fixture(scope="module")
def report():
    return run_evaluation(EMB, QUESTIONS, CALIBRATION, CHUNKS)


def test_two_runs_give_identical_numbers(report):
    assert run_evaluation(EMB, QUESTIONS, CALIBRATION, CHUNKS) == report


def test_the_report_covers_every_method_and_category_with_intervals(report):
    assert set(report["results"]) == {"vector", "bm25", "hybrid"}
    for method, per_category in report["results"].items():
        assert set(per_category) == set(harness.CATEGORIES)
        for category, cell in per_category.items():
            for metric, entry in cell.items():
                if metric == "n":
                    continue
                assert entry["lo"] <= entry["value"] <= entry["hi"], (method, category, metric)


def test_retrieval_beats_chance_on_direct_questions_for_every_method(report):
    # a weak harness-sanity bound: even a bag-of-words ranking must beat a random shuffle of 27 chunks
    for method in ("vector", "bm25", "hybrid"):
        assert report["results"][method]["direct"]["hit@3"]["value"] > report["chance"]["direct"]["hit@3"] * 2


def test_the_chance_baseline_is_reported_for_each_answerable_category(report):
    assert set(report["chance"]) == {"direct", "paraphrase", "multi-document"}
    assert 0 < report["chance"]["direct"]["hit@1"] < 0.2


def test_the_text_report_prints_the_caveats_the_readme_repeats(report):
    text = harness.format_report(report, "TITLE")
    assert "chance baseline" in text and "calibrated on" in text and "hybrid fallbacks" in text
    assert "intervals are wide" in text
