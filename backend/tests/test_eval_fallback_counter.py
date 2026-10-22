"""The hybrid fallback counter.

hybrid_search swallows every exception and answers with plain vector search, so a broken BM25 branch used to look exactly like
a working hybrid search. The counter makes that visible: the evaluation fails when it is above zero.

No live service. Run in the eval invocation (not together with test_hybrid_ranking.py, whose stubs are process wide):
    pytest tests/test_eval_fallback_counter.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")  # skips the torch cross-encoder import, as the CI job does
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from app.services.advanced_retrieval import AdvancedRetrieval  # noqa: E402


class _Store:
    def search_with_scores(self, query, k):
        return [(Document(page_content="a"), 0.9), (Document(page_content="b"), 0.4)][:k]

    def search(self, query, k):
        return [Document(page_content="a"), Document(page_content="b")][:k]


class _Bm25:
    def invoke(self, query):
        return [Document(page_content="b")]


def _engine(bm25_factory):
    eng = AdvancedRetrieval.__new__(AdvancedRetrieval)  # skip __init__: it wires an LLM and the live store
    eng.vector_store = _Store()
    eng._get_bm25_retriever = bm25_factory
    return eng


def _broken_bm25(k):
    raise RuntimeError("bm25 index could not be built")


def test_a_new_engine_starts_at_zero():
    assert _engine(lambda k: _Bm25()).hybrid_fallback_count == 0


def test_a_clean_hybrid_search_is_not_counted():
    eng = _engine(lambda k: _Bm25())
    eng.hybrid_search("q", k=2)
    assert eng.hybrid_fallback_count == 0


def test_a_broken_bm25_branch_is_counted_once_per_search_and_still_returns_vector_results():
    eng = _engine(_broken_bm25)
    docs = eng.hybrid_search("q", k=2)
    assert [d.page_content for d in docs] == ["a", "b"]  # the silent degradation the counter exists to expose
    assert eng.hybrid_fallback_count == 1
    eng.hybrid_search("q", k=2, with_scores=True)
    assert eng.hybrid_fallback_count == 2


def test_the_count_belongs_to_one_engine_not_to_the_class():
    broken, healthy = _engine(_broken_bm25), _engine(lambda k: _Bm25())
    broken.hybrid_search("q", k=2)
    assert healthy.hybrid_fallback_count == 0
    assert AdvancedRetrieval.hybrid_fallback_count == 0
