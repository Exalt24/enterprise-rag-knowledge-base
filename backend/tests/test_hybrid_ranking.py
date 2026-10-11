"""Regression test for the hybrid scoring bug fixed on 2026-08-03.

The collection uses cosine distance, so the vector store already returns a SIMILARITY (higher is better). The old code
ran 1/(1+score) over it, which inverted the ranking: a 0.82 match scored 0.549 and a 0.12 match scored 0.893, so
hybrid search ranked the worst chunks first. Nothing errored and every "results came back" test stayed green.

This test needs no live service: it feeds known scores through hybrid_search and asserts the order and the scale.
Run: pytest backend/tests/test_hybrid_ranking.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub(name, **attrs):
    """Register a stand-in module only when the real one is not importable (CI without the heavy dependencies)."""
    try:
        __import__(name)
    except Exception:
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


class _Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class _Any:
    """Accepts any construction, call, attribute or `|` chain: a stand-in for heavy libraries this test never uses."""

    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()

    def __or__(self, other):
        return _Any()


_stub("langchain_core")
_stub("langchain_core.documents", Document=_Document)
_stub("langchain_community")
_stub("langchain_community.retrievers", BM25Retriever=_Any())
_stub("langchain_core.prompts", ChatPromptTemplate=_Any())
_stub("langchain_core.output_parsers", StrOutputParser=_Any())
_stub("sentence_transformers", CrossEncoder=_Any())
_stub("langchain_ollama", OllamaLLM=_Any())
_stub("langchain_groq", ChatGroq=_Any())
_stub("app.services.vector_store", vector_store=object())
_stub("app.core.config", settings=types.SimpleNamespace(retrieval_top_k=3))

from app.services.advanced_retrieval import AdvancedRetrieval  # noqa: E402


class _FakeStore:
    """Returns cosine SIMILARITIES, the way the real collection does (higher is better)."""

    def __init__(self, scored):
        self._scored = scored

    def search_with_scores(self, query, k):
        return list(self._scored)


class _FakeBM25:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return list(self._docs)


def _engine(scored, bm25_docs=()):
    eng = AdvancedRetrieval.__new__(AdvancedRetrieval)       # skip __init__: it wires an LLM and the live store
    eng.vector_store = _FakeStore(scored)
    eng._get_bm25_retriever = lambda k: _FakeBM25(bm25_docs)
    return eng


def test_a_strong_match_outranks_a_poor_one():
    strong, poor = _Document("strong"), _Document("poor")
    eng = _engine([(poor, 0.12), (strong, 0.82)])            # fed in the WRONG order on purpose
    docs, scores = eng.hybrid_search("q", k=2, with_scores=True)
    assert [d.page_content for d in docs] == ["strong", "poor"]
    assert scores[0] > scores[1]


def test_the_vector_score_is_used_directly_not_inverted():
    # the bug turned 0.82 into 1/(1+0.82) = 0.549; the fix keeps it on the 0..1 cosine scale, weighted 70/30 by default
    doc = _Document("only")
    eng = _engine([(doc, 0.82)])
    _, scores = eng.hybrid_search("q", k=1, with_scores=True)
    assert abs(scores[0] - 0.82 * 0.7) < 1e-9


def test_a_negative_cosine_is_clamped_to_zero():
    doc = _Document("negative")
    eng = _engine([(doc, -0.4)])
    _, scores = eng.hybrid_search("q", k=1, with_scores=True)
    assert scores[0] == 0.0
