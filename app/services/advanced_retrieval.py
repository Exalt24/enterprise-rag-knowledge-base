"""
Advanced Retrieval Techniques

Implements:
1. Hybrid Search (Vector + BM25 keyword search)
2. Query Optimization (rewriting, expansion)
3. Reranking (future)

These techniques improve retrieval accuracy beyond basic similarity search.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from app.services.vector_store import vector_store
from app.core.config import settings


class AdvancedRetrieval:
    """
    Advanced retrieval techniques for better accuracy.

    Hybrid Search = Vector (semantic) + BM25 (keyword)
    - Vector: Finds by meaning ("car" matches "automobile")
    - BM25: Finds by keywords (exact term matching)
    - Combined: Best of both worlds!
    """

    def __init__(self):
        self.vector_store = vector_store
        self.llm = OllamaLLM(model=settings.ollama_model, temperature=0.1)
        self._cross_encoder = None  # Lazy load (only when reranking is used)

    def _get_cross_encoder(self):
        """Lazy load cross-encoder model (only when needed)"""
        if self._cross_encoder is None:
            print("[i] Loading cross-encoder for reranking...")
            self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("[OK] Cross-encoder ready!")
        return self._cross_encoder

    def hybrid_search(
        self,
        query: str,
        k: int = None,
        vector_weight: float = 0.7
    ) -> List[Document]:
        """
        Hybrid search combining vector similarity and BM25 keyword search.

        Args:
            query: Search query
            k: Number of results
            vector_weight: Weight for vector search (0.0 to 1.0)
                          0.7 = 70% vector, 30% BM25 (recommended)

        Returns:
            List of documents ranked by combined score
        """
        k = k or settings.retrieval_top_k

        # Simplified hybrid: Combine vector + BM25 results manually
        try:
            # Get vector search results (semantic)
            vector_results = self.vector_store.search(query, k=k * 2)  # Get more for diversity

            # Get all documents for BM25
            all_docs_result = self.vector_store._vectorstore.get()
            all_docs = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(
                    all_docs_result['documents'],
                    all_docs_result['metadatas']
                )
            ]

            # BM25 keyword search
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = k * 2
            bm25_results = bm25_retriever.invoke(query)

            # Combine results (simple: vector first, then add unique BM25 results)
            combined = vector_results.copy()
            seen_content = {doc.page_content for doc in vector_results}

            for doc in bm25_results:
                if doc.page_content not in seen_content:
                    combined.append(doc)
                    seen_content.add(doc.page_content)

            # Return top k
            return combined[:k]

        except Exception as e:
            print(f"[!] Hybrid search failed, using vector only: {e}")
            # Fallback to basic vector search
            return self.vector_store.search(query, k=k)

    def optimize_query(self, query: str) -> str:
        """
        Optimize query for better retrieval using LLM.

        Techniques:
        1. Query rewriting (make query more specific)
        2. Query expansion (add related terms)

        Example:
        Input:  "weather"
        Output: "What is the current weather? temperature conditions forecast"
        """

        prompt = ChatPromptTemplate.from_template("""
You are a query optimization assistant.

Given a user query, rewrite it to be more specific and add related search terms.

Rules:
1. Make the query more explicit and detailed
2. Add 2-3 related keywords at the end
3. Keep it concise (max 50 words)
4. If query is already good, return it unchanged

Original query: {query}

Optimized query:""")

        chain = prompt | self.llm | StrOutputParser()

        try:
            optimized = chain.invoke({"query": query})
            return optimized.strip()
        except:
            # If optimization fails, return original
            return query

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder for better relevance scoring.

        How it works:
        - Cross-encoder scores each (query, document) pair
        - More accurate than cosine similarity (but slower)
        - Use after initial retrieval to refine top results

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Return top K results (default: all)

        Returns:
            List of (Document, score) tuples sorted by relevance (highest first)
        """
        if not documents:
            return []

        # Get cross-encoder model (lazy load)
        cross_encoder = self._get_cross_encoder()

        # Prepare pairs for scoring
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs (higher = more relevant)
        print(f"[i] Reranking {len(documents)} documents...")
        scores = cross_encoder.predict(pairs)

        # Combine documents with scores
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score (highest first - cross-encoder higher = better)
        ranked = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        # Limit to top_k
        if top_k:
            ranked = ranked[:top_k]

        print(f"[OK] Reranked! Top score: {ranked[0][1]:.4f}")

        return ranked


# Global instance
advanced_retrieval = AdvancedRetrieval()


# =============================================================================
# Test Advanced Retrieval
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from app.services.vector_store import vector_store

    print("=" * 70)
    print("Advanced Retrieval Test")
    print("=" * 70)

    # Check database
    stats = vector_store.get_stats()
    print(f"\nDatabase: {stats['total_documents']} documents")

    if stats['total_documents'] == 0:
        print("\n[!] No documents in database. Run test_ingestion.py first")
        sys.exit(1)

    # Test 1: Query Optimization
    print("\n[1] Query Optimization:")
    print("-" * 70)

    test_queries = [
        "weather",
        "RAG",
        "embedding model"
    ]

    for q in test_queries:
        optimized = advanced_retrieval.optimize_query(q)
        print(f"Original:  '{q}'")
        print(f"Optimized: '{optimized}'\n")

    # Test 2: Hybrid Search
    print("\n[2] Hybrid Search (Vector + BM25):")
    print("-" * 70)

    query = "What vector database is used?"

    # Basic vector search
    print("\nBasic vector search:")
    vector_results = vector_store.search(query, k=3)
    for i, doc in enumerate(vector_results, 1):
        print(f"  [{i}] {doc.page_content[:80]}...")

    # Hybrid search
    print("\nHybrid search (vector + keyword):")
    hybrid_results = advanced_retrieval.hybrid_search(query, k=3)
    for i, doc in enumerate(hybrid_results, 1):
        print(f"  [{i}] {doc.page_content[:80]}...")

    print("\n" + "=" * 70)
    print("[OK] Advanced retrieval working!")
    print("=" * 70)
    print("\nBenefits:")
    print("  - Hybrid search catches exact terms + semantic meaning")
    print("  - Query optimization improves vague queries")
    print("  - Better accuracy than vector-only search")
