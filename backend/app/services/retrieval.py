"""
Retrieval Service

Handles document retrieval from vector database with optimizations:
- Similarity search
- Relevance scoring
- Query optimization (optional: rewriting, expansion)

This is the "R" in RAG - getting the right context!
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from app.services.vector_store import vector_store
from app.core.config import settings


class RetrievalResult(BaseModel):
    """
    Structured retrieval result with Pydantic validation.

    Benefits:
    - Type-safe access to results
    - Easy to serialize to JSON
    - Clear contract for downstream services
    """

    query: str = Field(..., description="Original user query")
    documents: List[Document] = Field(..., description="Retrieved documents")
    scores: List[float] = Field(default=[], description="Similarity scores (optional)")
    num_results: int = Field(..., description="Number of results returned")

    class Config:
        arbitrary_types_allowed = True  # Allow LangChain Document objects


class RetrievalService:
    """
    Retrieves relevant documents from vector database.

    Implements:
    - Similarity Search
    - Optional: Relevance Scoring
    - Optional: Query Optimization
    """

    def __init__(self):
        self.vector_store = vector_store

    def retrieve(
        self, query: str, k: int = None, with_scores: bool = False
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User question
            k: Number of documents to retrieve (default from config)
            with_scores: Include similarity scores

        Returns:
            RetrievalResult with documents and metadata
        """
        k = k or settings.retrieval_top_k

        if with_scores:
            # Get results with similarity scores
            results_with_scores = self.vector_store.search_with_scores(query, k=k)

            documents = [doc for doc, score in results_with_scores]
            scores = [float(score) for doc, score in results_with_scores]

            return RetrievalResult(
                query=query,
                documents=documents,
                scores=scores,
                num_results=len(documents),
            )
        else:
            # Get results without scores (faster)
            documents = self.vector_store.search(query, k=k)

            return RetrievalResult(
                query=query, documents=documents, num_results=len(documents)
            )

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context string for LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted string with numbered sources
        """
        if not documents:
            return "No relevant documents found."

        context_parts = []

        for i, doc in enumerate(documents, 1):
            # Extract metadata
            source = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""

            # Format with source attribution
            context_parts.append(
                f"[Source {i}: {source}{page_info}]\n{doc.page_content}\n"
            )

        return "\n".join(context_parts)


# Global instance
retrieval_service = RetrievalService()


# =============================================================================
# Test Retrieval
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 70)
    print("Retrieval Service Test")
    print("=" * 70)

    # Make sure we have data (run test_ingestion.py first if needed)
    stats = vector_store.get_stats()

    print(f"\nDatabase has {stats['total_documents']} documents")

    if stats["total_documents"] == 0:
        print("\n[!] No documents in database!")
        print("Run: python test_ingestion.py first")
        sys.exit(1)

    # Test queries
    test_queries = [
        "What embedding model is used?",
        "What is the query latency?",
        "How does RAG work?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print("=" * 70)

        # Retrieve without scores
        result = retrieval_service.retrieve(query, k=2)

        print(f"Retrieved {result.num_results} documents:\n")

        for i, doc in enumerate(result.documents, 1):
            print(f"[{i}] {doc.page_content[:100]}...")
            print(f"    From: {doc.metadata.get('file_name', 'unknown')}")
            print()

        # Show formatted context
        print("Formatted context for LLM:")
        print("-" * 70)
        context = retrieval_service.format_context(result.documents)
        print(context[:300] + "...")

    # Test with scores
    print(f"\n{'='*70}")
    print("Retrieval with Similarity Scores")
    print("=" * 70)

    result_with_scores = retrieval_service.retrieve(
        "What vector database is used?", k=3, with_scores=True
    )

    for i, (doc, score) in enumerate(
        zip(result_with_scores.documents, result_with_scores.scores), 1
    ):
        print(f"\n[{i}] Score: {score:.4f} (lower = more similar)")
        print(f"    Content: {doc.page_content[:80]}...")

    print("\n" + "=" * 70)
    print("[OK] Retrieval working perfectly!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("  - Semantic search finds by meaning, not keywords")
    print("  - Lower scores = more similar")
    print("  - Context formatting prepares docs for LLM")
    print("  - Source attribution enables citations")
