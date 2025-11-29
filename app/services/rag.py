"""
RAG Service - The Complete System

Combines Retrieval + Generation into one service.

Flow:
1. User asks question
2. Retrieve relevant documents from vector DB
3. Format documents as context
4. Pass context + question to LLM
5. Return answer with sources

This is what users interact with!
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from app.services.retrieval import retrieval_service, RetrievalResult
from app.services.generation import generation_service, GenerationResponse


class RAGResponse(BaseModel):
    """
    Complete RAG response with Pydantic validation.

    Contains:
    - Generated answer
    - Source documents
    - Metadata for debugging/monitoring
    """
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    sources: List[Dict] = Field(..., description="Source documents with metadata")
    num_sources: int = Field(..., description="Number of sources used")
    model_used: str = Field(..., description="LLM model")
    retrieval_scores: List[float] = Field(default=[], description="Similarity scores")

    class Config:
        arbitrary_types_allowed = True


class RAGService:
    """
    Main RAG service - combines retrieval and generation.

    Usage:
        rag = RAGService()
        response = rag.query("What is RAG?")
        print(response.answer)
    """

    def __init__(self):
        self.retrieval = retrieval_service
        self.generation = generation_service

    def query(
        self,
        question: str,
        k: int = None,
        include_scores: bool = False
    ) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: User question
            k: Number of documents to retrieve
            include_scores: Include similarity scores in response

        Returns:
            RAGResponse with answer and sources
        """

        print(f"\n[i] Processing query: '{question}'")
        print("-" * 70)

        # Step 1: Retrieve relevant documents
        print("[1/3] Retrieving relevant documents...")
        retrieval_result = self.retrieval.retrieve(
            question,
            k=k,
            with_scores=include_scores
        )

        print(f"[OK] Retrieved {retrieval_result.num_results} documents")

        if retrieval_result.num_results == 0:
            # No documents found
            return RAGResponse(
                answer="I don't have any relevant information to answer this question. Please add documents to the knowledge base.",
                query=question,
                sources=[],
                num_sources=0,
                model_used="none",
                retrieval_scores=[]
            )

        # Step 2: Format context for LLM
        print("[2/3] Formatting context...")
        context = self.retrieval.format_context(retrieval_result.documents)
        print(f"[OK] Context length: {len(context)} chars")

        # Step 3: Generate answer
        print("[3/3] Generating answer...")
        gen_response = self.generation.generate(question, context)
        print(f"[OK] Generated answer ({len(gen_response.answer)} chars)")

        # Extract source info
        sources = []
        for doc in retrieval_result.documents:
            sources.append({
                "file_name": doc.metadata.get("file_name", "unknown"),
                "page": doc.metadata.get("page"),
                "content_preview": doc.page_content[:100] + "..."
            })

        return RAGResponse(
            answer=gen_response.answer,
            query=question,
            sources=sources,
            num_sources=len(sources),
            model_used=gen_response.model_used,
            retrieval_scores=retrieval_result.scores
        )


# Global instance
rag_service = RAGService()


# =============================================================================
# Test Complete RAG System
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from typing import Dict
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from app.services.vector_store import vector_store

    print("=" * 70)
    print("Complete RAG System Test")
    print("=" * 70)

    # Check if database has documents
    stats = vector_store.get_stats()
    print(f"\nDatabase: {stats['total_documents']} documents")

    if stats['total_documents'] == 0:
        print("\n[!] No documents in database!")
        print("Run: python test_ingestion.py first")
        sys.exit(1)

    # Test queries
    test_queries = [
        "What is RAG?",
        "What embedding model is used?",
        "What is the query latency?",
        "Does it support PDF files?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print("=" * 70)

        # Get RAG response
        response = rag_service.query(query, include_scores=True)

        # Display answer
        print(f"\nANSWER:")
        print(response.answer)

        # Display sources
        print(f"\nSOURCES ({response.num_sources}):")
        for i, source in enumerate(response.sources, 1):
            print(f"  [{i}] {source['file_name']}")
            if include_scores and i <= len(response.retrieval_scores):
                print(f"      Relevance: {response.retrieval_scores[i-1]:.4f}")

    print("\n" + "=" * 70)
    print("[OK] COMPLETE RAG SYSTEM WORKING!")
    print("=" * 70)
    print("\nYou now have:")
    print("  [+] Document ingestion (PDF, DOCX, TXT)")
    print("  [+] Semantic search (384-dim embeddings)")
    print("  [+] Answer generation (Llama 3)")
    print("  [+] Source attribution")
    print("  [+] End-to-end RAG pipeline!")
    print("\nNext steps:")
    print("  1. Test with your own documents")
    print("  2. Build FastAPI endpoints")
    print("  3. Create web dashboard")
