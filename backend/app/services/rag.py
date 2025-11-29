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
from app.services.advanced_retrieval import advanced_retrieval


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
        self.advanced = advanced_retrieval

    def query(
        self,
        question: str,
        k: int = None,
        include_scores: bool = False,
        use_hybrid_search: bool = True,
        optimize_query: bool = False,
        use_reranking: bool = False
    ) -> RAGResponse:
        """
        Answer a question using RAG with advanced retrieval options.

        Args:
            question: User question
            k: Number of documents to retrieve
            include_scores: Include similarity scores in response
            use_hybrid_search: Use hybrid (vector + BM25) vs basic vector search
            optimize_query: Optimize query before retrieval (improves vague queries)
            use_reranking: Rerank results with cross-encoder (most accurate, slower)

        Returns:
            RAGResponse with answer and sources
        """

        print(f"\n[i] Processing query: '{question}'")
        print("-" * 70)

        # Step 0: Optimize query if requested
        search_query = question
        if optimize_query:
            print("[0/4] Optimizing query...")
            search_query = self.advanced.optimize_query(question)
            print(f"[OK] Optimized: '{search_query}'")

        # Step 1: Retrieve relevant documents
        print(f"[1/{4 if optimize_query else 3}] Retrieving relevant documents...")

        if use_hybrid_search:
            # Use hybrid search (vector + BM25)
            documents = self.advanced.hybrid_search(search_query, k=k)
            retrieval_result = type('obj', (object,), {
                'documents': documents,
                'num_results': len(documents),
                'scores': []
            })()
        else:
            # Use basic vector search
            retrieval_result = self.retrieval.retrieve(
                search_query,
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

        # Step 1.5: Reranking (optional)
        final_documents = retrieval_result.documents
        rerank_scores = []

        if use_reranking:
            step_num = 2 if not optimize_query else 2
            total_steps = 4 if not optimize_query and not use_reranking else 5
            print(f"[{step_num}/{total_steps}] Reranking with cross-encoder...")

            reranked = self.advanced.rerank(search_query, retrieval_result.documents, top_k=k)
            final_documents = [doc for doc, score in reranked]
            rerank_scores = [float(score) for doc, score in reranked]
            step_num += 1
        else:
            step_num = 2 if not optimize_query else 2
            total_steps = 3 if not optimize_query else 4

        # Step 2: Format context for LLM
        print(f"[{step_num}/{total_steps}] Formatting context...")
        context = self.retrieval.format_context(final_documents)
        print(f"[OK] Context length: {len(context)} chars")

        # Step 3: Generate answer
        step_num += 1
        print(f"[{step_num}/{total_steps}] Generating answer...")
        gen_response = self.generation.generate(question, context)
        print(f"[OK] Generated answer ({len(gen_response.answer)} chars)")

        # Extract source info
        sources = []
        for doc in final_documents:
            sources.append({
                "file_name": doc.metadata.get("file_name", "unknown"),
                "page": doc.metadata.get("page"),
                "content_preview": doc.page_content[:100] + "..."
            })

        # Use rerank scores if available, otherwise use retrieval scores
        final_scores = rerank_scores if rerank_scores else retrieval_result.scores

        return RAGResponse(
            answer=gen_response.answer,
            query=question,
            sources=sources,
            num_sources=len(sources),
            model_used=gen_response.model_used,
            retrieval_scores=final_scores
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
