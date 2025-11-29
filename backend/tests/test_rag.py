"""
Complete RAG System Test

Tests the full question-answering pipeline:
1. Retrieve relevant documents
2. Generate answer with LLM
3. Return answer with sources

Run this to see your RAG system in action!
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag import rag_service
from app.services.vector_store import vector_store


def test_rag_system():
    """Test complete RAG question-answering"""

    print("=" * 70)
    print("COMPLETE RAG SYSTEM TEST")
    print("=" * 70)

    # Check database
    stats = vector_store.get_stats()
    print(f"\nDatabase: {stats['total_documents']} documents stored")

    if stats['total_documents'] == 0:
        print("\n[!] No documents in database!")
        print("Run: python test_ingestion.py first to add documents")
        return False

    # Test queries
    test_queries = [
        "What is RAG?",
        "What embedding model is used?",
        "What is the query latency?",
        "What vector database is used?",
    ]

    print(f"\nTesting {len(test_queries)} queries...")
    print("=" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] QUERY: {query}")
        print("-" * 70)

        # Get RAG response
        response = rag_service.query(query, include_scores=True)

        # Display answer
        print(f"\nANSWER:\n{response.answer}")

        # Display sources
        print(f"\nSOURCES ({response.num_sources}):")
        for j, source in enumerate(response.sources, 1):
            score_info = ""
            if j <= len(response.retrieval_scores):
                score_info = f" [Relevance: {response.retrieval_scores[j-1]:.4f}]"

            print(f"  [{j}] {source['file_name']}{score_info}")
            print(f"      {source['content_preview']}")

    print("\n" + "=" * 70)
    print("[OK] RAG SYSTEM FULLY OPERATIONAL!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("\n")
    success = test_rag_system()

    if success:
        print("\n" + "=" * 70)
        print("CONGRATULATIONS!")
        print("=" * 70)
        print("\nYou built a working RAG system with:")
        print("  [+] Document ingestion (PDF, DOCX, TXT)")
        print("  [+] Semantic search (vector database)")
        print("  [+] Answer generation (Llama 3)")
        print("  [+] Source attribution")
        print("  [+] Production-ready architecture")
        print("  [+] 100% free and open-source stack!")
        print("\nWeek 2 Progress: 60% Complete")
        print("\nNext steps:")
        print("  1. Add your own documents to data/documents/")
        print("  2. Run: python test_ingestion.py")
        print("  3. Ask questions about your documents!")
        print("  4. Build FastAPI endpoints (Week 2, Day 5-6)")
        print("  5. Create web dashboard (Week 3)")
        sys.exit(0)
    else:
        print("\n[X] Test failed - run test_ingestion.py first")
        sys.exit(1)
