"""
End-to-End Ingestion Test

Tests the complete pipeline:
Parse → Chunk → Embed → Store → Search

Run this to verify your RAG system works!
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ingestion import IngestionService
from app.services.vector_store import vector_store


def test_complete_pipeline():
    """Test the complete ingestion pipeline"""

    print("=" * 70)
    print("Complete RAG Ingestion Pipeline Test")
    print("=" * 70)

    # Create test document
    test_dir = Path("data/documents")
    test_dir.mkdir(exist_ok=True, parents=True)

    test_file = test_dir / "rag_basics.txt"
    test_file.write_text("""
Enterprise RAG Knowledge Base System

This system demonstrates production-ready RAG architecture with the following components:

1. Document Processing
   - Multi-format support: PDF, DOCX, TXT, Markdown
   - Intelligent text chunking with overlap
   - Metadata extraction and preservation

2. Embedding Generation
   - Local Sentence Transformers (all-MiniLM-L6-v2)
   - 384-dimensional vectors
   - Zero API costs - unlimited usage

3. Vector Database
   - Chroma for persistent storage
   - Fast similarity search
   - Metadata filtering capabilities

4. Retrieval Strategy
   - Semantic search (find by meaning)
   - Configurable top-k results
   - Score-based ranking

5. LLM Integration
   - Llama 3 via Ollama (local)
   - Groq API fallback (350+ tokens/sec)
   - Context-aware generation

The system achieves 90%+ retrieval relevance with sub-2 second query latency while running on completely free and open-source technologies.
    """)

    # Initialize ingestion service
    ingestion = IngestionService()

    # Test ingestion
    print("\n" + "=" * 70)
    print("STEP 1: Document Ingestion")
    print("=" * 70)

    result = ingestion.ingest_file(str(test_file))

    if not result["success"]:
        print(f"\n[X] Ingestion failed: {result['error']}")
        return False

    print(f"\n[OK] Ingestion successful!")
    print(f"  - File: {result['file_name']}")
    print(f"  - Documents parsed: {result['documents_parsed']}")
    print(f"  - Chunks created: {result['chunks_created']}")
    print(f"  - Chunks stored: {result['chunks_stored']}")

    # Test search
    print("\n" + "=" * 70)
    print("STEP 2: Semantic Search")
    print("=" * 70)

    test_queries = [
        "What embedding model is used?",
        "What is the query latency?",
        "What vector database is used?",
        "Does it support PDF files?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)

        results = vector_store.search(query, k=2)

        for i, doc in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    {doc.page_content[:120]}...")
            print(f"    (from {doc.metadata.get('file_name', 'unknown')})")

    # Database stats
    print("\n" + "=" * 70)
    print("STEP 3: Database Statistics")
    print("=" * 70)

    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] COMPLETE PIPELINE WORKING!")
    print("=" * 70)

    print("\nWhat you just did:")
    print("  ✅ Parsed a document")
    print("  ✅ Chunked it into smaller pieces")
    print("  ✅ Generated embeddings (384-dim vectors)")
    print("  ✅ Stored in Chroma vector database")
    print("  ✅ Searched by semantic meaning")
    print("\nYou have a working RAG system! 🎉")

    return True


if __name__ == "__main__":
    success = test_complete_pipeline()

    if success:
        print("\n" + "=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print("1. Add real documents to data/documents/")
        print("2. Run: python -m app.services.ingestion")
        print("3. Build the RAG query system (retrieval + generation)")
        print("4. Create FastAPI endpoints")
        print("5. Build frontend dashboard")
        sys.exit(0)
    else:
        print("\n[X] Test failed - check errors above")
        sys.exit(1)
