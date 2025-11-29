"""
Document Ingestion Pipeline

Orchestrates the complete ingestion flow:
1. Parse document (PDF/DOCX/TXT)
2. Chunk text (500 tokens with overlap)
3. Generate embeddings (Sentence Transformers)
4. Store in vector database (Chroma)

This is the main service you'll use to add documents to your RAG system.
"""

from typing import List, Dict, Any
from pathlib import Path
from app.services.document_parser import DocumentParser
from app.services.chunking import TextChunker
from app.services.vector_store import vector_store


class IngestionService:
    """
    High-level document ingestion service.

    Usage:
        ingestion = IngestionService()
        result = ingestion.ingest_file("path/to/document.pdf")
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.vector_store = vector_store

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file into the knowledge base.

        Args:
            file_path: Path to document file

        Returns:
            Dict with ingestion results and stats
        """
        print(f"\n[i] Ingesting: {Path(file_path).name}")
        print("-" * 70)

        # Step 1: Parse document
        print("[1/4] Parsing document...")
        try:
            documents = self.parser.parse(file_path)
            print(f"[OK] Extracted {len(documents)} document(s)")
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file": file_path
            }

        # Step 2: Chunk documents
        print("[2/4] Chunking text...")
        chunked_docs = self.chunker.chunk_documents(documents)
        print(f"[OK] Created {len(chunked_docs)} chunks")

        # Step 3: Add to vector database (embeddings generated automatically)
        print("[3/4] Generating embeddings & storing...")
        try:
            doc_ids = self.vector_store.add_documents(chunked_docs)
            print(f"[OK] Stored {len(doc_ids)} chunks")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to store: {e}",
                "file": file_path
            }

        # Step 4: Return results
        print("[4/4] Done!")

        return {
            "success": True,
            "file": file_path,
            "file_name": Path(file_path).name,
            "documents_parsed": len(documents),
            "chunks_created": len(chunked_docs),
            "chunks_stored": len(doc_ids),
            "document_ids": doc_ids
        }

    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all supported files from a directory.

        Args:
            directory_path: Path to directory with documents

        Returns:
            Dict with aggregated results
        """
        dir_path = Path(directory_path)

        if not dir_path.exists():
            return {"success": False, "error": "Directory not found"}

        if not dir_path.is_dir():
            return {"success": False, "error": "Path is not a directory"}

        print(f"\n[i] Scanning directory: {directory_path}")
        print("=" * 70)

        # Find all supported files
        all_files = []
        for ext in DocumentParser.SUPPORTED_EXTENSIONS:
            all_files.extend(dir_path.glob(f"*{ext}"))

        if not all_files:
            return {
                "success": False,
                "error": "No supported files found",
                "supported_formats": list(DocumentParser.SUPPORTED_EXTENSIONS)
            }

        print(f"[i] Found {len(all_files)} file(s) to ingest\n")

        # Ingest each file
        results = []
        total_chunks = 0
        failed_files = []

        for file_path in all_files:
            result = self.ingest_file(str(file_path))
            results.append(result)

            if result["success"]:
                total_chunks += result["chunks_stored"]
            else:
                failed_files.append(str(file_path))

        # Summary
        print("\n" + "=" * 70)
        print("INGESTION SUMMARY")
        print("=" * 70)
        print(f"Files processed: {len(results)}")
        print(f"Successful: {len(results) - len(failed_files)}")
        print(f"Failed: {len(failed_files)}")
        print(f"Total chunks stored: {total_chunks}")

        if failed_files:
            print(f"\nFailed files:")
            for f in failed_files:
                print(f"  - {f}")

        return {
            "success": len(failed_files) == 0,
            "files_processed": len(results),
            "files_succeeded": len(results) - len(failed_files),
            "files_failed": len(failed_files),
            "total_chunks": total_chunks,
            "results": results,
            "failed_files": failed_files
        }


# =============================================================================
# Test Ingestion Pipeline
# =============================================================================
if __name__ == "__main__":
    from typing import Dict, Any

    print("=" * 70)
    print("Document Ingestion Pipeline Test")
    print("=" * 70)

    # Create test document
    test_dir = Path("../data/documents")
    test_dir.mkdir(exist_ok=True)

    test_file = test_dir / "test_rag_guide.txt"
    test_file.write_text("""
RAG System Architecture Guide

Retrieval-Augmented Generation (RAG) combines the power of retrieval systems with large language models to provide accurate, context-aware responses.

How RAG Works:

1. Document Ingestion
   - Parse documents (PDF, DOCX, TXT)
   - Split into chunks (500 tokens recommended)
   - Generate embeddings using Sentence Transformers
   - Store in vector database

2. Query Processing
   - User asks a question
   - Convert question to embedding
   - Search vector database for similar chunks
   - Retrieve top-k most relevant chunks

3. Response Generation
   - Pass retrieved chunks as context to LLM
   - LLM generates answer based on context
   - Return answer with source citations

Benefits of RAG:
- Accurate answers grounded in your documents
- No hallucinations (LLM uses provided context)
- Up-to-date information (just update documents)
- Cost-effective (smaller context than fine-tuning)

This system uses:
- Llama 3 for generation
- all-MiniLM-L6-v2 for embeddings
- Chroma for vector storage
- LangChain for orchestration
    """)

    # Test ingestion
    ingestion = IngestionService()

    print("\n[TEST] Ingesting single file:")
    result = ingestion.ingest_file(str(test_file))

    print("\n" + "=" * 70)
    print("INGESTION RESULT:")
    print("=" * 70)
    for key, value in result.items():
        if key != "document_ids":  # Skip long ID list
            print(f"  {key}: {value}")

    # Test search
    if result["success"]:
        print("\n[TEST] Searching ingested content:")
        print("-" * 70)

        test_queries = [
            "How does RAG work?",
            "What are the benefits of RAG?",
            "What embedding model is used?"
        ]

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            search_results = vector_store.search(query, k=2)

            for i, doc in enumerate(search_results):
                print(f"  Result {i+1}: {doc.page_content[:100]}...")

    # Cleanup test file
    # test_file.unlink()  # Uncomment to delete test file

    print("\n" + "=" * 70)
    print("[OK] Ingestion pipeline working!")
    print("=" * 70)
    print("\nNow you can:")
    print("  1. Add PDF/DOCX files to data/documents/")
    print("  2. Run ingestion.ingest_directory('data/documents')")
    print("  3. Search your documents!")
