"""
Vector Store Service

Manages Chroma vector database for semantic search.

Operations:
- Add documents (with embeddings)
- Search by similarity
- Delete documents
- Get stats

Using Chroma because:
- Open source, free
- Persistent storage (survives restarts)
- Fast similarity search
- Metadata filtering
"""

from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.core.config import settings
import os

# Use cloud embeddings on Render (512MB RAM limit), local otherwise
if os.getenv("RENDER"):
    from app.services.embeddings_cloud import embedding_service
else:
    from app.services.embeddings import embedding_service


class VectorStoreService:
    """
    Manages Chroma vector database.

    Singleton pattern - one database instance for the app.
    """

    _instance = None
    _vectorstore = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize or load existing vector database"""
        if self._vectorstore is None:
            print(f"[i] Loading vector database from: {settings.chroma_persist_dir}")

            self._vectorstore = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=embedding_service.get_embeddings(),
                collection_name="enterprise_rag"  # Name your collection
            )

            print(f"[OK] Vector database ready!")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector database.

        Args:
            documents: List of Document objects (already chunked!)

        Returns:
            List of document IDs

        Note: Embeddings are generated automatically by Chroma!
        """
        if not documents:
            return []

        print(f"[i] Adding {len(documents)} documents to vector store...")

        ids = self._vectorstore.add_documents(documents)

        print(f"[OK] Added {len(ids)} documents!")

        return ids

    def search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return (default from config)
            filter: Metadata filter (e.g., {"file_type": "pdf"})

        Returns:
            List of most similar Documents with metadata
        """
        k = k or settings.retrieval_top_k

        if filter:
            results = self._vectorstore.similarity_search(
                query,
                k=k,
                filter=filter
            )
        else:
            results = self._vectorstore.similarity_search(query, k=k)

        return results

    def search_with_scores(
        self,
        query: str,
        k: int = None
    ) -> List[tuple[Document, float]]:
        """
        Search with similarity scores.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (Document, score) tuples
            Score: Lower is more similar (distance metric)
        """
        k = k or settings.retrieval_top_k

        results = self._vectorstore.similarity_search_with_score(query, k=k)

        return results

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        self._vectorstore.delete(ids=ids)
        print(f"[OK] Deleted {len(ids)} documents")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        collection = self._vectorstore._collection
        count = collection.count()

        return {
            "total_documents": count,
            "collection_name": "enterprise_rag",
            "persist_directory": settings.chroma_persist_dir,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": 384  # MiniLM-L6-v2
        }

    def clear_database(self) -> None:
        """Delete ALL documents (use carefully!)"""
        collection = self._vectorstore._collection
        collection.delete(where={})
        print("[!] All documents deleted!")


# Global instance
vector_store = VectorStoreService()


# =============================================================================
# Test Vector Store
# =============================================================================
if __name__ == "__main__":
    from typing import Dict, Any

    print("=" * 70)
    print("Vector Store Service Test")
    print("=" * 70)

    # Sample documents
    docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "doc1.txt", "topic": "RAG"}
        ),
        Document(
            page_content="Vector databases store embeddings for semantic search.",
            metadata={"source": "doc2.txt", "topic": "VectorDB"}
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "doc3.txt", "topic": "LangChain"}
        ),
        Document(
            page_content="Chroma is an open-source vector database.",
            metadata={"source": "doc4.txt", "topic": "VectorDB"}
        ),
    ]

    # Add documents
    print("\n[1] Adding documents:")
    print("-" * 70)
    ids = vector_store.add_documents(docs)
    print(f"Document IDs: {ids}")

    # Search
    print("\n[2] Semantic search:")
    print("-" * 70)
    query = "What is RAG?"
    results = vector_store.search(query, k=2)

    print(f"Query: '{query}'")
    print(f"Results: {len(results)}")
    for i, doc in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Content: {doc.page_content}")
        print(f"    Metadata: {doc.metadata}")

    # Search with scores
    print("\n[3] Search with similarity scores:")
    print("-" * 70)
    results_with_scores = vector_store.search_with_scores(query, k=2)

    for i, (doc, score) in enumerate(results_with_scores):
        print(f"\n  Result {i+1}:")
        print(f"    Content: {doc.page_content[:60]}...")
        print(f"    Score: {score:.4f} (lower = more similar)")

    # Metadata filtering
    print("\n[4] Filtered search (only VectorDB topic):")
    print("-" * 70)
    filtered_results = vector_store.search(
        "What stores embeddings?",
        k=2,
        filter={"topic": "VectorDB"}
    )

    print(f"Results (filtered to topic='VectorDB'): {len(filtered_results)}")
    for doc in filtered_results:
        print(f"  - {doc.page_content}")
        print(f"    Topic: {doc.metadata['topic']}")

    # Stats
    print("\n[5] Database stats:")
    print("-" * 70)
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("[OK] Vector store working perfectly!")
    print("=" * 70)
    print("\nKey Learnings:")
    print("  - Semantic search finds by MEANING, not keywords")
    print("  - Lower score = more similar (distance metric)")
    print("  - Metadata filtering narrows search scope")
    print("  - Chroma handles embedding generation automatically")
