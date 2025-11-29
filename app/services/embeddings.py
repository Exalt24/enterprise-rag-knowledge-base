"""
Embeddings Service

Converts text to vector embeddings for semantic search.

Uses Sentence Transformers (local, free):
- Model: all-MiniLM-L6-v2
- Dimension: 384
- Speed: ~500 embeddings/sec on CPU
- Quality: Good for most use cases

Alternative models (uncomment to try):
- all-mpnet-base-v2 (dimension: 768, slower but higher quality)
- paraphrase-MiniLM-L6-v2 (optimized for paraphrase detection)
"""

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings


class EmbeddingService:
    """
    Singleton service for generating embeddings.

    Why singleton? Loading the model takes time (~2-3 seconds).
    Load once, reuse everywhere!
    """

    _instance = None
    _embeddings = None

    def __new__(cls):
        """Singleton pattern - only create one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize embeddings model (only once)"""
        if self._embeddings is None:
            print(f"[i] Loading embedding model: {settings.embedding_model}")

            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={
                    'device': 'cpu',  # Use CPU (no GPU needed for this model)
                },
                encode_kwargs={
                    'normalize_embeddings': True,  # Normalize for cosine similarity
                    'batch_size': 32  # Process 32 texts at once (faster)
                }
            )

            print(f"[OK] Embedding model loaded!")

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get the embeddings model instance"""
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (384-dimensional vector)
        """
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts (batch processing - faster!).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)


# Global instance (import this in other files)
embedding_service = EmbeddingService()


# =============================================================================
# Alternative Models (Uncomment to Try)
# =============================================================================

# def create_embeddings_mpnet():
#     """
#     Higher quality embeddings (but slower).
#
#     Model: all-mpnet-base-v2
#     Dimension: 768 (vs 384 for MiniLM)
#     Speed: ~200 embeddings/sec (vs ~500 for MiniLM)
#     Quality: Better for complex queries
#
#     Use when: Accuracy > speed
#     """
#     return HuggingFaceEmbeddings(
#         model_name="all-mpnet-base-v2",
#         model_kwargs={'device': 'cpu'},
#         encode_kwargs={'normalize_embeddings': True}
#     )


# =============================================================================
# Test Embeddings
# =============================================================================
if __name__ == "__main__":
    from typing import List
    import numpy as np

    print("=" * 70)
    print("Embeddings Service Test")
    print("=" * 70)

    # Test single text
    print("\n[1] Single text embedding:")
    print("-" * 70)

    text = "What is RAG?"
    embedding = embedding_service.embed_text(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Embedding type: {type(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    print("\n[2] Batch embedding (faster for multiple texts):")
    print("-" * 70)

    texts = [
        "RAG stands for Retrieval-Augmented Generation",
        "Vector databases store embeddings",
        "LangChain is a framework for LLMs"
    ]

    embeddings = embedding_service.embed_texts(texts)

    print(f"Input: {len(texts)} texts")
    print(f"Output: {len(embeddings)} embeddings")
    print(f"Each embedding dimension: {len(embeddings[0])}")

    # Test similarity (cosine similarity)
    print("\n[3] Semantic similarity test:")
    print("-" * 70)

    query = "What is RAG?"
    doc1 = "RAG stands for Retrieval-Augmented Generation"
    doc2 = "The weather is sunny today"

    query_emb = embedding_service.embed_text(query)
    doc1_emb = embedding_service.embed_text(doc1)
    doc2_emb = embedding_service.embed_text(doc2)

    # Cosine similarity (dot product of normalized vectors)
    similarity_1 = np.dot(query_emb, doc1_emb)
    similarity_2 = np.dot(query_emb, doc2_emb)

    print(f"Query: '{query}'")
    print(f"\nDoc 1: '{doc1}'")
    print(f"Similarity: {similarity_1:.4f}")
    print(f"\nDoc 2: '{doc2}'")
    print(f"Similarity: {similarity_2:.4f}")

    print(f"\n[OK] Doc 1 is more similar! ({similarity_1:.4f} > {similarity_2:.4f})")

    print("\n" + "=" * 70)
    print("KEY LEARNINGS:")
    print("=" * 70)
    print("[+] Embeddings convert text to 384-dimensional vectors")
    print("[+] Similar texts have similar vectors (high dot product)")
    print("[+] This enables semantic search (find by meaning, not keywords)")
    print("[+] Batch processing is faster than one-by-one")
    print("[+] Singleton pattern = load model once, reuse everywhere")
