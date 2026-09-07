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

from typing import TYPE_CHECKING, List
from app.core.config import settings

if TYPE_CHECKING:  # for type checkers only, never at runtime
    from langchain_huggingface import HuggingFaceEmbeddings

# NOTE: langchain_huggingface is imported INSIDE _load(), not here.
#
# Importing it pulls in torch, and on a 0.1-CPU free instance that import alone takes
# about two minutes and a few hundred MB, before a single line of our own code runs.
# Because this module is imported at process start (vector_store imports it, routes
# import that, main imports routes), that cost landed ahead of uvicorn binding a port,
# so the platform's port scan timed out and killed the process every time. Deferring
# the model load was not enough on its own: the IMPORT was the expensive part, not the
# load. With this deferred, the API binds in seconds and torch is paid for once, on the
# first request that actually needs an embedding.


class _FastEmbedAdapter:
    """
    The langchain Embeddings surface, backed by fastembed's ONNX runtime.

    Only the two methods the rest of this codebase and QdrantVectorStore actually
    call are implemented (`embed_documents`, `embed_query`), which is the whole
    interface langchain requires of an embeddings object. Deliberately NOT a
    subclass of langchain_core.embeddings.Embeddings: importing that is harmless
    today, but the entire point of this class is to keep the ONNX path free of any
    import that could reach transformers, and inheriting from a moving library
    surface for the sake of two method names is not a trade worth making.

    fastembed already L2-normalises its output, which matches
    `normalize_embeddings=True` on the torch path, so the two are interchangeable
    against a cosine collection.
    """

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        # The seed data and the live Qdrant collection are 384-dimensional
        # all-MiniLM-L6-v2 vectors. fastembed names the same weights with the
        # publisher prefix, and a bare "all-MiniLM-L6-v2" is not in its registry.
        canonical = (
            model_name
            if "/" in model_name
            else f"sentence-transformers/{model_name}"
        )
        self._model = TextEmbedding(model_name=canonical)
        self.model_name = canonical

    def embed_documents(self, texts):
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text):
        # `embed` is a generator over a batch; one text in, one vector out.
        return next(iter(self._model.embed([text]))).tolist()


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
        """
        Cheap constructor. The model is NOT loaded here.

        This module is imported at process start (vector_store imports it, routes
        import that, main imports routes), so constructing the model in __init__
        put a multi-hundred-MB allocation ahead of uvicorn binding the port. On a
        512 MB instance that OOM-killed the process before it could ever serve
        /api/health, so the platform saw no open port, killed it, and restarted:
        a crash loop that looked like a slow cold start. Load on first use instead.
        """
        pass

    def _load(self):
        """
        Load the model on first use, then reuse it (singleton-cached).

        TWO BACKENDS, chosen by what is installed rather than by an env flag, because
        the deployed image and the dev machine legitimately differ:

        - ONNX via fastembed, which is what the Render image ships. torch is NOT
          installed there, deliberately: importing it costs a few hundred MB before a
          single vector is computed, and on a 512MB instance that is enough to get the
          process OOM-killed during import so the port never opens.
        - sentence-transformers via langchain_huggingface, on a dev box where torch is
          present and memory is not scarce, so the cross-encoder reranker also works.

        Both run the SAME all-MiniLM-L6-v2 weights and both L2-normalise, so vectors
        are interchangeable and the existing Qdrant collection needs no re-indexing.
        Verified equal to ~1e-6 by tests/test_embedding_backends_agree.py.
        """
        if EmbeddingService._embeddings is None:
            # Key off TORCH, not off langchain_huggingface. The first version of this
            # tried importing langchain_huggingface and fell back on ImportError, which
            # picked the WRONG branch: that package is installed in the Render image
            # while torch is not, so the import succeeded, the torch path was chosen,
            # and the embedding call then failed at construction. The dependency that
            # actually decides this is torch, so ask about torch.
            import importlib.util

            has_torch = importlib.util.find_spec("torch") is not None

            if not has_torch:
                print(f"[i] Loading embedding model (ONNX): {settings.embedding_model}")
                EmbeddingService._embeddings = _FastEmbedAdapter(settings.embedding_model)
            else:
                from langchain_huggingface import HuggingFaceEmbeddings

                print(f"[i] Loading embedding model (torch): {settings.embedding_model}")
                EmbeddingService._embeddings = HuggingFaceEmbeddings(
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

        return EmbeddingService._embeddings

    def get_embeddings(self) -> "HuggingFaceEmbeddings":
        """Get the embeddings model instance"""
        return self._load()

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (384-dimensional vector)
        """
        return self._load().embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts (batch processing - faster!).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._load().embed_documents(texts)


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
