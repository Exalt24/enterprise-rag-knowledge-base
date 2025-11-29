"""
HuggingFace Inference API Embeddings (Lightweight for Render)

Uses HuggingFace's free Inference API instead of local models.

Benefits:
- Zero memory footprint (no model loading)
- Free tier (generous limits)
- Same model as local (all-MiniLM-L6-v2)
- Perfect for 512MB RAM limit

For local dev: Use embeddings.py (local models)
For Render: Use this file (API)
"""

from typing import List
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.core.config import settings


class HFAPIEmbeddingService:
    """
    Lightweight embedding service using HuggingFace Inference API.

    Memory: ~0MB (no model loading!)
    vs Local: ~200MB
    """

    _instance = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._embeddings is None:
            import os
            print(f"[i] Using HuggingFace Inference API embeddings (cloud, 0MB)")

            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise ValueError("HUGGINGFACEHUB_API_TOKEN required for Render deployment")

            # Use HuggingFace's free Inference API
            # Same model as local: sentence-transformers/all-MiniLM-L6-v2
            self._embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                task="feature-extraction",
                huggingfacehub_api_token=hf_token
            )

            print(f"[OK] Cloud embeddings ready! (0MB memory)")

    def get_embeddings(self):
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)


# Global instance
embedding_service = HFAPIEmbeddingService()
