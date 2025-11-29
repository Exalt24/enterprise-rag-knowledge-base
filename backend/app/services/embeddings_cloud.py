"""
Cloud Embeddings Service (Lightweight for Free Tier Deployment)

Uses Google Gemini Embeddings API instead of local models.

Benefits:
- Zero memory footprint (no model loading)
- Fast, high-quality embeddings
- Free tier: 1500 requests/day
- Perfect for free hosting (512MB RAM limit)

For local development: Use embeddings.py (local models)
For deployment: Use this file (cloud API)
"""

from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


class CloudEmbeddingService:
    """
    Lightweight embedding service using Gemini API.

    Memory usage: ~0MB (no model loading!)
    vs. Local Sentence Transformers: ~200MB
    """

    _instance = None
    _embeddings = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._embeddings is None:
            print(f"[i] Using Gemini Embeddings API (cloud, lightweight)")

            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY required for cloud embeddings")

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=settings.gemini_api_key
            )

            print(f"[OK] Cloud embeddings ready! (0MB memory footprint)")

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        return self._embeddings

    def embed_text(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)


# Global instance
embedding_service = CloudEmbeddingService()
