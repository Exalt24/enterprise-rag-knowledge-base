"""
Application Configuration

Centralized settings using Pydantic for validation.
All environment variables loaded here.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings from environment variables.

    Pydantic validates all settings on startup - catches config errors early!
    """

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama3",
        description="Ollama model name"
    )

    # Cloud API Keys
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key for fast inference (required for Render deployment)"
    )
    # Qdrant Cloud Vector Database
    qdrant_url: str = Field(
        default="",
        description="Qdrant Cloud URL"
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant Cloud API key"
    )
    qdrant_collection: str = Field(
        default="enterprise_rag",
        description="Qdrant collection name"
    )

    # Redis Cache
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching"
    )

    # Document Processing
    chunk_size: int = Field(
        default=500,
        description="Token size for text chunks"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Token overlap between chunks"
    )
    max_documents: int = Field(
        default=1000,
        description="Maximum documents to store"
    )

    # Embeddings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=3,
        description="Number of documents to retrieve per query"
    )

    # Performance
    request_timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )

    # Cache Settings
    cache_ttl: int = Field(
        default=3600,
        description="Cache time-to-live in seconds (1 hour)"
    )

    # File Upload Limits
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file upload size in MB"
    )

    # Redis Connection Pool
    redis_max_connections: int = Field(
        default=10,
        description="Maximum Redis connections in pool"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Helper to display current config
def print_config():
    """Print current configuration (for debugging)"""
    print("=" * 70)
    print("Current Configuration")
    print("=" * 70)
    print(f"Ollama URL: {settings.ollama_base_url}")
    print(f"Ollama Model: {settings.ollama_model}")
    print(f"Chroma DB: {settings.chroma_persist_dir}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Chunk Size: {settings.chunk_size} tokens")
    print(f"Chunk Overlap: {settings.chunk_overlap} tokens")
    print(f"Retrieval Top-K: {settings.retrieval_top_k}")
    print(f"Groq API: {'Configured' if settings.groq_api_key else 'Not set'}")
    print(f"Redis Cache: {'Configured' if settings.redis_url else 'Not set (using in-memory)'}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
