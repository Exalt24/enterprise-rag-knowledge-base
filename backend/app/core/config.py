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
    ollama_timeout: int = Field(
        default=30,
        description=(
            "Seconds the Ollama client waits on the server. The default of 30 suits a small "
            "model that is already loaded; raise it (OLLAMA_TIMEOUT) for a larger local "
            "model that has to be read into memory on the first call."
        )
    )
    ollama_reasoning: Optional[bool] = Field(
        default=None,
        description=(
            "Passed to Ollama's think option. Unset (the default) leaves the model's own "
            "behaviour. Set OLLAMA_REASONING=false for a reasoning model such as qwen3, "
            "which otherwise spends hundreds of tokens thinking before the first answer "
            "token, so a streamed answer sits silent for most of a minute."
        )
    )

    # Cloud API Keys
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key for fast inference (required for Render deployment)"
    )
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description=(
            "Groq chat model. A SETTING rather than a literal because Groq retires "
            "models and the previous default, llama-3.3-70b-versatile, was retired out "
            "from under this service. The key stayed valid and the client still "
            "constructed, so every call simply failed and the API answered 'All LLM "
            "providers unavailable' while looking healthy. Overridable with GROQ_MODEL "
            "so the next retirement is an env change, not a deploy."
        ),
    )
    # Which language model answers first
    llm_provider: str = Field(
        default="auto",
        description=(
            "Which provider is tried first: auto, ollama, groq or vertex. auto keeps the original "
            "order (Ollama, then Groq, then Vertex AI if it is configured). Any other value moves "
            "that provider to the front and keeps the rest, in the same order, as fallbacks."
        ),
    )

    # Vertex AI (Gemini) configuration. The tier exists only when a service account is configured.
    vertex_credentials_file: Optional[str] = Field(
        default=None,
        description="Path to a Google Cloud service account key JSON file (VERTEX_CREDENTIALS_FILE)",
    )
    vertex_credentials_json: Optional[str] = Field(
        default=None,
        description="The same key as inline JSON, for a container with no file (VERTEX_CREDENTIALS_JSON)",
    )
    vertex_project: Optional[str] = Field(
        default=None,
        description="Google Cloud project id. Falls back to project_id in the service account key.",
    )
    vertex_location: str = Field(
        default="us-central1",
        description="Vertex AI region, or global",
    )
    vertex_model: str = Field(
        default="gemini-3.5-flash",
        description=(
            "Gemini model id. A SETTING for the same reason groq_model is: models get retired, "
            "and the next retirement should be an env change."
        ),
    )
    vertex_api_base: Optional[str] = Field(
        default=None,
        description="Override the service endpoint (the tests point it at a stand-in server)",
    )
    vertex_safety_threshold: Optional[str] = Field(
        default=None,
        description=(
            "One block threshold applied to all four harm categories, for example BLOCK_ONLY_HIGH. "
            "Unset leaves the model's own defaults."
        ),
    )
    vertex_system_instruction: Optional[str] = Field(
        default=None,
        description="Optional systemInstruction text sent with every Vertex request",
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

    ingest_batch_size: int = Field(
        default=16,
        ge=1,
        description=(
            "Chunks embedded and upserted per batch during ingestion. Bounds peak memory so "
            "large files don't OOM small instances (production runs on a 512Mi box)."
        )
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
    print(f"Vector DB: Qdrant Cloud")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Chunk Size: {settings.chunk_size} tokens")
    print(f"Chunk Overlap: {settings.chunk_overlap} tokens")
    print(f"Retrieval Top-K: {settings.retrieval_top_k}")
    print(f"Groq API: {'Configured' if settings.groq_api_key else 'Not set'}")
    print(f"Vertex AI: {'Configured' if (settings.vertex_credentials_file or settings.vertex_credentials_json) else 'Not set'}")
    print(f"LLM provider order: {settings.llm_provider}")
    print(f"Redis Cache: {'Configured' if settings.redis_url else 'Not set (using in-memory)'}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
