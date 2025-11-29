"""
API Request/Response Schemas

Pydantic models for API validation and documentation.
FastAPI uses these for:
- Request validation
- Response validation
- Auto-generated OpenAPI docs
- Type safety
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Query Endpoint Schemas
# =============================================================================

class QueryRequest(BaseModel):
    """Request schema for /query endpoint"""
    question: str = Field(
        ...,
        description="User question to answer",
        min_length=1,
        max_length=500,
        examples=["What is RAG?", "How does semantic search work?"]
    )
    k: Optional[int] = Field(
        default=3,
        description="Number of documents to retrieve",
        ge=1,
        le=10
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response"
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + BM25 keyword) for better accuracy"
    )
    optimize_query: bool = Field(
        default=False,
        description="Optimize query with LLM before retrieval (improves vague queries)"
    )
    use_reranking: bool = Field(
        default=False,
        description="Rerank results with cross-encoder (most accurate, slower)"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for multi-turn chat with memory"
    )


class Source(BaseModel):
    """Source document information"""
    file_name: str = Field(..., description="Source file name")
    page: Optional[int] = Field(None, description="Page number (for PDFs)")
    content_preview: str = Field(..., description="Preview of source content")
    relevance_score: Optional[float] = Field(None, description="Similarity score")


class QueryResponse(BaseModel):
    """Response schema for /query endpoint"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original question")
    sources: List[Source] = Field(default=[], description="Source documents")
    num_sources: int = Field(..., description="Number of sources used")
    model_used: str = Field(..., description="LLM model name")


# =============================================================================
# Ingest Endpoint Schemas
# =============================================================================

class IngestResponse(BaseModel):
    """Response schema for /ingest endpoint"""
    success: bool = Field(..., description="Whether ingestion succeeded")
    message: str = Field(..., description="Status message")
    file_name: Optional[str] = Field(None, description="Ingested file name")
    chunks_stored: Optional[int] = Field(None, description="Number of chunks stored")
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Stats Endpoint Schemas
# =============================================================================

class StatsResponse(BaseModel):
    """Response schema for /stats endpoint"""
    total_documents: int = Field(..., description="Total chunks in database")
    collection_name: str = Field(..., description="Chroma collection name")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimension: int = Field(..., description="Vector dimension")
    llm_model: str = Field(..., description="LLM model name")


# =============================================================================
# Health Endpoint Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Response schema for /health endpoint"""
    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    ollama_connected: bool = Field(..., description="Ollama connection status")
    vector_db_connected: bool = Field(..., description="Vector DB connection status")
    total_documents: int = Field(..., description="Documents in database")
