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
        examples=["What is RAG?", "How does semantic search work?"],
    )
    k: Optional[int] = Field(
        default=3, description="Number of documents to retrieve", ge=1, le=10
    )
    include_sources: bool = Field(
        default=True, description="Include source documents in response"
    )
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + BM25 keyword) for better accuracy",
    )
    optimize_query: bool = Field(
        default=False,
        description="Optimize query with LLM before retrieval (improves vague queries)",
    )
    use_reranking: bool = Field(
        default=False,
        description="Rerank results with cross-encoder (most accurate, slower)",
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for multi-turn chat with memory"
    )


class Source(BaseModel):
    """Source document information"""

    file_name: str = Field(..., description="Source file name")
    page: Optional[int] = Field(None, description="Page number (for PDFs)")
    location: Optional[str] = Field(None, description="Sheet and cell range (for workbooks), for example Budget!A2:C26")
    doc_id: Optional[str] = Field(None, description="Document id (SHA-256 of the file), set for files ingested since the source view was added")
    chunk_id: Optional[int] = Field(None, description="Position of the chunk in its document; with doc_id it opens GET /documents/{doc_id}/chunks/{chunk_id}")
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
# Archive Ingestion Schemas
# =============================================================================


class ArchiveSubmitResponse(BaseModel):
    """Response schema for POST /ingest/archive"""

    run_id: str = Field(..., description="Hatchet run id; GET /ingest/{run_id} shows the result table")
    status_url: str


class ArchiveFileRow(BaseModel):
    file: str = Field(..., description="Path of the file inside the archive")
    status: str = Field(..., description="ingested, skipped_duplicate or failed")
    reason: str = Field("", description="Why a file was skipped or failed")
    chunks: int = 0
    doc_id: Optional[str] = None


class ArchiveError(BaseModel):
    code: str
    reason: str


class ArchiveRunStatus(BaseModel):
    """Response schema for GET /ingest/{run_id}"""

    run_id: str
    state: str = Field(..., description="queued, validating, running, completed, rejected (the archive was refused) or failed (an unplanned error)")
    archive: Optional[str] = None
    counts: Dict[str, int]
    files: List[ArchiveFileRow]
    entries: Optional[int] = Field(None, description="File entries the archive holds")
    ignored: Optional[int] = Field(None, description="Mac and Windows leftovers that were not counted as files")
    to_process: Optional[int] = Field(None, description="Files that still needed indexing after the duplicate check")
    seconds: Optional[float] = Field(None, description="Seconds from the start of the check to the end of the run")
    error: Optional[ArchiveError] = Field(None, description="Set when the whole archive was refused")


# =============================================================================
# Source View Schemas
# =============================================================================


class ChunkLocation(BaseModel):
    """Where a chunk sits in its file, in words a reader would use"""

    kind: str = Field(..., description="page, cells or chunk")
    label: str = Field(..., description="Page 3, Budget!A2:C26 or Chunk 4 of 40")


class ChunkPassage(BaseModel):
    """One chunk of a document"""

    chunk_id: int = Field(..., description="Position of the chunk in its document, counted from 0")
    text: str = Field(..., description="The chunk text exactly as stored")
    page: Optional[int] = Field(None, description="PDF page, when the loader recorded one")
    location: ChunkLocation


class ChunkViewResponse(BaseModel):
    """Response schema for GET /documents/{doc_id}/chunks/{chunk_id}"""

    doc_id: str
    file_name: str
    file_type: Optional[str] = None
    total_chunks: Optional[int] = Field(None, description="Chunks the document produced")
    chunk: ChunkPassage
    before: List[ChunkPassage] = Field(default=[], description="Chunks before the cited one, in file order")
    after: List[ChunkPassage] = Field(default=[], description="Chunks after the cited one, in file order")


# =============================================================================
# Stats Endpoint Schemas
# =============================================================================


class StatsResponse(BaseModel):
    """Response schema for /stats endpoint"""

    total_documents: int = Field(..., description="Total chunks in database")
    collection_name: str = Field(..., description="Collection name")
    embedding_model: str = Field(..., description="Embedding model name")
    embedding_dimension: int = Field(..., description="Vector dimension")
    llm_model: str = Field(..., description="LLM model name")
    cache_stats: Optional[dict] = Field(
        None, description="Cache statistics (hits, misses, type)"
    )


# =============================================================================
# Health Endpoint Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Response schema for /health endpoint"""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    ollama_connected: bool = Field(..., description="Ollama connection status")
    vector_db_connected: bool = Field(..., description="Vector DB connection status")
    total_documents: int = Field(..., description="Documents in database")


# =============================================================================
# File Management Endpoint Schemas
# =============================================================================


class DocumentInfoSchema(BaseModel):
    """Schema for document information"""

    file_name: str = Field(..., description="Document filename")
    file_type: str = Field(..., description="File extension")
    file_size_kb: float = Field(..., description="File size in KB")
    upload_date: str = Field(..., description="Upload timestamp")
    chunk_count: int = Field(..., description="Number of chunks in vector DB")


class ListDocumentsResponse(BaseModel):
    """Response schema for /documents endpoint"""

    documents: List[DocumentInfoSchema] = Field(..., description="List of documents")
    total_documents: int = Field(..., description="Total document count")
    total_chunks: int = Field(..., description="Total chunks across all documents")


class DeleteDocumentResponse(BaseModel):
    """Response schema for /documents/{file_name} DELETE endpoint"""

    success: bool = Field(..., description="Whether deletion succeeded")
    message: str = Field(..., description="Status message")
    chunks_deleted: Optional[int] = Field(None, description="Number of chunks removed")
    error: Optional[str] = Field(None, description="Error message if failed")
