"""
API Routes

REST endpoints for the RAG system:
- POST /query - Ask questions
- POST /ingest - Upload documents
- GET /stats - Database statistics
- GET /health - Health check
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import json
import shutil
import time
import uuid
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    StatsResponse,
    HealthResponse,
    Source
)
from app.services.rag import rag_service
from app.services.conversation import conversation_service
from app.services.generation import generation_service
from app.services.ingestion import IngestionService
from app.services.vector_store import vector_store
from app.core.config import settings

router = APIRouter()
ingestion_service = IngestionService()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Render and most reverse proxies buffer responses by default, which defeats streaming
    # entirely: the client sits silent and then receives the whole answer at once.
    "X-Accel-Buffering": "no",
}


def _sse(event: str, data: dict) -> str:
    """Format one server-sent event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# =============================================================================
# Streaming Query - Server-Sent Events
# =============================================================================

@router.post("/query/stream")
async def query_knowledge_base_stream(request: QueryRequest):
    """
    Same retrieval as /query, streamed over SSE.

    Frames:
      event: sources  -> citations, sent before the first token
      event: token    -> {"text": "..."} per chunk
      event: done     -> {"finish_reason": "stop"}
      event: error    -> {"message": "..."} if generation dies mid-stream

    Stateless: nothing about this request is retained between calls, and no request state is
    shared across concurrent callers.
    """
    def generate():
        try:
            sources, tokens = rag_service.query_stream(
                question=request.question,
                k=request.k,
                use_hybrid_search=getattr(request, "use_hybrid_search", True),
                optimize_query=getattr(request, "optimize_query", False),
                use_reranking=getattr(request, "use_reranking", False),
            )
            yield _sse("sources", {"sources": sources, "num_sources": len(sources)})
            for chunk in tokens:
                yield _sse("token", {"text": chunk})
            yield _sse("done", {"finish_reason": "stop"})
        except Exception as e:
            # The stream has already started, so a raised exception would just truncate the
            # body with no explanation. Send a real error frame instead.
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


# =============================================================================
# OpenAI-Compatible Chat Completions
# =============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Subset of OpenAI's chat completions request that this service honours."""

    messages: List[ChatMessage] = Field(..., min_length=1)
    model: Optional[str] = Field(default="enterprise-rag")
    stream: bool = Field(default=False)
    k: Optional[int] = Field(default=3, ge=1, le=10)
    use_reranking: bool = Field(default=False)


def _latest_user_message(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    raise HTTPException(status_code=400, detail="No user message found in `messages`.")


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions, backed by the RAG pipeline.

    Works with any OpenAI client by pointing base_url at this service. Set stream=true for
    incremental `chat.completion.chunk` frames terminated by [DONE], or leave it false for a
    single `chat.completion` object.

    Only the newest user message is used for retrieval. This endpoint is stateless by design:
    prior turns are not stored, and the caller owns conversation history. Nothing from the
    prompt or the response is persisted or logged.
    """
    question = _latest_user_message(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not request.stream:
        result = rag_service.query(
            question=question,
            k=request.k,
            use_reranking=request.use_reranking,
        )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": result.model_used,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }],
        }

    def generate():
        def frame(delta: dict, finish_reason=None):
            return "data: " + json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"

        try:
            _sources, tokens = rag_service.query_stream(
                question=question,
                k=request.k,
                use_reranking=request.use_reranking,
            )
            yield frame({"role": "assistant"})
            for chunk in tokens:
                yield frame({"content": chunk})
            yield frame({}, finish_reason="stop")
        except Exception as e:
            yield frame({"content": f"\n[error] {e}"}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


# =============================================================================
# Query Endpoint - Ask Questions
# =============================================================================

@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a question and get an AI-generated answer based on your documents.

    Flow:
    1. Retrieve relevant documents from vector database
    2. Generate answer using Llama 3 with context
    3. Return answer with source citations

    Example:
        POST /query
        {"question": "What is RAG?", "k": 3, "include_sources": true}
    """
    try:
        # Use conversation memory if conversation_id provided
        if request.conversation_id:
            # Conversation mode - uses LangGraph for memory
            conv_result = conversation_service.query_with_memory(
                request.question,
                request.conversation_id
            )
            
            # Build response from conversation result
            sources = []
            if request.include_sources:
                for source_dict in conv_result['sources']:
                    sources.append(Source(
                        file_name=source_dict['file_name'],
                        page=source_dict.get('page'),
                        content_preview=source_dict['content_preview'],
                        relevance_score=None
                    ))
            
            return QueryResponse(
                answer=conv_result['answer'],
                query=request.question,
                sources=sources,
                num_sources=len(sources),
                model_used=conv_result.get("model_used") or generation_service.primary_model_name
            )
        
        # Standard query (no memory)
        rag_response = rag_service.query(
            request.question,
            k=request.k,
            include_scores=request.include_sources,
            use_hybrid_search=request.use_hybrid_search,
            optimize_query=request.optimize_query,
            use_reranking=request.use_reranking
        )

        # Format sources
        sources = []
        if request.include_sources:
            for i, source_dict in enumerate(rag_response.sources):
                score = None
                if i < len(rag_response.retrieval_scores):
                    score = rag_response.retrieval_scores[i]

                sources.append(Source(
                    file_name=source_dict["file_name"],
                    page=source_dict.get("page"),
                    content_preview=source_dict["content_preview"],
                    relevance_score=score
                ))

        return QueryResponse(
            answer=rag_response.answer,
            query=rag_response.query,
            sources=sources,
            num_sources=rag_response.num_sources,
            model_used=rag_response.model_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# =============================================================================
# Ingest Endpoint - Upload Documents
# =============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats: PDF, DOCX, TXT, MD

    Flow:
    1. Save uploaded file to data/documents/
    2. Parse document
    3. Chunk text
    4. Generate embeddings
    5. Store in vector database

    Example:
        POST /ingest
        Content-Type: multipart/form-data
        file: <your_document.pdf>
    """
    try:
        # Validate file extension
        file_path = Path(file.filename)
        if file_path.suffix.lower() not in {'.pdf', '.docx', '.txt', '.md'}:
            return IngestResponse(
                success=False,
                message="Unsupported file format",
                error=f"Supported formats: PDF, DOCX, TXT, MD. Got: {file_path.suffix}"
            )

        # Security: Check file size (configurable via settings)
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Seek back to start

        if size > max_size_bytes:
            return IngestResponse(
                success=False,
                message="File too large",
                error=f"Max file size: {settings.max_file_size_mb}MB"
            )

        # Save uploaded file
        upload_dir = Path("data/documents")
        upload_dir.mkdir(exist_ok=True, parents=True)

        # Security: Sanitize filename to prevent path traversal
        safe_filename = Path(file.filename).name  # Only get basename, no directory
        save_path = upload_dir / safe_filename

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest document
        result = ingestion_service.ingest_file(str(save_path))

        if result["success"]:
            return IngestResponse(
                success=True,
                message="Document ingested successfully",
                file_name=result["file_name"],
                chunks_stored=result["chunks_stored"]
            )
        else:
            return IngestResponse(
                success=False,
                message="Ingestion failed",
                file_name=file.filename,
                error=result.get("error", "Unknown error")
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# =============================================================================
# Stats Endpoint - Database Statistics
# =============================================================================

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database and cache statistics.

    Returns:
    - Total documents stored
    - Embedding model info
    - LLM model info
    - Collection details
    - Cache statistics (hits, misses, hit rate)
    """
    try:
        stats = vector_store.get_stats()

        # Import cache service for stats
        from app.services.cache import cache_service
        cache_stats = cache_service.get_stats()

        return StatsResponse(
            total_documents=stats["total_documents"],
            collection_name=stats["collection_name"],
            embedding_model=stats["embedding_model"],
            embedding_dimension=stats["embedding_dimension"],
            llm_model=generation_service.primary_model_name,
            cache_stats=cache_stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# =============================================================================
# Health Endpoint - System Health Check
# =============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check system health.

    Validates:
    - Ollama connection
    - Vector database connection
    - Database has documents
    """
    try:
        import os

        # Test LLM (Ollama locally, Groq on Render)
        llm_ok = True
        try:
            if os.getenv("RENDER"):
                from langchain_groq import ChatGroq
                llm = ChatGroq(api_key=settings.groq_api_key, model="llama-3.3-70b-versatile")
            else:
                from langchain_ollama import OllamaLLM
                llm = OllamaLLM(model=settings.ollama_model)
            llm.invoke("test")
        except:
            llm_ok = False

        # Test vector DB
        vector_db_ok = True
        total_docs = 0
        try:
            stats = vector_store.get_stats()
            total_docs = stats["total_documents"]
        except:
            vector_db_ok = False

        # Determine overall status
        status = "healthy" if (llm_ok and vector_db_ok) else "unhealthy"

        return HealthResponse(
            status=status,
            ollama_connected=llm_ok,
            vector_db_connected=vector_db_ok,
            total_documents=total_docs
        )

    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            ollama_connected=False,
            vector_db_connected=False,
            total_documents=0
        )


# =============================================================================
# File Management Endpoints
# =============================================================================

from app.services.file_management import file_management
from app.api.schemas import (
    DocumentInfoSchema,
    ListDocumentsResponse,
    DeleteDocumentResponse
)


@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents():
    """
    List all uploaded documents with metadata.

    Returns:
    - File names, types, sizes
    - Upload dates
    - Chunk counts per document
    """
    try:
        docs = file_management.list_documents()

        total_chunks = docs[0].total_chunks if docs else 0

        doc_schemas = [
            DocumentInfoSchema(
                file_name=doc.file_name,
                file_type=doc.file_type,
                file_size_kb=doc.file_size_kb,
                upload_date=doc.upload_date,
                chunk_count=doc.chunk_count
            )
            for doc in docs
        ]

        return ListDocumentsResponse(
            documents=doc_schemas,
            total_documents=len(doc_schemas),
            total_chunks=total_chunks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{file_name}", response_model=DeleteDocumentResponse)
async def delete_document(file_name: str):
    """
    Delete a document from filesystem and vector database.

    Args:
        file_name: Name of file to delete

    Returns:
        Deletion status and chunks removed
    """
    try:
        result = file_management.delete_document(file_name)

        if result["success"]:
            return DeleteDocumentResponse(
                success=True,
                message=f"Document {file_name} deleted successfully",
                chunks_deleted=result.get("chunks_deleted", 0)
            )
        else:
            return DeleteDocumentResponse(
                success=False,
                message="Deletion failed",
                error=result.get("error", "Unknown error")
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
