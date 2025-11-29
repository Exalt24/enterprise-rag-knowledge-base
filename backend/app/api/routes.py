"""
API Routes

REST endpoints for the RAG system:
- POST /query - Ask questions
- POST /ingest - Upload documents
- GET /stats - Database statistics
- GET /health - Health check
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
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
from app.services.ingestion import IngestionService
from app.services.vector_store import vector_store
from app.core.config import settings

router = APIRouter()
ingestion_service = IngestionService()


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
                model_used="ollama/llama3"
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

        # Save uploaded file
        upload_dir = Path("data/documents")
        upload_dir.mkdir(exist_ok=True, parents=True)

        save_path = upload_dir / file.filename

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
    Get database statistics.

    Returns:
    - Total documents stored
    - Embedding model info
    - LLM model info
    - Collection details
    """
    try:
        stats = vector_store.get_stats()

        return StatsResponse(
            total_documents=stats["total_documents"],
            collection_name=stats["collection_name"],
            embedding_model=stats["embedding_model"],
            embedding_dimension=stats["embedding_dimension"],
            llm_model=settings.ollama_model
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
        # Test Ollama
        ollama_ok = True
        try:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=settings.ollama_model)
            llm.invoke("test")
        except:
            ollama_ok = False

        # Test vector DB
        vector_db_ok = True
        total_docs = 0
        try:
            stats = vector_store.get_stats()
            total_docs = stats["total_documents"]
        except:
            vector_db_ok = False

        # Determine overall status
        status = "healthy" if (ollama_ok and vector_db_ok) else "unhealthy"

        return HealthResponse(
            status=status,
            ollama_connected=ollama_ok,
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
