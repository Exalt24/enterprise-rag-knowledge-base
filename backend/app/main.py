"""
FastAPI Application - Enterprise RAG Knowledge Base

REST API for the RAG system.

Endpoints:
- POST /api/query - Ask questions
- POST /api/ingest - Upload documents
- GET /api/stats - Database statistics
- GET /api/health - Health check
- GET /docs - Interactive API documentation (auto-generated!)

Run:
    uvicorn app.main:app --reload

Then visit:
    http://localhost:8000/docs (Interactive API docs)
    http://localhost:8000/redoc (Alternative docs)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings
from app.core.rate_limiter import RateLimitMiddleware


# =============================================================================
# Lifespan Event Handler (Modern FastAPI Pattern - No Deprecation!)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan handler (replaces deprecated on_event).

    Runs on startup and shutdown.
    """
    # Startup
    print("\n" + "=" * 70)
    print("Enterprise RAG API Starting...")
    print("=" * 70)
    print(f"LLM Model: {settings.ollama_model}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Vector DB: {settings.chroma_persist_dir}")
    print("=" * 70)
    print("\nAPI Ready!")
    print("  - Docs: http://localhost:8001/docs")
    print("  - ReDoc: http://localhost:8001/redoc")
    print("=" * 70 + "\n")

    yield  # Server runs here

    # Shutdown (if needed)
    print("\nShutting down...")


# Create FastAPI application
app = FastAPI(
    title="Enterprise RAG Knowledge Base API",
    description="Production-ready RAG system with Llama 3, Chroma, and LangChain",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc UI
    lifespan=lifespan  # Modern lifespan handler
)

# Rate limiting middleware (before CORS)
app.add_middleware(RateLimitMiddleware)

# CORS middleware (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["RAG"])


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Enterprise RAG Knowledge Base API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "query": "POST /api/query",
            "ingest": "POST /api/ingest",
            "stats": "GET /api/stats",
            "health": "GET /api/health"
        },
        "tech_stack": {
            "llm": settings.ollama_model,
            "embedding_model": settings.embedding_model,
            "vector_db": "Chroma",
            "framework": "FastAPI + LangChain"
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os

    # Use Render's PORT env var, default to 8001 for local
    port = int(os.getenv("PORT", 8001))

    # Disable reload on Render (production)
    is_production = os.getenv("RENDER") == "true"

    print("=" * 70)
    print(f"Starting FastAPI server on port {port}...")
    print(f"Environment: {'Production (Render)' if is_production else 'Development'}")
    print(f"Host: 0.0.0.0")
    print(f"Port: {port}")
    print("=" * 70)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False if is_production else True,  # No reload in production
        log_level="info"
    )
