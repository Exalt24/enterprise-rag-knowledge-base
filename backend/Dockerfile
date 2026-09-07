# Enterprise RAG Backend - Production Docker Image
#
# Multi-stage build for optimized image size
# Python 3.13 + FastAPI + LangChain + Qdrant Cloud

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.13-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
#
# IMPORTANT: this installs requirements-render.txt, NOT requirements.txt.
# requirements.txt is the full local freeze and pins bare `torch`, whose default
# PyPI linux wheel bundles the CUDA runtime: 526.6 MB, on a box with 512 MB of RAM.
# requirements-render.txt pins torch==2.13.0+cpu (191.8 MB) and drops the packages
# that never run in production (onnxruntime, scikit-learn/scipy, pytesseract,
# pdf2image, kubernetes, google-ai-generativelanguage). Installing the wrong file
# OOM-kills the container during import, before uvicorn ever binds the port.
COPY requirements-render.txt .

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir --user -r requirements-render.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app /app/app
COPY tests /app/tests

# Create data directory for uploaded documents
RUN mkdir -p /app/data/documents

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Bake the embedding model into the image.
# Without this, the container re-downloads all-MiniLM-L6-v2 from the HF Hub
# unauthenticated on every cold wake, which is what made cold starts 8-9 minutes.
ENV HF_HOME=/opt/hf
ENV SENTENCE_TRANSFORMERS_HOME=/opt/hf

# CPU torch allocates a scratch arena PER THREAD and defaults to one thread per core,
# so on a many-core host it reserves far more than a 512MB instance can spare while
# doing nothing useful for a 22MB model. Pinning to one thread is the single largest
# remaining saving here. TOKENIZERS_PARALLELISM silences the fork warning that the
# tokenizer prints once these are set.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && chmod -R a+rX /opt/hf

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/health || exit 1

# Run application
CMD ["python", "-m", "app.main"]
