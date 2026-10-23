# Reference

Moved out of the README: run steps, API examples, deployment and environment variables. The old hosted address (enterprise-rag-api.onrender.com) is offline; use your own backend at http://localhost:8001.

## Quick start

### Prerequisites

- Python 3.13+
- Node.js 18+
- Docker, to run Qdrant locally (or a Qdrant Cloud cluster, the free tier works)
- Ollama ([download](https://ollama.ai/download)) for local generation, or a Groq API key
- Redis (optional, for caching and rate limiting)

This is the path the recording above was made with. The first backend start downloads the embedding model, and `pip install` pulls torch, so allow a few minutes.

### 1. Start Qdrant and pull a model

```bash
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant
ollama pull llama3
```

The code defaults to `llama3`. Any Ollama model works through the settings in step 2: the recording used `qwen3:14b`.

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash. PowerShell: .\venv\Scripts\activate
pip install -r requirements.txt
```

Settings are environment variables or a `backend/.env` file (`.env.example` lists the cloud ones). For a fully local run:

```bash
QDRANT_URL=http://localhost:6333     # no QDRANT_API_KEY needed for a local container
OLLAMA_MODEL=llama3                  # the model you pulled
# Only for a reasoning model such as qwen3, which otherwise "thinks" for a long
# time before the first answer token. A larger model that has to be loaded on its
# first call may also need a longer timeout than the default 30 seconds:
OLLAMA_REASONING=false
OLLAMA_TIMEOUT=300
```

```bash
python test_setup.py            # checks Ollama, the embedding model and Qdrant
python -m app.main              # http://localhost:8001, docs at /docs
```

### 3. Frontend

```bash
cd frontend
npm install
# Create .env.local containing:
# NEXT_PUBLIC_API_URL=http://localhost:8001/api
npm run dev                     # http://localhost:3000
```

For a faster page, `npm run build && npm run start` serves the production build (the recording used that, with `NEXT_PUBLIC_API_URL` set when building).

### 4. Load the sample documents

Drop the three files from `docs/sample-corpus/` onto the upload panel, or from the repo root:

```bash
for f in docs/sample-corpus/*.md; do curl -X POST http://localhost:8001/api/ingest -F "file=@$f"; done
```

That is 14 chunks (5, 5 and 4). Then ask, for example, "How many weeks of parental leave do I get, and can I split them?" or "What should I do if my laptop is stolen?".

### 5. Use it

Upload documents in the web UI (PDF, DOCX, TXT, MD) and ask questions in the chat. The UI has a hybrid search toggle and a cross-encoder reranking toggle; the reranking toggle is disabled in the deployed UI. Or use the API:

- `POST /api/query`, with an optional `conversation_id` for multi-turn memory
- `POST /api/query/stream`, SSE and stateless
- `POST /api/v1/chat/completions`, OpenAI-compatible
- `POST /api/ingest`, `GET /api/stats`, `GET /api/documents`, `DELETE /api/documents/{name}`, `GET /api/health`

## Advanced usage

### Retrieval strategies as library calls

```python
from app.services.rag import rag_service

# Vector only
rag_service.query(question="What are the key features?", k=3, use_hybrid_search=False)

# Hybrid, the default
rag_service.query(question="What are the key features?", k=3, use_hybrid_search=True)

# Hybrid plus the cross-encoder (local only)
rag_service.query(question="What are the key features?", k=3,
                  use_hybrid_search=True, use_reranking=True)
```

HyDE and multi-query are not exposed through the API:

```python
from app.services.advanced_retrieval import advanced_retrieval

docs, scores = advanced_retrieval.hyde_search(query="What are the key features?", k=3, with_scores=True)
docs, scores = advanced_retrieval.multi_query_search(query="What are the key features?", k=3, with_scores=True)
```

### Streaming and the OpenAI-compatible endpoint

`POST /api/query/stream` emits a `sources` frame before the first token, so citations render while the answer is still arriving, then `token` frames, then `done`. It is the stateless route: only the newest question drives retrieval, nothing is kept between calls, and it bypasses the response cache on purpose, because replaying a cached string as synthetic tokens would report a model that never ran for that request.

```bash
curl -N -X POST http://localhost:8001/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"How much is the learning budget?","k":3,"use_hybrid_search":true}'
```

This is the output from the sample documents, run locally on 2026-10-11 (the `sources` frame is shortened here; the real one carries a 100-character preview for each of the three chunks, and the token frames are cut down to a few):

```
event: sources
data: {"sources": [{"file_name": "larkspur-employee-handbook.md", "page": null, "content_preview": "## Learning budget\n\nEvery employee has a learning budget of 1,500 euros per year for courses, books ..."}, ...], "num_sources": 3}

event: token
data: {"text": "The"}

event: token
data: {"text": " learning"}
...
event: token
data: {"text": "."}

event: done
data: {"finish_reason": "stop"}
```

The web UI lists the file name of each source under the answer. It does not show the preview text, so there is nothing to click open; the preview is only in the API response.

`POST /api/v1/chat/completions` speaks the OpenAI shape in both forms, streaming (`chat.completion.chunk` frames ending in `data: [DONE]`) and blocking (`chat.completion`):

```python
from openai import OpenAI
# The service does not check keys, but the client wants one: set OPENAI_API_KEY to any value.
client = OpenAI(base_url="http://localhost:8001/api/v1")
stream = client.chat.completions.create(
    model="enterprise-rag",
    messages=[{"role": "user", "content": "How does hybrid search work?"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Two implementation notes. The first chunk is buffered before anything is emitted, because providers almost always fail on the opening call and buffering lets the fallback take over instead of stranding the client mid-answer. And `X-Accel-Buffering: no` is mandatory on Render: without it the platform proxy buffers the whole response and streaming does nothing, which looks fine on localhost.

### Conversation memory

`POST /api/query` with a `conversation_id` uses a LangGraph path that rewrites a follow-up into a standalone question from the chat history before retrieval runs, since you cannot vector-match a pronoun:

```
"What is hybrid search?"        -> answers with the weighting
"What weighting does it use?"   -> rewritten to "What weighting does hybrid search use?"
```

The checkpointer is `MemorySaver`, so history lives in the process and does not survive a restart. Memory is also off on the streaming route by design.

### OCR for scanned PDFs (local only)

```python
from app.services.document_parser import DocumentParser

docs = DocumentParser.parse("scanned_document.pdf", use_ocr=True)
```

OCR needs system binaries: `tesseract-ocr` and `poppler-utils` on Debian or Ubuntu, `brew install tesseract poppler` on macOS, or the [UB-Mannheim Windows build](https://github.com/UB-Mannheim/tesseract/wiki).

## API examples

```bash
# Query with the advanced options
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What technologies are mentioned?",
    "k": 3,
    "include_sources": true,
    "use_hybrid_search": true,
    "use_reranking": false,
    "optimize_query": true
  }'

# Upload a document
curl -X POST http://localhost:8001/api/ingest -F "file=@document.pdf"

# Statistics and health
curl http://localhost:8001/api/stats
curl http://localhost:8001/api/health
```

## Project structure

```
enterprise-rag-knowledge-base/
├── backend/
│   ├── app/
│   │   ├── api/             # routes.py, schemas.py
│   │   ├── core/            # config.py, rate_limiter.py
│   │   ├── services/
│   │   │   ├── document_parser.py     # PDF, DOCX, TXT, Markdown, OCR
│   │   │   ├── chunking.py            # recursive splitter, 500 / 50
│   │   │   ├── embeddings.py          # fastembed (ONNX) or sentence-transformers
│   │   │   ├── vector_store.py        # Qdrant client
│   │   │   ├── retrieval.py           # vector retrieval
│   │   │   ├── advanced_retrieval.py  # hybrid, HyDE, multi-query, reranking
│   │   │   ├── generation.py          # Groq / Ollama
│   │   │   ├── rag.py                 # orchestration
│   │   │   ├── cache.py               # Redis and in-memory cache
│   │   │   ├── conversation.py        # multi-turn memory
│   │   │   ├── file_management.py
│   │   │   └── ingestion.py
│   │   └── main.py
│   ├── tests/                         # see Testing
│   ├── requirements.txt               # full local set
│   ├── requirements-render.txt        # deployed image, no torch
│   ├── keep_alive.py                  # pings Qdrant and Redis so the free tiers are not reclaimed; run it by hand, no workflow does
│   └── Dockerfile
├── frontend/                          # Next.js app
├── docker-compose.yml
├── docs/
│   ├── demo.gif                       # the README hero
│   ├── engineering-notes.md           # the failures behind the README
│   ├── reference.md                   # this file
│   ├── sample-corpus/                 # documents for the quick start
│   └── SYSTEM-KNOWLEDGE.md            # design notes
└── README.md
```

## Behaviour worth knowing

- File upload keeps only the base name of the uploaded filename, and uploads are capped at 10 MB (`MAX_FILE_SIZE_MB`).
- CORS allows all origins (`allow_origins=["*"]`). That is fine for a demo and should be narrowed before real use.
- If Qdrant is unreachable at start-up the API still boots in a degraded mode, and the retrieval endpoints return 503 until it recovers.
- Each answer carries a `model_used` field, and `/api/stats` reports the model that is actually configured.

## Known issues

- Conversation memory is in-process and lost on restart.
- CORS allows every origin. `GET /api/health` makes a real model call and can stall other requests on a slow local model.
- `docker-compose.yml` predates the move to Qdrant and is unmaintained.
- `DELETE /api/documents/{name}` deletes the chunks but reports `chunks_deleted: 0`.

## The recording

The GIF in the README was recorded on a local stack: Qdrant in Docker, this repo's backend and front end, and qwen3:14b served by Ollama. It uses three short made-up handbook files, about 14 chunks. The two answer stretches are shown at half speed, marked on screen. Sources are listed per chunk, so one file can appear twice in a citation list, and the dates in the document list read month/day/year.

The commands in the README use the code's default model, `llama3`. To match the recording, pull `qwen3:14b` and set `OLLAMA_MODEL=qwen3:14b`, `OLLAMA_REASONING=false` and `OLLAMA_TIMEOUT=300` before starting the backend (step 2 above). Answers from `llama3` will read differently.

## Deployment

### Docker

`docker-compose.yml` predates the move from Chroma to Qdrant: it still sets a Chroma directory, defines no Qdrant service, and its port and front-end API address do not match the defaults above. I have not run it since, so treat it as unmaintained and use the step-by-step run in Quick start, which I did run. The single `docker run` for Qdrant in step 1 is the only Docker piece that path needs.

### Backend on Render

Render builds from the backend Dockerfile, which installs `requirements-render.txt`. Set these environment variables:

```
RENDER=true
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
REDIS_URL=your_redis_url
```

### Frontend on Vercel

Set the root directory to `frontend` and add `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com/api`.

## Environment variables

Local development:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
QDRANT_URL=http://localhost:6333   # or https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key # Qdrant Cloud only
```

Optional:

```bash
OLLAMA_TIMEOUT=30                  # seconds the Ollama client waits; raise it for a big local model
OLLAMA_REASONING=false             # unset by default; false stops a reasoning model such as qwen3 from thinking first
QDRANT_COLLECTION=enterprise_rag   # collection name
GROQ_API_KEY=gsk_...               # cloud generation, required on Render
GROQ_MODEL=openai/gpt-oss-120b     # change this when Groq retires a model
REDIS_URL=redis://localhost:6379   # cache and rate limiter
CACHE_TTL=3600                     # cache lifetime in seconds
MAX_FILE_SIZE_MB=10                # upload limit
REDIS_MAX_CONNECTIONS=10           # connection pool size
```

