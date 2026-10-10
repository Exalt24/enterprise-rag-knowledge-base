# Enterprise RAG Knowledge Base

A retrieval-augmented generation service: upload documents, ask questions, get cited answers. FastAPI and LangChain on the backend, Qdrant for vectors, vector, BM25 hybrid and optional cross-encoder retrieval, server-sent-event streaming, an OpenAI-compatible endpoint, and a Next.js front end. It runs as a portfolio deployment on free tiers.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Next.js](https://img.shields.io/badge/next.js-16-black)

## Demo

The hosted copy is offline: the free-tier host suspended its services in October 2026, so the links that used to be here led to empty pages and were removed. Everything runs locally from the steps below. The notes on cold starts and the wake test that used to be here described that hosted copy. The example commands further down still show its old address (enterprise-rag-api.onrender.com): point them at your own backend (http://localhost:8000) instead.

---

## What broke

The best parts of this repo are the failures I had to diagnose, so they come first.

### The retrieval numbers I retracted

An earlier revision of this README advertised an accuracy series for vector, hybrid and reranked retrieval (about 40, 60 and 67.7 percent). Those figures predate a scoring bug I found on 2026-08-03, and they were never re-validated through the corrected code, so they are gone rather than restated.

The bug: `hybrid_search` applied `1 / (1 + score)` to "convert distance to similarity", but the collection uses cosine distance, where Qdrant already returns a similarity and higher is better. That inverted the ranking. A 0.82 match scored 0.549 while a 0.12 match scored 0.893, so the default retrieval path was promoting the least relevant chunks. It showed up as "I don't have that information" on questions that plain vector search answered correctly. The fix is to use the cosine score directly, clamped to 0 to 1. See [Retrieval accuracy](#retrieval-accuracy) for what I measured afterwards.

### What broke in deployment

The first deployment crash-looped on Render's 512 MB free instance, and from outside it looked exactly like a slow cold start. It took several rounds to find the real causes:

1. **The wrong torch.** The Dockerfile installed `requirements.txt`, which pins bare `torch`, and the default PyPI Linux wheel is the CUDA build at 526.6 MB. On a 512 MB instance the process was OOM-killed during import, before uvicorn bound a port, so the platform saw no open port and restarted it. Render builds from the Dockerfile and ignores the build command, which is why the lightweight requirements file that already existed was never used.
2. **The model loaded before the port opened.** Connecting the vector store loaded the embedding model at import time, ahead of the port bind. The model now loads on first use.
3. **The model was downloaded on every wake.** The container re-fetched all-MiniLM-L6-v2 from the HuggingFace Hub each time the instance woke, which is where the old "8 to 9 minute cold start" came from. The model is now baked into the image.
4. **torch arrived through a library I never imported.** `langchain_groq` pulls `langchain_core`, which does an optional `from transformers import GPT2TokenizerFast` for token counting, and `transformers` imports torch. Nothing in the request path needs it. I tried the CPU wheel (191.8 MB) and a one-thread pin, and neither was enough.
5. **The fix: no torch at all.** The deployed image uses fastembed, which runs the same all-MiniLM-L6-v2 weights through ONNX Runtime, so the 384-dimension vectors already in Qdrant stay valid and nothing was re-indexed. `transformers` is simply not installed, `langchain_core` skips the optional import, and `langchain_groq` still works. See `backend/requirements-render.txt`, which explains the whole chain in comments.

Measured in the built image under a real 512 MB cap, before and after: import time about two minutes down to 1.87 seconds, peak memory 438 MiB down to 259 MiB, and `/api/health` answering 200 in 0.14 seconds. Before the change the platform OOM-killed it.

Two smaller ones from the same week. The first ONNX adapter did not subclass LangChain's `Embeddings`, and `QdrantVectorStore` type-checks that argument, so on the deployment Qdrant connected, the collection was created, and then the store refused to build and every retrieval returned 503 under a startup log that read as healthy. The backend choice also first keyed off whether `langchain_huggingface` would import, which picked the wrong branch, because that package is installed in the image while torch is not. It now keys off torch.

### The model retirement

In early September 2026 Groq retired `llama-3.3-70b-versatile`, which this service named as a literal in seven places across four files. The API key stayed valid and the client still constructed, so every call failed, the fallback chain reported "All LLM providers unavailable", and the health endpoint still said the vector database was connected. The same retirement took the [multi-agent research demo](https://github.com/Exalt24/multi-agent-research) down the same day. The model is now a setting (`GROQ_MODEL`, default `openai/gpt-oss-120b`), so the next retirement is an environment change.

Two defects turned up while proving that fix. A failed generation was cached for an hour next to successful ones, so a short provider outage became an hour of serving "unavailable"; failed generations are no longer cached. And `/api/health` reported "unhealthy" on the deployment because it required an Ollama check, and Ollama is the local-development provider that does not exist there; on the deployment the vector store now decides the status.

---

## Features

### Retrieval

Three strategies are available through the API:

- **Vector search:** cosine similarity over 384-dimension embeddings.
- **Hybrid search (the default):** each of the top 2k vector hits scores its cosine similarity times 0.7, and every chunk that BM25 also returns gets a flat 0.15 added (a chunk found only by BM25 scores 0.15). The best k by that score are returned. So BM25 here is a keyword-agreement bonus, not a normalised BM25 score.
- **Cross-encoder reranking:** a neural rescoring pass using `cross-encoder/ms-marco-MiniLM-L-6-v2`. **It does not run on the free-tier deployment**: the module is not imported when `RENDER` is set, and the call returns the hybrid ordering unchanged. It works locally.

Two more are implemented but not wired to the API, and can only be called as library functions: **HyDE** (hypothetical document embeddings) and **multi-query** (LLM-generated query variations).

### Documents

- PDF, DOCX, TXT and Markdown, with optional OCR for scanned PDFs (local only).
- Recursive chunking at 500 characters with 50 overlap.
- Metadata per file: word count, upload date, file size, page numbers.
- File management: list and delete documents.

### Serving

- **Streaming:** `POST /api/query/stream` sends server-sent events, with citations before the first token.
- **OpenAI-compatible:** `POST /api/v1/chat/completions`, streaming and blocking, so an OpenAI client only needs its `base_url` repointed.
- **Generation:** Groq on the deployment. Locally, Ollama is tried first and Groq is the fallback.
- **Cache:** Redis, keyed on the question plus the retrieval options, with a one-hour TTL and an in-memory fallback.
- **Rate limiting:** a Redis-backed sliding window per IP and path: 60 requests per minute on `/api/query`, 10 on `/api/ingest`, 120 on every other path except health and docs. Without Redis the limiter is off.
- **Conversation memory:** pass a `conversation_id` to `/api/query` and follow-ups are rewritten into standalone questions before retrieval.

---

## Tech stack

**Backend:** FastAPI, LangChain, LangGraph (conversation memory), Pydantic, Python 3.13.

**LLMs:** Groq, default `openai/gpt-oss-120b`, set with `GROQ_MODEL`. Ollama (llama3) for local development.

**Embeddings:** all-MiniLM-L6-v2 (384 dimensions). On the deployment it runs through fastembed on ONNX Runtime, and torch is deliberately absent from the image (`backend/requirements-render.txt`). Locally it runs through sentence-transformers, which also provides the cross-encoder.

**Vector database:** Qdrant Cloud.

**Retrieval:** rank-bm25 for keyword search.

**Cache and rate limiter:** Redis.

**Documents:** pypdf, python-docx, pytesseract and pdf2image (OCR, local only).

**Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS.

**Deployment:** Render (backend, free 512 MB instance), Vercel (frontend), Docker.

---

## Measured results

<a id="retrieval-accuracy"></a>
### Retrieval accuracy

Measured on 2026-08-03 against the live deployment with `backend/tests/eval_retrieval.py`: 20 questions over a 26-chunk corpus, scored hit@k on whether the chunk containing the answer was retrieved at all.

| hit@k | vector only | hybrid |
|---|---|---|
| k=1 | 85.0% | 90.0% |
| k=3 | 100% | 100% |

- k=3 is saturated, so this benchmark is too easy to separate the strategies. k=1 is the number that discriminates.
- Hybrid beat plain vector by 5 points at k=1, which is one question out of twenty. That is a direction, not a settled result.
- A third column in the earlier measurement, "hybrid + rerank", is not reported here. Reranking does not run on the deployment (see above), so that column was the hybrid ordering again and said nothing about the reranker. I have not measured it locally.
- The corpus is small, and the live collection has changed since (it now holds 93 chunks from other documents), so the harness needs the original corpus re-ingested to reproduce these numbers. It reads the stored chunk text from `/documents`, or from Qdrant directly when `QDRANT_URL` and `QDRANT_API_KEY` are set:

```bash
python backend/tests/eval_retrieval.py --base-url https://enterprise-rag-api.onrender.com/api
```

### Cache

Two identical `POST /api/query` calls with a question that had not been asked before, timed with curl from my machine on 2026-10-07: 2.7 seconds for the first, 0.28 seconds for the second (a Redis hit). Both include network time.

```bash
curl -s -o /dev/null -w "%{time_total}s\n" -X POST https://enterprise-rag-api.onrender.com/api/query \
  -H "Content-Type: application/json" -d '{"question":"Why does the service stream citations before the first token?","k":3}'
```

### Memory

The deployed image peaks at 259 MiB with the model loaded and a real batch embedded, under a 512 MB cap (see [What broke in deployment](#what-broke-in-deployment)).

---

## Quick start

### Prerequisites

- Python 3.13+
- Node.js 18+
- Ollama ([download](https://ollama.ai/download)) for local generation, or a Groq API key
- A Qdrant Cloud cluster (the free tier works)
- Redis (optional, for caching and rate limiting)

### 1. Pull the local model

```bash
ollama pull llama3
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash. PowerShell: .\venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env:
# - QDRANT_URL and QDRANT_API_KEY (required)
# - GROQ_API_KEY (required on Render, optional locally)
# - REDIS_URL (optional)

python test_setup.py            # checks the setup
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

### 4. Use it

Upload documents in the web UI (PDF, DOCX, TXT, MD) and ask questions in the chat. The UI has a hybrid search toggle and a cross-encoder reranking toggle; the reranking toggle is disabled in the deployed UI. Or use the API:

- `POST /api/query`, with an optional `conversation_id` for multi-turn memory
- `POST /api/query/stream`, SSE and stateless
- `POST /api/v1/chat/completions`, OpenAI-compatible
- `POST /api/ingest`, `GET /api/stats`, `GET /api/documents`, `DELETE /api/documents/{name}`, `GET /api/health`

---

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
curl -N -X POST https://enterprise-rag-api.onrender.com/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"How does hybrid search work?","k":3,"use_hybrid_search":true}'
```

```
event: sources
data: {"sources": [...], "num_sources": 3}

event: token
data: {"text": "According"}
...
event: done
data: {"finish_reason": "stop"}
```

`POST /api/v1/chat/completions` speaks the OpenAI shape in both forms, streaming (`chat.completion.chunk` frames ending in `data: [DONE]`) and blocking (`chat.completion`):

```python
from openai import OpenAI
client = OpenAI(base_url="https://enterprise-rag-api.onrender.com/api/v1", api_key="not-used")
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

---

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

---

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
│   ├── keep_alive.py                  # pings Qdrant and Redis so the free tiers are not reclaimed
│   └── Dockerfile
├── frontend/                          # Next.js app
├── .github/workflows/keep-alive.yml   # runs keep_alive.py every 5 days
├── docker-compose.yml
├── SYSTEM-KNOWLEDGE.md                # design notes
└── README.md
```

---

## Testing

The tests are scripts you run by hand against a local backend, not a pytest suite wired into CI.

```bash
cd backend

# Ingestion pipeline
python -c "import sys; sys.path.insert(0, '.'); from tests.test_ingestion import test_complete_pipeline; test_complete_pipeline()"

# RAG query path
python -c "import sys; sys.path.insert(0, '.'); from tests.test_rag import test_rag_system; test_rag_system()"

# API endpoints, rate limiting
python tests/test_api.py
python tests/test_rate_limiting.py

# Retrieval accuracy, 20 questions, against a running API
python tests/eval_retrieval.py --base-url http://localhost:8001/api
```

---

## Design decisions

**Qdrant Cloud over Chroma.** The way I used Chroma, it kept its data on local disk, and Render's filesystem is ephemeral. Qdrant keeps the vectors remotely, supports metadata filtering, and has a free tier. I have not benchmarked the two against each other. pgvector would be the other option I would try.

**Local embeddings over a hosted embedding API.** The HuggingFace Inference API timed out with 504s. Running all-MiniLM-L6-v2 in-process removes the network dependency, costs nothing and keeps document text on the server. The cost was memory, which is what the deployment story above is about.

**Groq in the cloud, Ollama locally.** Both serve open-weight models and both are free to use at this scale. Groq is the only provider on the deployment, because Ollama is not installed there.

**500-character chunks with 50 overlap.** A common starting range that keeps enough context per chunk without diluting it. I did not tune it against the eval set.

**Hybrid by default.** Vector search misses exact keywords, and BM25 catches them. The measured gain is small (see above).

---

## Behaviour worth knowing

- File upload keeps only the base name of the uploaded filename, and uploads are capped at 10 MB (`MAX_FILE_SIZE_MB`).
- CORS allows all origins (`allow_origins=["*"]`). That is fine for a demo and should be narrowed before real use.
- If Qdrant is unreachable at start-up the API still boots in a degraded mode, and the retrieval endpoints return 503 until it recovers.
- Each answer carries a `model_used` field, and `/api/stats` reports the model that is actually configured.

---

## Deployment

### Docker

```bash
docker-compose up -d
docker-compose logs -f backend
docker-compose down
```

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

---

## Environment variables

Local development:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

Optional:

```bash
GROQ_API_KEY=gsk_...               # cloud generation, required on Render
GROQ_MODEL=openai/gpt-oss-120b     # change this when Groq retires a model
REDIS_URL=redis://localhost:6379   # cache and rate limiter
CACHE_TTL=3600                     # cache lifetime in seconds
MAX_FILE_SIZE_MB=10                # upload limit
REDIS_MAX_CONNECTIONS=10           # connection pool size
```

---

## Cost

The deployment runs on free tiers: Groq for generation, Qdrant Cloud for vectors, Redis for the cache, Render for the backend and Vercel for the frontend. Local development uses Ollama.

---

## Not done

- Reranking does not run on the deployment, and I have not measured it locally.
- HyDE and multi-query are implemented but not exposed through the API, and not evaluated.
- The retrieval eval is 20 questions over a small corpus; it is a regression check, not a benchmark.
- Conversation memory is in-process and is lost on restart.
- The tests are manual scripts, and there is no CI test run.
- CORS is open to every origin.

---

## License

MIT, see [LICENSE](LICENSE).

## Author

**Daniel Alexis Cruz**

- Portfolio: https://dacruz.vercel.app
- GitHub: https://github.com/Exalt24
- LinkedIn: https://linkedin.com/in/dacruz24
