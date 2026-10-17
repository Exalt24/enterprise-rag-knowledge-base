# Design notes

Design notes for the Enterprise RAG Knowledge Base: how a document becomes searchable, how a question becomes an answer, and why the pieces are shaped the way they are. Everything here can be checked against `backend/app/`. The README covers setup, measured results and what broke.

## The shape of the system

Each service in `backend/app/services/` has one job: `document_parser`, `chunking`, `embeddings`, `vector_store`, `retrieval`, `advanced_retrieval`, `generation`, `cache`, `conversation`, `file_management`, `ingestion`, and `rag`, which orchestrates the rest. Heavy resources (the embedding model, the vector store connection, the cache) are singletons that load on first use rather than at import. That second part is not style: on a 512 MB instance, anything expensive at import time runs before uvicorn binds its port, and the platform kills the process.

## Ingestion

Upload, parse, chunk, embed, store.

1. `document_parser.py` reads PDF (pypdf), DOCX (python-docx), TXT and Markdown. OCR for scanned PDFs uses pytesseract and pdf2image and only runs locally, because it needs system binaries.
2. `chunking.py` uses LangChain's `RecursiveCharacterTextSplitter` with separators `\n\n`, `\n`, `. `, space, then nothing, at 500 characters with 50 overlap (`length_function=len`, so these are characters, not tokens). The overlap exists so a sentence spanning a chunk boundary still appears whole in one chunk.
3. `embeddings.py` produces 384-dimension all-MiniLM-L6-v2 vectors, L2-normalised so that cosine similarity is a dot product.
4. `vector_store.py` upserts into a Qdrant collection created with cosine distance. Ingestion embeds and upserts in batches (`INGEST_BATCH_SIZE`, default 16) to bound peak memory, so a large file cannot OOM a small instance.

Uploads are capped at 10 MB, and the filename is reduced to its base name before it touches disk.

### Two embedding backends

`embeddings.py` chooses a backend by whether torch is installed, not by an environment flag. The deployed image has no torch and uses fastembed, which runs the same weights through ONNX Runtime. A development machine has torch and uses sentence-transformers through `langchain_huggingface`, which the cross-encoder also needs. Both normalise their output, so the vectors are interchangeable and the existing collection needed no re-indexing when the deployment switched. The adapter has to subclass LangChain's `Embeddings`, because `QdrantVectorStore` type-checks that argument and rejects a duck type.

## Query

`rag.py` runs these steps for `POST /api/query`:

1. Check the cache. The key is an MD5 of the question, `k`, the hybrid flag and the rerank flag.
2. Optionally rewrite the question with the LLM (`optimize_query`).
3. Retrieve, either vector only or hybrid.
4. Optionally rerank with the cross-encoder.
5. Format the retrieved chunks into a context block.
6. Generate with a prompt that tells the model to answer only from the context and to say it does not have the information when the context does not contain it. Temperature is 0.1.
7. Cache the result, unless generation failed. A failed generation is never cached, because a provider outage is usually shorter than the one-hour TTL.

### Hybrid scoring

`advanced_retrieval.hybrid_search` takes the top 2k vector hits and the top 2k BM25 hits. Each vector hit scores its cosine similarity (clamped to 0 to 1) times 0.7. Every chunk BM25 also returns gets a flat 0.15, which is 0.5 times the 0.3 keyword weight, and a chunk only BM25 found scores 0.15. The best k by that total are returned. BM25 is therefore a keyword-agreement bonus and not a normalised BM25 score. The index is cached in memory and rebuilt when the document count changes.

The first version of this function ran `1 / (1 + score)` over the vector scores, as if they were distances. The collection uses cosine, where Qdrant already returns a similarity, so that inverted the ranking and hybrid search promoted the worst chunks. The README has the details.

### Reranking

The cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`, scored per query and chunk pair and min-max normalised. `advanced_retrieval.py` only imports it when the `RENDER` environment variable is unset. On the deployment, `rerank` returns the input documents in their existing order with a uniform score of 0.5, so a request with `use_reranking: true` is silently a hybrid request there. HyDE and multi-query retrieval exist as library methods and are not wired to any route.

## Generation

`generation.py` builds the providers by environment. With `RENDER=true` it uses Groq only. Otherwise it tries Ollama first (30 second timeout) and falls back to Groq. The Groq model comes from `GROQ_MODEL`, default `openai/gpt-oss-120b`, because the previous hardcoded default was retired by the provider and took the service down. `model_used` on every response says which provider actually answered.

For streaming, `generate_stream` buffers the first chunk before emitting anything, so a provider that fails on the opening call can fall through to the next one instead of leaving the client with half an answer.

## Streaming and the OpenAI endpoint

`POST /api/query/stream` sends `sources`, then `token` frames, then `done` (or `error` if generation dies mid-stream). It is stateless and skips the cache. `POST /api/v1/chat/completions` wraps the same machinery in the OpenAI response shape, streaming as `chat.completion.chunk` frames ending in `[DONE]`. Both set `X-Accel-Buffering: no`, which Render's proxy needs in order not to buffer the whole body.

## Conversation memory

`conversation.py` is a small LangGraph with three nodes: reformulate, retrieve, generate. The reformulation step turns a follow-up like "what weighting does it use" into a standalone question using the history, because retrieval cannot resolve a pronoun. The checkpointer is `MemorySaver`, so memory is per process and lost on restart.

## Cache and rate limiting

`cache.py` uses Redis when `REDIS_URL` is set and an in-memory dictionary otherwise, with a one-hour TTL and a pooled connection. `rate_limiter.py` is a Redis-backed sliding-window middleware keyed on client IP and path: 60 requests per minute on `/api/query`, 10 on `/api/ingest`, 120 on any other path, with health and docs exempt. Without Redis it does nothing.

## Degraded modes

- If Qdrant is unreachable at start-up, the API still boots, records why, and the retrieval routes return 503.
- If Redis is unreachable, the cache falls back to memory and the rate limiter turns itself off.
- If the cross-encoder cannot load, reranking is skipped.
- `/api/health` decides the deployment's health from the vector store alone, because the Ollama check it also runs does not apply where Ollama is not installed.

## Testing

`backend/tests/` holds scripts run by hand against a local backend: ingestion, the RAG query path, the API endpoints, rate limiting, and `eval_retrieval.py`, which scores 20 questions by hit@k against a running API. There is no CI test run. The eval is a regression check on a small corpus, not a benchmark.

## What I would change

- Measure reranking locally and report it, since it has never been measured.
- Score BM25 properly and normalise it, instead of the flat bonus.
- Replace the in-process conversation checkpointer with a Redis or Postgres one.
- Wire the tests into CI.
