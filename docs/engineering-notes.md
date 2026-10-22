# Engineering notes

The failures behind the README, in full.


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

## Design decisions

**Qdrant Cloud over Chroma.** The way I used Chroma, it kept its data on local disk, and Render's filesystem is ephemeral. Qdrant keeps the vectors remotely, supports metadata filtering, and has a free tier. I have not benchmarked the two against each other. pgvector would be the other option I would try.

**Local embeddings over a hosted embedding API.** The HuggingFace Inference API timed out with 504s. Running all-MiniLM-L6-v2 in-process removes the network dependency, costs nothing and keeps document text on the server. The cost was memory, which is what the deployment story above is about.

**Groq in the cloud, Ollama locally.** Both serve open-weight models and both are free to use at this scale. Groq is the only provider on the deployment, because Ollama is not installed there.

**500-character chunks with 50 overlap.** A common starting range that keeps enough context per chunk without diluting it. I did not tune it against the eval set.

**Hybrid by default.** Vector search misses exact keywords, and BM25 catches them. The measured gain is small (see above).

## Measured results

<a id="retrieval-accuracy"></a>
### Retrieval accuracy

A rerunnable evaluation over a committed synthetic corpus replaced this one on 2026-10-23; its tables and limits are in the [README](../README.md#measured-results) and its method in [evaluation.md](evaluation.md). The section below is the record of the earlier run, kept apart because it measured a different corpus.

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
