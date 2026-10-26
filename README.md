# Enterprise RAG Knowledge Base

[![CI](https://github.com/Exalt24/enterprise-rag-knowledge-base/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Exalt24/enterprise-rag-knowledge-base/actions/workflows/ci.yml)

FastAPI RAG service on Qdrant: vector and BM25 hybrid retrieval, SSE streaming and an OpenAI-compatible endpoint.

![Uploading three documents to a local knowledge base, then asking two questions and watching cited answers stream in](docs/demo.gif)

*Recorded on a local stack: Qdrant in Docker, this repo's backend and front end, and qwen3:14b served by Ollama, over three short made-up handbook files. The answer stretches are shown at half speed, marked on screen. More in [docs/reference.md](docs/reference.md#the-recording).*

**Proof:** a retrieval evaluation anyone can rerun in one command (invented documents, labelled questions, a pinned baseline, results below), a [read-only MCP server](#mcp-server-read-only-local) for Claude Code and other MCP clients, and CI that runs the tests plus a mutation check. An older 85% and 90% hit rate from 2026-08-03 (a corpus that is no longer in this repo, so not comparable with the tables below) is kept apart in [docs/engineering-notes.md](docs/engineering-notes.md#retrieval-accuracy).

The hosted copy is offline: the free-tier host suspended its services in October 2026. It runs locally in a few commands below.

## What broke

Upload documents (PDF, DOCX, TXT, Markdown), ask questions, get answers with citations: FastAPI and LangChain over Qdrant, a Next.js front end. I built it to see where a retrieval pipeline fails, so most of this README is about the failures.


- **A scoring bug that inverted the ranking.** `hybrid_search` applied `1 / (1 + score)` to a cosine similarity that was already "higher is better", so a 0.82 match scored 0.549 and a 0.12 match scored 0.893. Every test stayed green because they only checked that results came back. The fix uses the cosine score directly, `backend/tests/test_hybrid_ranking.py` now fails on the original line, and the mutation check puts the bug back and shows a test going red. I retracted an earlier accuracy series (about 40, 60 and 67.7 percent) because it predated the fix.
- **A silent fallback.** `hybrid_search` catches every exception and answers with plain vector search, so a broken BM25 branch looked like working hybrid retrieval. The engine now counts its fallbacks and the evaluation fails when the count is above zero.
- **A deploy that looked like a slow cold start.** The 512 MB free instance was OOM-killed on import by the 526.6 MB default torch wheel. The deployed image now uses fastembed (the same all-MiniLM-L6-v2 weights through ONNX Runtime, so stored vectors stayed valid): under a real 512 MB cap, import went from about two minutes to 1.87 seconds and peak memory from 438 MiB to 259 MiB (measured once on 2026-09-07; the output was not kept).
- **A retired model.** Groq retired the model this service named in seven places and calls failed while the health endpoint still said the vector database was connected. The model is now one setting, `GROQ_MODEL`.

Full write-up: [docs/engineering-notes.md](docs/engineering-notes.md).

## Measured results

### Rerunnable evaluation (run on 2026-10-23)

The corpus is **synthetic**: four documents invented for this repository, in [`backend/eval/corpus/`](backend/eval/corpus/), describing no real organisation and claiming no domain experience. They give 27 chunks through the repo's own chunker. The 42 questions in [`backend/eval/questions.jsonl`](backend/eval/questions.jsonl) are 12 direct, 12 paraphrase, 8 multi-document and 10 unanswerable. Each answerable question carries a `must_contain` phrase copied verbatim from the corpus, and a retrieved chunk counts only if it holds it.

Everything goes through the repo's own chunker, vector store and retrieval code; only the embedder is swapped. Wiring, labels, baseline and mutants: [docs/evaluation.md](docs/evaluation.md). Cells show the value and a 95% bootstrap interval over questions. A cell at the ceiling (1.00 to 1.00) has no spread: every resample hits the same value, so the method found it every time on these questions, not on others.

**What a hit means.** On a multi-document question hit@k counts it when any one needed chunk is in the top k, and recall@k counts how many of the needed chunks are. Elsewhere a question has one needed chunk, so the two are equal.

```bash
cd backend && pip install -r requirements-ci.txt
python eval/run_eval.py                     # table 1: deterministic, no download, no network
python eval/run_eval.py --check             # same run, then fail on a fallback, a chunking change or a regression
python eval/run_eval.py --embedder real     # table 2: the repo's own embedding model, run locally
python eval/run_eval.py --markdown t.md     # the results table below, written from the run
python eval/mutation_check.py               # break the guarded code one way at a time (--skip-mcp drops the server mutants)
```

**Table 1, HARNESS CHECK (hashing embedder).** This embedder has no idea what any word means, so the table shows that the harness and the ranking code run and are pinned, and says nothing about retrieval quality. CI compares it with a fresh run.

| method | category | n | hit@1 | hit@3 | MRR | recall@3 | recall@5 |
|---|---|---|---|---|---|---|---|
| vector | direct | 12 | 0.83 (0.58 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.79 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| vector | paraphrase | 12 | 0.50 (0.25 to 0.75) | 0.67 (0.42 to 0.92) | 0.59 (0.35 to 0.83) | 0.67 (0.42 to 0.92) | 0.75 (0.50 to 1.00) |
| vector | multi-document | 8 | 0.88 (0.62 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.75 to 1.00) | 0.88 (0.69 to 1.00) | 1.00 (1.00 to 1.00) |
| bm25 | direct | 12 | 0.75 (0.50 to 1.00) | 1.00 (1.00 to 1.00) | 0.88 (0.75 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| bm25 | paraphrase | 12 | 0.42 (0.17 to 0.67) | 0.75 (0.50 to 1.00) | 0.54 (0.31 to 0.76) | 0.75 (0.50 to 1.00) | 0.75 (0.50 to 1.00) |
| bm25 | multi-document | 8 | 0.88 (0.62 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.75 to 1.00) | 0.75 (0.56 to 0.94) | 0.81 (0.62 to 1.00) |
| hybrid | direct | 12 | 0.83 (0.58 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.79 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| hybrid | paraphrase | 12 | 0.50 (0.25 to 0.75) | 0.75 (0.50 to 1.00) | 0.58 (0.33 to 0.83) | 0.75 (0.50 to 1.00) | 0.75 (0.50 to 1.00) |
| hybrid | multi-document | 8 | 0.88 (0.62 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.75 to 1.00) | 0.88 (0.75 to 1.00) | 0.94 (0.81 to 1.00) |
| chance | direct |  | 0.04 | 0.11 | 0.08 | 0.11 | 0.19 |
| chance | paraphrase |  | 0.04 | 0.11 | 0.08 | 0.11 | 0.19 |
| chance | multi-document |  | 0.07 | 0.21 | 0.16 | 0.11 | 0.19 |

Hybrid fallbacks in this run: 0.

**Table 2, real model (local run on 2026-10-23).** The embedding model is the repo's default, all-MiniLM-L6-v2 through `HuggingFaceEmbeddings` (torch 2.14.1 CPU, sentence-transformers as installed then, the repo pins 6.1.0, Python 3.13). The fastembed ONNX path printed the same numbers. CI does not repeat this run.

| method | category | n | hit@1 | hit@3 | MRR | recall@3 | recall@5 |
|---|---|---|---|---|---|---|---|
| vector | direct | 12 | 0.92 (0.75 to 1.00) | 1.00 (1.00 to 1.00) | 0.96 (0.88 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| vector | paraphrase | 12 | 0.67 (0.42 to 0.92) | 0.83 (0.58 to 1.00) | 0.72 (0.47 to 0.94) | 0.83 (0.58 to 1.00) | 0.83 (0.58 to 1.00) |
| vector | multi-document | 8 | 0.62 (0.25 to 0.88) | 1.00 (1.00 to 1.00) | 0.77 (0.54 to 1.00) | 0.69 (0.50 to 0.88) | 0.69 (0.56 to 0.88) |
| bm25 | direct | 12 | 0.75 (0.50 to 1.00) | 1.00 (1.00 to 1.00) | 0.88 (0.75 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| bm25 | paraphrase | 12 | 0.42 (0.17 to 0.67) | 0.75 (0.50 to 1.00) | 0.54 (0.31 to 0.76) | 0.75 (0.50 to 1.00) | 0.75 (0.50 to 1.00) |
| bm25 | multi-document | 8 | 0.88 (0.62 to 1.00) | 1.00 (1.00 to 1.00) | 0.92 (0.75 to 1.00) | 0.75 (0.56 to 0.94) | 0.81 (0.62 to 1.00) |
| hybrid | direct | 12 | 0.92 (0.75 to 1.00) | 1.00 (1.00 to 1.00) | 0.96 (0.88 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) |
| hybrid | paraphrase | 12 | 0.58 (0.25 to 0.83) | 0.83 (0.58 to 1.00) | 0.72 (0.52 to 0.92) | 0.83 (0.58 to 1.00) | 0.92 (0.75 to 1.00) |
| hybrid | multi-document | 8 | 0.62 (0.25 to 0.88) | 1.00 (1.00 to 1.00) | 0.77 (0.56 to 0.94) | 0.69 (0.56 to 0.88) | 0.88 (0.69 to 1.00) |

What it says: every method beats the chance rows by a wide margin. Hybrid did not beat vector on this set (hit@1 is equal on direct and multi-document questions and lower on paraphrase, every interval overlapping), so there is no direction to report. On the multi-document questions hit@3 is 1.00 for all three methods in both tables, but with the real model (2026-10-23) recall@3 is 0.69 for vector, 0.75 for BM25 and 0.69 for hybrid, so a hit there often means one of the needed chunks and not all of them.

**Unanswerable questions.** The retrieval code has no abstention, so the evaluation defines one: decline when the best chunk's score is below a threshold calibrated on a separate 16-question set (`backend/eval/calibration.jsonl`). With the real model (2026-10-23) 5 of the 10 unanswerable questions were declined (0.50, interval 0.20 to 0.80) while 84% (vector) and 75% (hybrid) of the answerable ones were kept. That is weak, and it wrongly declines paraphrase questions; the calibration questions are closer to the direct style than the paraphrase style.

## MCP server (read-only, local)

**Use case:** a Claude Code or other MCP client asks this knowledge base questions and gets answers with cited sources, through three read-only tools (`search_documents`, `ask_question`, `list_documents`). [`backend/mcp_server/server.py`](backend/mcp_server/server.py) runs over stdio on the official Python SDK, Python 3.11 or newer.

```bash
cd backend && pip install -r requirements.txt -r requirements-mcp.txt
# macOS and Linux
claude mcp add enterprise-rag -e QDRANT_URL=http://localhost:6333 -- /abs/backend/venv/bin/python /abs/backend/mcp_server/server.py
# Windows
claude mcp add enterprise-rag -e QDRANT_URL=http://localhost:6333 -- C:\abs\backend\venv\Scripts\python.exe C:\abs\backend\mcp_server\server.py
```

Use absolute paths (`abs` stands for your clone's location), and add `--demo-corpus` to try it with no Qdrant and no model download. Tools, citation ids, settings, errors and tests: [docs/mcp-server.md](docs/mcp-server.md).

**Limits:** local stdio only, read-only, not deployed, no rate limiting, and never run in Claude Desktop or Claude Code (I used the SDK's client and a raw JSON-RPC session). Citations are the chunks the model was given, not proof the answer is supported by them, and retrieved text may carry an instruction, so the client must treat it as untrusted.

## Agent skill (written scenarios, not results)

[`skills/querying-knowledge-base/`](skills/querying-knowledge-base/SKILL.md) is an Agent Skill that tells an agent when to search or ask, how to cite and how to recover when `ask_question` fails. Copy the folder to your client's skills directory (Claude Code: `.claude/skills/`). Its scenarios were written before the skill and **I have not run them against a live agent**: the tests only replay their tool calls against the server ([details](docs/evaluation.md#agent-skill)).

## Architecture

The diagram and the ingestion and query flow are in [docs/SYSTEM-KNOWLEDGE.md](docs/SYSTEM-KNOWLEDGE.md#the-shape-of-the-system). Hybrid retrieval (the default) adds a flat 0.15 bonus to any chunk BM25 also returns, so BM25 is a keyword-agreement bonus, not a normalised score. The streaming and OpenAI-compatible endpoints are in [docs/reference.md](docs/reference.md).

## Run it

You need Docker, Python 3.13+, Node 20.9+ and Ollama with one model pulled. The local path needs no account and no API key. `pip install` pulls torch and the first backend start downloads the all-MiniLM-L6-v2 model, so allow a few minutes. The commands use `llama3`; the recording used `qwen3:14b` ([how to match it](docs/reference.md#the-recording)).

```bash
docker run -d --name rag-qdrant -p 6333:6333 qdrant/qdrant && ollama pull llama3
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt   # Windows Git Bash: venv/Scripts/activate
QDRANT_URL=http://localhost:6333 python -m app.main            # http://localhost:8001, docs at /docs; this keeps running, so use a second terminal for the next line
cd frontend && npm install && echo "NEXT_PUBLIC_API_URL=http://localhost:8001/api" > .env.local && npm run dev
```

Sample documents: [`docs/sample-corpus/`](docs/sample-corpus/). Full steps, API examples, Groq and deployment: [docs/reference.md](docs/reference.md).

## Tests

```bash
cd backend && pip install -r requirements-ci.txt
python -m pytest tests/test_hybrid_ranking.py        # the ranking regression test, stubbed modules
python -m pytest tests/test_eval_*.py                # the evaluation tests (the shell expands the glob)
pip install -r requirements-mcp.txt                  # the MCP server tests need the SDK
python -m pytest tests/test_mcp_server*.py tests/test_skill_*.py
```

None of these needs a running service, a model download, a key or a network; run the pytest commands separately, as CI does. A test compares the corpus, question and table counts the docs state with a fresh run. Baseline failures, skipped files, counts and CI timings: [docs/evaluation.md](docs/evaluation.md#running-the-tests).

## Limits and not done

- Table 1 only proves the harness and the ranking code. Table 2 is a local run on my machine (2026-10-23) that CI does not repeat.
- The corpus is invented and small and I wrote the questions from it. That flatters retrieval and gives wide intervals, so hybrid against vector is a direction at best, and on this set not even that. It is a regression check, not a benchmark.
- The BM25 branch does not lowercase or strip punctuation and returns its top k chunks even with no matching term, so the flat 0.15 bonus can go to a chunk with no keyword overlap ([detail](docs/evaluation.md#known-limits-of-the-retrieval-code)).
- The fallback counter is read by the evaluation only, not by the API routes. CI runs Python 3.12 and my local runs 3.13; the Table 1 output matched.
- Cross-encoder reranking works locally only, and I have not measured it. HyDE and multi-query exist as library functions, not API routes, and are not evaluated.
- Smaller known issues (conversation memory lost on restart, CORS open to every origin, a health check that calls the model, an unmaintained `docker-compose.yml`, a `DELETE` that reports `chunks_deleted: 0`): [docs/reference.md](docs/reference.md#known-issues).

## Stack

Python 3.13, FastAPI, LangChain, LangGraph, Qdrant, Redis, fastembed or sentence-transformers (all-MiniLM-L6-v2), rank-bm25, Groq, Ollama, Next.js 16, React 19, TypeScript, Tailwind CSS.

## License

MIT, see [LICENSE](LICENSE). By Daniel Alexis Cruz, [dacruz.vercel.app](https://dacruz.vercel.app).
