# MCP server

The read-only MCP server in `backend/mcp_server/server.py`: what it exposes, how to register it, what it checks and where it stops. The README has the use case and the one `claude mcp add` command; how it is built and tested is in [evaluation.md](evaluation.md#mcp-server).

**Use case:** a Claude Code or other MCP client asks this knowledge base questions and gets answers with cited sources. The server runs over stdio on the official Python SDK (`mcp`, pinned in [`backend/requirements-mcp.txt`](../backend/requirements-mcp.txt), Python 3.11 or newer).

## Tools

| tool | what it does | annotations |
|---|---|---|
| `search_documents(query, limit=5, document=None)` | Hybrid search. Hits carry a citation id, the chunk text (`snippet`) and a score. | read-only, closed world |
| `ask_question(question)` | Hybrid search, then the configured model. Returns the answer and the chunks it was given. | read-only, open world |
| `list_documents()` | The indexed documents and their chunk counts. | read-only, closed world |

A citation id is `file_name#chunk_index`, for example `risk-register-procedure.md#3`. `limit` is 1 to 20 and a query at most 500 characters. No tool writes, ingests or deletes. Every failure (empty query, bad limit, unknown document, Qdrant unreachable, no model answering) is an error result with a plain message, never error text inside a normal result.

## Register it

```bash
cd backend && pip install -r requirements.txt -r requirements-mcp.txt
# macOS and Linux
claude mcp add enterprise-rag -e QDRANT_URL=http://localhost:6333 -- /abs/backend/venv/bin/python /abs/backend/mcp_server/server.py
# Windows
claude mcp add enterprise-rag -e QDRANT_URL=http://localhost:6333 -- C:\abs\backend\venv\Scripts\python.exe C:\abs\backend\mcp_server\server.py
```

Use absolute paths; `abs` stands for the location of your clone, and the interpreter is the one from the virtual environment you installed into (`venv/bin/python` on macOS and Linux, `venv\Scripts\python.exe` on Windows). In a JSON `mcpServers` entry the Windows backslashes are doubled. The server calls the backend's services in its own process, so it needs the backend's settings: `QDRANT_URL` (and `QDRANT_API_KEY` for a cloud cluster), and for `ask_question` a running Ollama or `GROQ_API_KEY`. A `.env` is read from the client's working directory, so pass the values with `-e` (or the `env` block of a Claude Desktop `mcpServers` entry, which does not inherit your shell). Add `--demo-corpus` to try it with no Qdrant and no model download: it serves the synthetic corpus from memory with the hashing embedder, and `ask_question` then fails with a plain error unless a model is configured.

## Checked by

In-process tests through the SDK's `Client`, a generated table of every tool against an empty query, an unknown document, a generation that is down and an empty index, and a subprocess test that checks stdout holds JSON-RPC and nothing else (the services print to stdout). `ask_question` raises when the generation layer reports `model_used == "none"`, because that layer returns error text as a normal response. The mutation check breaks the server several ways and each must fail a named test. Commands for the counts and the rest: [evaluation.md](evaluation.md#mcp-server).

## Limits

- Local stdio only, read-only, not deployed, and not tested against a hosted client: I ran it through the SDK's client and a raw JSON-RPC session, and did not register it in Claude Desktop or Claude Code.
- No rate limiting.
- Citations are the chunks the model was given, not proof the answer is supported by them.
- One test shows an instruction inside a document comes back as `snippet` text and changes nothing on the server; a model reading it may still be misled, so the client must treat retrieved text as untrusted.
