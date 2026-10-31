# Using the MCP server from Hermes Agent

The read-only MCP server ([mcp-server.md](mcp-server.md)) was also connected to [Hermes Agent](https://github.com/NousResearch/hermes-agent) from Nous Research, an open source agent that runs skills written as Markdown and talks to MCP servers. This page says what was run and what happened. The raw session logs are in [hermes/evidence/sessions.md](hermes/evidence/sessions.md).

**Use case:** a Hermes agent answers questions from the indexed documents, cites each claim with its `file_name#chunk_index` id, and says so when the documents do not cover the question or a tool fails.

## Setup that was run

Hermes Agent 2026.9.24 on Windows, installed from its lockfile into its own folder and its own home directory so nothing else on the machine changed. Python 3.14 was needed; the first environment on 3.13 would not install.

```bash
hermes mcp add enterprise-rag --command /abs/backend/venv/Scripts/python.exe --args /abs/backend/mcp_server/server.py --demo-corpus
hermes mcp test enterprise-rag        # lists the three tools
```

Hermes registers each tool as `mcp__enterprise_rag__<tool>` (the server's resource and prompt calls come along, seven in all). The server runs with `--demo-corpus`, so the four synthetic documents in `backend/eval/corpus` were the index, with no Qdrant. The Agent Skill in [`skills/querying-knowledge-base/`](../skills/querying-knowledge-base/SKILL.md) loads into Hermes unchanged: copy the folder into the Hermes home's `skills/` directory and start a session with `-s querying-knowledge-base`.

```bash
hermes chat -Q -t enterprise-rag -s querying-knowledge-base --query-file question.txt
```

## What happened

| Session | Question | Result |
|---|---|---|
| Hosted Llama 3.3 70B with the skill | How is the contingency fund used? | `ask_question` failed because no answering model was configured behind the server. The agent followed the skill, switched to `search_documents`, and answered in two sentences citing `risk-register-procedure.md#5` and `lessons-learned-note.md#4`. |
| Hosted Llama 3.3 70B with the skill | What is the customer refund policy? | A first call used the wrong argument name and Hermes rejected it before the server saw it. The retry searched, found nothing relevant (best score 0.15) and the agent said the documents do not say. |
| Local qwen3.5 9B | Search with an empty query | The server returned its plain error. The agent reported the error and the documents it did not retrieve, with nothing made up. |

## Limits

- A local 9B model through Ollama was not reliable here. Of nine sessions run after the tools registered, three called the tools and reported what came back; the others skipped the tools, searched Hermes' own tool list instead, or named documents that are not in the index. Hosted 70B model sessions did the job, but only two were run before the account's credits ran out.
- Three sessions are a trial, not an evaluation. The retrieval numbers are in [evaluation.md](evaluation.md).
- Only the demo corpus was used, over stdio, on one machine. The Hermes messaging gateway (Telegram, Discord, Slack) was not set up.
- Citations are the chunks the tool returned. They do not prove the answer is supported by them, and the 9B model's longer answers added statements the chunks do not make.
