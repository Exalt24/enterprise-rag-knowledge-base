# Evaluation

How the rerunnable retrieval evaluation in `backend/eval/` works, and what each guard is for. The results are in the [README](../README.md#measured-results); this page is the reference.

## What runs

`python eval/run_eval.py` (from `backend/`) does this, all through the repo's own code:

1. Reads the four documents in `eval/corpus/` (a risk register procedure, a change control procedure, a monthly cost report guide and a lessons learned note on an invented project) and chunks them with `TextChunker`, the same chunker ingestion uses. Each chunk gets a stable id built from its file name and chunk index, so the order of a scroll over the store, and therefore BM25 tie-breaks, is the same on every run.
2. Builds a `QdrantVectorStore` on `QdrantClient(":memory:")` and points the real `VectorStoreService` at it, then ingests through `VectorStoreService.add_documents`. Its search, scroll and stats code runs for real. The singleton's connection attributes are put back afterwards.
3. Runs three retrieval modes: vector through `retrieval_service`, BM25 through the `BM25Retriever` that `AdvancedRetrieval` builds, and hybrid through `AdvancedRetrieval.hybrid_search`. Each returns its top 5 chunks.
4. Scores every question and prints hit@1, hit@3, MRR, recall@3 and recall@5 per category, with a 95% bootstrap interval (2,000 resamples of the questions, fixed seed), the chance baseline, the number of hybrid fallbacks and the abstain thresholds.

The services print to stdout on every call, so the harness swallows that to keep the report readable.

## Reading the tables

The README tables show each value with a 95% bootstrap interval over questions. With so few questions per category the intervals are wide, and the chance rows show what a random ranking of the same chunks scores. A cell at the ceiling (1.00 to 1.00) has no spread because every resample of the questions hits the same value; it means the method found it every time on these questions, not that it would on others.

**What a hit means.** A multi-document question needs chunks from more than one document. hit@k counts it when any one needed chunk is in the top k; recall@k counts how many of the needed chunks are. So a hit@3 of 1.00 on a multi-document row says every question had at least one of its chunks in the top 3, and the recall columns say how much of what it needed was found. For the other categories a question has one needed chunk, so hit and recall are the same number.

**Table 1 is a harness check.** Its embedder hashes words into a vector and has no idea what any of them mean, so the table shows that the harness and the ranking code run and are pinned. No quality claim and no "hybrid beats vector" claim comes from it. `eval/baseline.json` pins its chunk count, a hash of the chunk texts, a hash of each question file and its numbers.

## Embedders

The default is `HashingEmbeddings` in `eval/embedders.py`: signed feature hashing of lowercased words into 384 buckets, normalised, using blake2b so it is the same in every process. It subclasses `langchain_core.embeddings.Embeddings`, because `QdrantVectorStore` rejects a duck type with a `TypeError`. It is a test double. It cannot tell a synonym from an unrelated word, so a table from it shows that the harness runs, not how well retrieval works.

`--embedder real` uses the repo's own `embedding_service`, so it runs whichever backend the repo would pick: `HuggingFaceEmbeddings` when torch is installed, the fastembed ONNX adapter when it is not. The report title names the class that ran.

## Labels

A question has an `expected_document` and a `must_contain` phrase copied verbatim from that document. A multi-document question adds an `also` list with more document and phrase pairs. A retrieved chunk satisfies a need only when it belongs to that document and its text contains the phrase (case sensitive), so the label is at chunk level. For a question with several needs, hit@k asks whether any need is covered in the top k, and recall@k is the share of needs covered. An unanswerable question has empty fields and is scored on whether the system declines.

Labels are checked when a question file loads: a wrong document name, a phrase that is not in the document it names, an empty phrase, an unknown category, a repeated id or a broken line each raise one `LabelError` that names the question id and what is wrong, and `run_eval.py` prints it as `LABEL ERROR` and exits with 2 instead of failing later inside the metrics. A phrase that sits in its document but is cut in half by the splitter is named before the run starts. Tests prove that every phrase appears in exactly the document named, once, and inside a single chunk after chunking, that the calibration questions are different from the eval questions, and that the words which would answer each unanswerable question are in no corpus document.

The eval chunks with its own constants (`EVAL_CHUNK_SIZE` and `EVAL_CHUNK_OVERLAP` in `eval/harness.py`), not the app settings, so a `CHUNK_SIZE` in the environment cannot move the corpus under the baseline.

## Unanswerable questions

The retrieval code has no abstention. The evaluation defines the simplest one: decline when the best chunk's score is below a threshold. Each mode has its own score (cosine for vector, the highest BM25 score for BM25, the hybrid score for hybrid) and its own threshold. The threshold is picked on `eval/calibration.jsonl`, 16 calibration questions kept apart from the eval set, as the midpoint that gives the best balanced accuracy (answerable kept plus unanswerable declined, halved). It is printed with every run.

## Baseline

`eval/baseline.json` holds the chunk count, a SHA-256 of every chunk's file, index and text, a SHA-256 of the parsed questions in `questions.jsonl` and in `calibration.jsonl` (so only a changed question or label moves it, not a reformatted file), and the deterministic point values. `python eval/run_eval.py --check` and `tests/test_eval_baseline.py` compare a run with it, and both print the same text.

- If the chunk count or the chunk hash differs, the message is `CHUNKING CHANGED` and no number is judged, because numbers on different chunks cannot be compared. A Dependabot bump of the text splitter shows up this way.
- If a question file differs, the message is `QUESTIONS CHANGED` and no number is judged either, because a fixed label is not a retrieval regression. It is printed before any number is compared.
- If the chunks and the questions are the same, each pinned number is judged one-sided. A number may rise freely (an improvement never fails). A number that fell by more than 0.02 (absolute) prints `REGRESSION: vector/direct/hit@1 fell from 0.83 to 0.5, by 0.33, more than the tolerance of 0.02`, naming the method, category and metric.
- A hybrid fallback fails the run on its own. The first exception behind it is kept (`AdvancedRetrieval.hybrid_fallback_cause`, type and message) and printed once, beside the count. The hybrid rows of such a run are vector numbers, so they are not judged and print `NOT COMPARED` instead of a second round of `REGRESSION` lines.

To re-pin after reading why it changed: `python eval/run_eval.py --write-baseline`.

The README tables come from the run: `python eval/run_eval.py --markdown t.md` writes the table (hit@1, hit@3, MRR, recall@3, recall@5 for every row), and `tests/test_eval_readme_claims.py` compares Table 1 with a fresh deterministic run cell by cell. Table 2 comes from a local real-model run that CI cannot repeat, so that test checks its shape and that its columns agree with each other.

**Every count the docs state is checked.** The chunk count, the number of documents, the question total, each category's question count, the calibration set size, the `n` column of both tables, the hybrid fallback count, the bootstrap resamples and the number of skill scenarios are compared with a fresh run by `tests/test_eval_readme_counts.py`. It scans every markdown page of the repo for each of them (`tests/doc_pages.py` finds the pages: the root, `docs/`, `skills/` and anything they link, so a page added later is read the day it is added), so a number written in prose and not in its table fails the test the moment the corpus or a question file changes. When a count moves, re-pin (`--write-baseline`), paste the new table with `--markdown`, and change the sentence that states it; the failing test names which one. Figures that only a real-model run produces cannot be rerun in CI: the same file requires every sentence about Table 2 or the real model to carry the date of its run, and Table 2's `n` column must still equal the current question counts, so adding a question means running `--embedder real` again.

**Claims that stopped being true are checked too.** `tests/test_eval_docs_not_stale.py` crosses every markdown page with a list of sentences that used to be true about CI, the tests and the evaluation (the patterns are in the file) and fails when one comes back, and it requires the design notes to mention the workflow, the mutation check and the MCP server. A paragraph dated to the earlier live-deployment run is a record and is skipped. The design notes once kept such a claim for weeks because the count test only read three pages.

## Running the tests

The README has the three pytest commands. None of them needs a running service, a model download, a language model key or a network. Run them separately, as CI does. `backend/pytest.ini` skips what cannot run on a plain machine: the files that need the backend on port 8001 or Qdrant and an LLM, and `backend/test_setup.py`, which starts services at import and crashes a bare `pytest` during collection. Those are run by hand against a local backend.

`eval/baseline.json` fails with a different message for each cause: `CHUNKING CHANGED` when the chunks changed (a text splitter bump), `QUESTIONS CHANGED` when a question or label was edited, `REGRESSION` when a pinned number fell by more than 0.02. A number may rise freely. To print how many tests and mutants there are, add `--collect-only -q` to a pytest command and read its last line, and run `python eval/mutation_check.py --list | wc -l`.

## CI

Timing from the first run on a cold pip cache, read from the job log (run 39863561524, 2026-10-22): install 31 seconds, whole job 146 seconds, against a 15 minute timeout. That run predates the MCP tests and their mutants; the current job takes about 5 minutes.

The `python-backend` job runs on every push and installs `requirements-ci.txt` (the Render set plus pytest, no torch). It runs the stubbed ranking test, the evaluation tests and the MCP server tests as three separate pytest invocations (the MCP SDK is installed in its own step from `requirements-mcp.txt`), because `test_hybrid_ranking.py` replaces modules that fail to import with stubs for the whole process and the evaluation must run against the real ones. Then it runs `python eval/run_eval.py --check` and `python eval/mutation_check.py`, whose last line reads `N of N mutants caught`.

## Known limits of the retrieval code

Two behaviours of the BM25 branch showed up while building the evaluation and are not changed. `BM25Retriever` splits on whitespace without lowercasing or stripping punctuation, so `Risks?` and `risks` are different terms. It also returns its top k chunks even when no query term matches, so the flat 0.15 bonus can go to a chunk with no keyword overlap. The fallback counter is read by the evaluation only, not by the API routes.

## Mutation check

`python eval/mutation_check.py` copies `app/`, `eval/`, `tests/` and `mcp_server/` (plus the README, `docs/` and `skills/` for the tests that read them) to a temporary directory, applies one change found by walking the syntax tree (never by line number), runs the evaluation tests against the copy (the server tests for a server mutant) and requires pytest to exit with code 1, a real test failure. Exit code 0 means the mutant survived and the tests are weak. Any other code means the mutant did not run, which proves nothing. A target that cannot be found fails the check, and `tests/test_eval_mutation_targets.py` runs that cheap half on every test run so a renamed function cannot silently leave a mutant guarding nothing. `python eval/mutation_check.py --list` prints every mutant (and `| wc -l` counts them). They break:

- the BM25 results removed from `hybrid_search`
- the keyword bonus subtracted instead of added
- the original inverted ranking, `1 / (1 + score)`, put back
- the fallback counter never incremented, the fallback cause never kept, or the cause overwritten by a later fault
- the abstain comparison flipped, and its boundary loosened from `<` to `<=`
- phrase matching negated, and an empty phrase allowed to match
- the baseline tolerance ignored, a rise judged like a fall, the chunk hash or the question hashes not compared, a method that fell back still judged
- the README table losing its recall columns
- the bootstrap interval read from the wrong percentile at either end
- the eval chunking following the environment, and label validation skipped
- the MCP server, in `mcp_server/server.py`: see the next section
- the docs, which are edited as text rather than code: the README chunk count left stale, a question total, a category count or a Table 1 `n` cell edited by one, Table 2 losing the date of its run, the calibration count in this page edited by one, and the design notes denying again that CI runs any test. Each must fail a named test in `test_eval_readme_counts.py` or `test_eval_docs_not_stale.py`.

The server mutants need the SDK (`pip install -r requirements-mcp.txt`); `--skip-mcp` leaves them out.

Most of these are caught by the generated class tests (`test_eval_baseline_changes.py`, `test_eval_fallback_cause.py`, `test_eval_readme_claims.py`, `test_eval_readme_counts.py`, `test_eval_docs_not_stale.py`, `test_eval_labels.py`), each of which builds its cases from a product of every field, exception, column or label kind, so a case nobody listed is covered too.

## MCP server

`backend/mcp_server/server.py` is a read-only MCP server over stdio. The README has the use case and the `claude mcp add` command, [mcp-server.md](mcp-server.md) has the tool table, settings and limits; this section is the reference for how it is built and checked.

**SDK.** `mcp==2.3.0` (`backend/requirements-mcp.txt`), Python 3.11 or newer; CI runs 3.12 and the author's machine 3.13. Before any tool was written the imports were checked against this release: `from mcp.server import MCPServer`, `from mcp import Client` (the same class as `mcp.client.Client`), `from mcp.server.mcpserver.exceptions import ToolError` and `from mcp_types import ToolAnnotations` (`ToolAnnotations` is not exported by `mcp` itself). `tests/test_mcp_server.py` keeps that proof: it fails if the pin and the installed release disagree or an import moves.

**stdout.** The existing services print progress lines to stdout, at import and on every call, and stdout is the protocol channel, so one stray line corrupts the stream. The service imports run inside `contextlib.redirect_stdout(sys.stderr)`, every tool call runs the same way, and the SDK additionally diverts file descriptor 1 to stderr while it serves. `tests/test_mcp_server_stdio.py` starts `python -m mcp_server.server --demo-corpus` as a subprocess, sends a JSON-RPC session and requires every stdout line to be a JSON-RPC message, with the progress lines on stderr (so the test cannot pass by the services printing nothing). A global swap of `sys.stdout` was not used: the SDK reads `sys.stdout` when it starts serving, so a permanent swap would send the protocol to stderr.

**Errors.** Any failure becomes a `ToolError` with a plain message: an empty or over-long query, a `limit` outside 1 to 20, an unknown `document` (the message lists the real ones), an unreachable vector store (the message names `QDRANT_URL`), any other exception (type and message, no traceback). `generation_service.generate` does not raise when every provider fails: it returns a normal response whose text is `[X] All LLM providers unavailable...` and whose `model_used` is `"none"`. `ask_question` turns that into a `ToolError`, so a client never reads the error text as an answer. An empty index returns an empty result with a message, and `ask_question` then calls no model.

**Citations.** The id is `file_name#chunk_index`, built from the chunk's own metadata, returned by `search_documents` and `ask_question`. `ask_question` does not call `rag_service.query`: its sources carry only file name, page and a 100 character preview, so no id can be built from them, and the response cache could return sources from before this field existed. It runs the same steps itself (`hybrid_search`, `format_context`, `generate`), so the citations are exactly the chunks the model was given. Tests resolve every returned id against the chunker's output, built separately from the server.

**Document filter.** `document` is an addition to the `search_documents(query, limit)` signature in the plan: it lets an agent scope a search, and gives the unknown-document failure something to fail on. It filters the top 100 ranked candidates, so a document whose chunks all rank below that is not reached.

**Tests.** None needs Qdrant, a language model key or a model download: the index is `QdrantClient(":memory:")` through the real `VectorStoreService` with the hashing embedder, and generation is stubbed where an answer is needed.

- `tests/test_mcp_server.py`: the tool list and annotations, search results and bounds, citation ids against the chunks, `ask_question` with a stub model and with a failing one, backend exceptions, stdout in process, an instruction inside a document returned as data.
- `tests/test_mcp_server_classes.py`: a generated table of every tool against four failure kinds (the cases are an `itertools.product`, each pair with its expected outcome: a readable error, an empty result, a normal result or not applicable, which is proved from the tool's input schema).
- `tests/test_mcp_server_stdio.py`: the real process over real stdio, once with raw JSON-RPC lines and once through the SDK's own stdio client.

Print the counts with `python -m pytest tests/test_mcp_server.py tests/test_mcp_server_classes.py tests/test_mcp_server_stdio.py --collect-only -q` (last line) and `python eval/mutation_check.py --list | grep -c mcp-`.

**Mutants.** The MCP mutants break `mcp_server/server.py`: the import-time stdout redirect removed, the per-call redirect removed, a citation id built from `total_chunks`, the `model_used == "none"` check removed, a `delete_document` tool registered, `read_only_hint` dropped, the limit check removed, the query length cap removed, the document filter ignored. Each must fail a test, and the runner prints which.

**Not done, on purpose.** No rate limiting. Chunks without a file name or chunk index are left out of results (the chunker always sets both, so this only affects foreign data). `hybrid_search` still falls back to vector-only silently on an internal error and the server does not report that. The server was not registered in Claude Desktop or Claude Code, and nothing here was run against a hosted client.

## Agent skill

`skills/querying-knowledge-base/` holds `SKILL.md` (frontmatter with `name` and `description` only, a body well under 500 lines) and `errors.md` (one level deep, linked from the body). The scenarios in `evaluations/` were committed before the skill, following the Agent Skills guidance to write evaluations first; `git log` shows the order.

Each scenario file has a user request, the steps a correct agent takes, the failures it catches, and a JSON block the tests read: the tool calls with their arguments and expected outcome, the citation ids a correct answer uses, phrases that must sit in those chunks, terms that must be absent (for the "nothing found" case) and, for the instruction-in-a-document case, an extra chunk to add to the demo index.

- `tests/test_skill_scenarios.py`: a generated table of every scenario against every required element (sections present, a JSON block with every key, only real tools, arguments and fields, well-formed ids, citations only from what a call returns, facts inside the cited chunks), and a replay of each scenario's calls on the demo index through the in-process client.
- `tests/test_skill_file.py`: the format rules (allowed fields, name and description limits, third person, body length, links one level deep), the tools and fields the skill names against the server's own schemas, the example in the skill against what the server returns, `errors.md` against the real error messages in both directions, and a table of scenario against what the skill must hold for it.

What this does not show: that a model follows the skill. No scenario has been run against a live agent. The replay uses the hashing embedder, so it proves the scenario's data exists and is reachable with the stated calls, not that the real embedding model ranks the same chunk first.
