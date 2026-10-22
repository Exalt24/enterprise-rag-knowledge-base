# Evaluation

How the rerunnable retrieval evaluation in `backend/eval/` works, and what each guard is for. The results are in the [README](../README.md#measured-results); this page is the reference.

## What runs

`python eval/run_eval.py` (from `backend/`) does this, all through the repo's own code:

1. Reads the four documents in `eval/corpus/` and chunks them with `TextChunker`, the same chunker ingestion uses. Each chunk gets a stable id built from its file name and chunk index, so the order of a scroll over the store, and therefore BM25 tie-breaks, is the same on every run.
2. Builds a `QdrantVectorStore` on `QdrantClient(":memory:")` and points the real `VectorStoreService` at it, then ingests through `VectorStoreService.add_documents`. Its search, scroll and stats code runs for real. The singleton's connection attributes are put back afterwards.
3. Runs three retrieval modes: vector through `retrieval_service`, BM25 through the `BM25Retriever` that `AdvancedRetrieval` builds, and hybrid through `AdvancedRetrieval.hybrid_search`. Each returns its top 5 chunks.
4. Scores every question and prints hit@1, hit@3, MRR, recall@3 and recall@5 per category, with a 95% bootstrap interval (2,000 resamples of the questions, fixed seed), the chance baseline, the number of hybrid fallbacks and the abstain thresholds.

The services print to stdout on every call, so the harness swallows that to keep the report readable.

## Embedders

The default is `HashingEmbeddings` in `eval/embedders.py`: signed feature hashing of lowercased words into 384 buckets, normalised, using blake2b so it is the same in every process. It subclasses `langchain_core.embeddings.Embeddings`, because `QdrantVectorStore` rejects a duck type with a `TypeError`. It is a test double. It cannot tell a synonym from an unrelated word, so a table from it shows that the harness runs, not how well retrieval works.

`--embedder real` uses the repo's own `embedding_service`, so it runs whichever backend the repo would pick: `HuggingFaceEmbeddings` when torch is installed, the fastembed ONNX adapter when it is not. The report title names the class that ran.

## Labels

A question has an `expected_document` and a `must_contain` phrase copied verbatim from that document. A multi-document question adds an `also` list with more document and phrase pairs. A retrieved chunk satisfies a need only when it belongs to that document and its text contains the phrase (case sensitive), so the label is at chunk level. For a question with several needs, hit@k asks whether any need is covered in the top k, and recall@k is the share of needs covered. An unanswerable question has empty fields and is scored on whether the system declines.

Labels are checked when a question file loads: a wrong document name, a phrase that is not in the document it names, an empty phrase, an unknown category, a repeated id or a broken line each raise one `LabelError` that names the question id and what is wrong, and `run_eval.py` prints it as `LABEL ERROR` and exits with 2 instead of failing later inside the metrics. A phrase that sits in its document but is cut in half by the splitter is named before the run starts. Tests prove that every phrase appears in exactly the document named, once, and inside a single chunk after chunking, that the calibration questions are different from the eval questions, and that the words which would answer each unanswerable question are in no corpus document.

The eval chunks with its own constants (`EVAL_CHUNK_SIZE` and `EVAL_CHUNK_OVERLAP` in `eval/harness.py`), not the app settings, so a `CHUNK_SIZE` in the environment cannot move the corpus under the baseline.

## Unanswerable questions

The retrieval code has no abstention. The evaluation defines the simplest one: decline when the best chunk's score is below a threshold. Each mode has its own score (cosine for vector, the highest BM25 score for BM25, the hybrid score for hybrid) and its own threshold. The threshold is picked on `eval/calibration.jsonl`, 16 questions kept apart from the eval set, as the midpoint that gives the best balanced accuracy (answerable kept plus unanswerable declined, halved). It is printed with every run.

## Baseline

`eval/baseline.json` holds the chunk count, a SHA-256 of every chunk's file, index and text, a SHA-256 of the parsed questions in `questions.jsonl` and in `calibration.jsonl` (so only a changed question or label moves it, not a reformatted file), and the deterministic point values. `python eval/run_eval.py --check` and `tests/test_eval_baseline.py` compare a run with it, and both print the same text.

- If the chunk count or the chunk hash differs, the message is `CHUNKING CHANGED` and no number is judged, because numbers on different chunks cannot be compared. A Dependabot bump of the text splitter shows up this way.
- If a question file differs, the message is `QUESTIONS CHANGED` and no number is judged either, because a fixed label is not a retrieval regression. It is printed before any number is compared.
- If the chunks and the questions are the same, each pinned number is judged one-sided. A number may rise freely (an improvement never fails). A number that fell by more than 0.02 (absolute) prints `REGRESSION: vector/direct/hit@1 fell from 0.83 to 0.5, by 0.33, more than the tolerance of 0.02`, naming the method, category and metric.
- A hybrid fallback fails the run on its own. The first exception behind it is kept (`AdvancedRetrieval.hybrid_fallback_cause`, type and message) and printed once, beside the count. The hybrid rows of such a run are vector numbers, so they are not judged and print `NOT COMPARED` instead of a second round of `REGRESSION` lines.

To re-pin after reading why it changed: `python eval/run_eval.py --write-baseline`.

The README tables come from the run: `python eval/run_eval.py --markdown t.md` writes the table (hit@1, hit@3, MRR, recall@3, recall@5 for every row), and `tests/test_eval_readme_claims.py` compares Table 1 with a fresh deterministic run cell by cell. Table 2 comes from a local real-model run that CI cannot repeat, so that test checks its shape and that its columns agree with each other.

## CI

The `python-backend` job installs `requirements-ci.txt` (the Render set plus pytest, no torch). It runs the stubbed ranking test, the evaluation tests and the MCP server tests as three separate pytest invocations (the MCP SDK is installed in its own step from `requirements-mcp.txt`), because `test_hybrid_ranking.py` replaces modules that fail to import with stubs for the whole process and the evaluation must run against the real ones. Then it runs `python eval/run_eval.py --check` and `python eval/mutation_check.py`, whose last line reads `N of N mutants caught`.

## Mutation check

`python eval/mutation_check.py` copies `app/`, `eval/`, `tests/` and `mcp_server/` to a temporary directory, applies one change found by walking the syntax tree (never by line number), runs the evaluation tests against the copy (the server tests for a server mutant) and requires pytest to exit with code 1, a real test failure. Exit code 0 means the mutant survived and the tests are weak. Any other code means the mutant did not run, which proves nothing. A target that cannot be found fails the check, and `tests/test_eval_mutation_targets.py` runs that cheap half on every test run so a renamed function cannot silently leave a mutant guarding nothing. `python eval/mutation_check.py --list` prints every mutant (and `| wc -l` counts them). They break:

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

The server mutants need the SDK (`pip install -r requirements-mcp.txt`); `--skip-mcp` leaves them out.

Most of these are caught by the generated class tests (`test_eval_baseline_changes.py`, `test_eval_fallback_cause.py`, `test_eval_readme_claims.py`, `test_eval_labels.py`), each of which builds its cases from a product of every field, exception, column or label kind, so a case nobody listed is covered too.

## MCP server

`backend/mcp_server/server.py` is a read-only MCP server over stdio. The README has the use case, the tool table and the `claude mcp add` command; this section is the reference for how it is built and checked.

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
