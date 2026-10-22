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

Tests prove that every phrase appears in exactly the document named, once, and inside a single chunk after chunking, and that the calibration questions are different from the eval questions.

## Unanswerable questions

The retrieval code has no abstention. The evaluation defines the simplest one: decline when the best chunk's score is below a threshold. Each mode has its own score (cosine for vector, the highest BM25 score for BM25, the hybrid score for hybrid) and its own threshold. The threshold is picked on `eval/calibration.jsonl`, 16 questions kept apart from the eval set, as the midpoint that gives the best balanced accuracy (answerable kept plus unanswerable declined, halved). It is printed with every run.

## Baseline

`eval/baseline.json` holds the chunk count, a SHA-256 of every chunk's file, index and text, and the deterministic point values. `python eval/run_eval.py --check` and `tests/test_eval_baseline.py` compare a run with it.

- If the chunk count or the hash differs, the message is `CHUNKING CHANGED` and no number is judged, because numbers on different chunks cannot be compared. A Dependabot bump of the text splitter shows up this way.
- If the chunks are the same and a number fell by more than 0.02 (absolute), the message is `REGRESSION`. A number may rise freely.
- A hybrid fallback fails the run on its own.

To re-pin after reading why it changed: `python eval/run_eval.py --write-baseline`.

## Mutation check

`python eval/mutation_check.py` copies `app/`, `eval/` and `tests/` to a temporary directory, applies one change found by walking the syntax tree (never by line number), runs the evaluation tests against the copy and requires pytest to exit with code 1, a real test failure. Exit code 0 means the mutant survived and the tests are weak. Any other code means the mutant did not run, which proves nothing. A target that cannot be found fails the check, and `tests/test_eval_mutation_targets.py` runs that cheap half on every test run so a renamed function cannot silently leave a mutant guarding nothing. `python eval/mutation_check.py --list` prints the ten mutants:

- the BM25 results removed from `hybrid_search`
- the keyword bonus subtracted instead of added
- the original inverted ranking, `1 / (1 + score)`, put back
- the fallback counter never incremented
- the abstain comparison flipped, and its boundary loosened from `<` to `<=`
- phrase matching negated, and an empty phrase allowed to match
- the baseline tolerance ignored
- the chunk hash not compared
