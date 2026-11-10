# Archive ingestion with Hatchet

A data room arrives as one zip with hundreds or thousands of files in it. `ingest_archive` takes the zip, checks it, and starts one run per file that parses it, splits it, embeds it and stores it in Qdrant. [Hatchet](https://docs.hatchet.run) is the orchestrator: it queues the runs, hands them to a worker, retries the ones that deserve it and keeps the history. The loaders, the chunker and the vector store are the repo's own services (`document_parser.py`, `chunking.py`, `vector_store.py`), called as they are.

The single file upload (`POST /api/ingest`) is unchanged.

## Run it

Pinned versions: Hatchet Python SDK **1.45.1** (`backend/requirements-hatchet.txt`), hatchet-lite **v0.114.2**, Postgres 15.6, Qdrant v1.20.0 (`docker-compose.hatchet.yml`). Docker is the only other thing you need.

```bash
# 1. Postgres, hatchet-lite (engine, API and dashboard in one container) and Qdrant
docker compose -f docker-compose.hatchet.yml up -d --wait postgres hatchet-lite qdrant

# 2. A client token from the lite container (Git Bash on Windows: put MSYS_NO_PATHCONV=1 in front)
export HATCHET_CLIENT_TOKEN=$(docker compose -f docker-compose.hatchet.yml exec -T hatchet-lite /hatchet-admin token create --config /config --tenant-id 707d0855-80ab-4e1f-a156-f1c4546cbf52)

# 3. The worker, built from backend/Dockerfile.worker
docker compose -f docker-compose.hatchet.yml --profile worker up -d --build worker

# 4. A generated archive (invented text, mixed formats, a few bad files on purpose), then run it
mkdir -p ingest-data && python backend/eval/archive_fixture.py ingest-data/room.zip --files 300
docker compose -f docker-compose.hatchet.yml --profile worker run --rm worker python -m app.workflows run /data/room.zip
```

The dashboard is at http://localhost:8888 (`admin@example.com`, `Admin123!!`, the lite image's own seeded login for local use). Stop and remove everything with `docker compose -f docker-compose.hatchet.yml --profile worker down -v`.

On Linux or macOS the worker can run without Docker: `pip install -r requirements.txt -r requirements-hatchet.txt`, set `HATCHET_CLIENT_TOKEN`, `HATCHET_CLIENT_TLS_STRATEGY=none` and `QDRANT_URL`, then `cd backend && python -m app.workflows worker`. The SDK's worker registers `SIGQUIT`, which Windows does not have, so on Windows the worker runs in the container above.

Through the API (the zip must be a path the worker can read, which is why the API and the worker share the spool folder `INGEST_SPOOL_DIR`):

```bash
curl -F "file=@room.zip" http://localhost:8001/api/ingest/archive        # 202, {"run_id": "...", "status_url": "/api/ingest/<run_id>"}
curl http://localhost:8001/api/ingest/<run_id>                           # the result table, while it runs and after
```

## What runs

```
ingest_archive                            ingest_file (one run per file)
  validate  ->  fan_out  ------------>      parse  ->  chunk  ->  embed_and_upsert
  check, unpack, skip what is               DocumentParser   TextChunker   VectorStoreService
  already indexed                           (no retries)     (no retries)  (retries with backoff)
```

- `validate` checks the zip (below), unpacks the fit files under their SHA-256, drops files whose bytes were already seen in this archive or are already indexed, and writes the list of files still to do.
- `fan_out` starts one `ingest_file` run per file, in batches, waits for them with `return_exceptions=True` (a failed run comes back as a value, not as a reason to stop), and writes the result table.
- `parse`, `chunk` and `embed_and_upsert` hand each other small files in the run's work folder. Only counts go through Hatchet, so a large file does not travel through the orchestrator's database.
- Embedding and upsert are one task. The repo's `VectorStoreService.add_documents` embeds a batch and upserts it in one call, and calling it as it is was the brief. The vectors never leave the worker.

All the rules are plain functions in `backend/app/workflows/pipeline.py` and `archive_guard.py`. `ingest.py` is the thin Hatchet layer. That is what lets the tests drive the rules with a stub orchestrator and no Hatchet.

## What is refused

The zip is treated as hostile. Everything below reads the zip's own file table before anything is written.

| Check | Result |
|---|---|
| not a zip, or a damaged file table | whole archive refused, `not_a_zip` |
| the path cannot be opened (a typo, or a path the worker cannot see) | whole archive refused, `unreadable_archive` |
| an entry name that leaves the unpack folder (`../x`, `/x`, `C:/x`, backslash forms) | whole archive refused, `path_traversal` |
| more files than `ARCHIVE_MAX_FILES` (5000) | whole archive refused, `too_many_files` |
| entries that add up to more than `ARCHIVE_MAX_TOTAL_MB` (1024) once unpacked, read from the file table | whole archive refused, `archive_too_large` (the zip bomb guard) |
| no readable file left | whole archive refused, `empty_archive` |
| a file over `MAX_FILE_SIZE_MB` (10, the same cap as the single upload) | that file is a failed row, `file_too_large`, the rest go on |
| an extension the loaders do not read | failed row, `extension_not_allowed` |
| a zip, 7z, rar, tar or gz inside the zip | failed row, `nested_archive`, never opened |
| an entry stored encrypted | failed row, `password_protected` |
| an entry whose data fails the zip's own CRC | failed row, `corrupt_entry` |
| `__MACOSX/`, `._x`, `.DS_Store`, `Thumbs.db` | ignored, counted in `ignored` |

Entry names never reach the file system: each file is written under its SHA-256 and renamed once the digest is known, so even a name the checks missed could not choose where a file lands. A refused archive is the archive's fault, so the `validate` task raises Hatchet's `NonRetryableException` and the status says `rejected` with the code. An error nobody planned for (a full disk, say) ends the run with state `failed` and the error in the status, so a run never sits at `running` for ever.

## Idempotency

A file's identity is the SHA-256 of its bytes (`doc_id`). Before a file is started, the vector store is asked whether every chunk of that `doc_id` is stored: the stored count must have reached `doc_chunks_total`, the number written into each chunk at ingestion. So:

- the same archive again does no new work: every file is `skipped_duplicate`, `to_process` is 0, nothing is parsed or written;
- a renamed or re-zipped copy of an indexed file is recognised by content, not by name;
- a crash that stopped the run between two batches leaves a file with only some of its chunks stored. That file does not count as indexed, so the next run does exactly that file again, and nothing else;
- a chunk's point id is derived from `doc_id` and its position, so writing it again replaces it. A retried task or a rerun never leaves a second copy.

## Failures and retries

| Where | What happens |
|---|---|
| a corrupt, locked, empty or unreadable file in `parse` or `chunk` | `BadFile`, a failed row with the loader's reason, Hatchet's `NonRetryableException`, `retries=0`. The same bytes would fail the same way, so there is no second attempt. |
| the embedding model or Qdrant unreachable in `embed_and_upsert` | the error is raised as it is and Hatchet retries it: three retries, waiting 2, 4, 8 seconds up to 30 (`backoff_factor=2.0`, `backoff_max_seconds=30`). The failed row is written only on the last attempt, so a file that succeeds on a retry never shows a failure. |
| a file run that ends without reporting | `fan_out` writes a failed row with the error Hatchet returned |
| the whole archive is refused | `rejected`, no rows, nothing stored |

One bad file never stops the others: each file is its own Hatchet run, and `fan_out` collects them as values.

## The result table

Each row is `{file, status, reason, chunks, doc_id}` with `status` one of `ingested`, `skipped_duplicate`, `failed`. The workers write one small JSON file per row under `<spool>/runs/<run_id>/rows/`; `GET /api/ingest/{run_id}` reads them with the manifest (state, counts, seconds, the refusal if any). A run id has to look like the UUIDs Hatchet issues, because it becomes a folder name.

```
run 98e08a2c-...  state completed  archive a300.zip
  skipped_duplicate      0  copies/doc-0000-copy.txt  same content as leases/doc-0000.txt
  failed                 0  damaged/corrupt-report.pdf  unreadable_file: PDF corrupted or encrypted: damaged/corrupt-report.pdf - Stream has ended unexpectedly
  failed                 0  damaged/locked-report.pdf  unreadable_file: PDF corrupted or encrypted: damaged/locked-report.pdf - File has not been decrypted
  failed                 0  damaged/nested.zip  nested_archive: damaged/nested.zip is an archive inside the archive; nested archives are not opened
  failed                 0  images/site-photo.png  extension_not_allowed: images/site-photo.png has an extension the loaders do not read (allowed: .docx, .md, .pdf, .txt, .xlsx)
  ...
```

## Measured

All runs on 2026-11-10, one laptop class machine with Docker Desktop: hatchet-lite v0.114.2, one worker container (the default 100 slots), Qdrant v1.20.0 and the repo's own embedding model (all-MiniLM-L6-v2 on ONNX Runtime through fastembed). The archives come from `eval/archive_fixture.py`: invented text, with a few bad files on purpose. The model was loaded by a small warm-up run first, so no run below pays for the download. "Seconds" is what the run reports, from the start of the zip check to the last file. "Submit to done" is what a person waits, and it includes Hatchet scheduling the run.

| run | files | ingested | skipped duplicate | failed | chunks stored | seconds | submit to done |
|---|---|---|---|---|---|---|---|
| mixed archive, first run | 300 | 293 | 2 | 5 | 919 | 39.89 | 42.8 |
| the same archive again | 300 | 0 | 295 | 5 | 0 | 6.29 | 9.1 |
| the same archive a third time | 300 | 0 | 295 | 5 | 0 | 6.19 | 9.0 |
| small generated text files | 3000 | 3000 | 0 | 0 | 4125 | 481.37 | 484.0 |

- **The mixed archive** holds documents in five formats (txt, md, pdf, docx, xlsx) spread over folders, plus seven entries on purpose: two exact copies of other files, a corrupt PDF, a password protected PDF, a `.xlsx` that is not a workbook, a nested zip and a `.png`. The two copies are `skipped_duplicate`. The five others are failed rows, each with its own reason: the nested zip (`nested_archive`) and the `.png` (`extension_not_allowed`) are set aside by the guard, and the other three are read by the loaders and refused (`unreadable_file`). Every other file was stored.
- **The rerun** wrote nothing and found every stored file by its hash. It still started a run for the three unreadable files, because nothing was ever stored for them, and they failed the same way. That is the 3 in `to_process`, and the reason its seconds are not near zero: each of those runs goes through Hatchet.
- **The 3,000 file run** is plain text, a few paragraphs each, so it measures the per file cost of the orchestration more than the cost of parsing. It is about 6 files a second with one worker. An earlier run of the same size took 643 seconds, and about 4 minutes of that was one task sitting queued with free slots until another run started and woke the scheduler. I saw it once and did not find the cause, so the table keeps the clean run.
- **Not measured:** more than one worker, 3,000 files of mixed formats, a different embedder, a real data room.

`tests/test_hatchet_integration.py` repeats the behaviours that matter against the same containers and passed on 2026-11-10: the mixed archive and its table, the rerun, an outage retried with backoff and a file that gave up after the retries (both through a fault switch in the worker, `app/workflows/testing_hooks.py`), a zip slip archive refused, and a worker killed in the middle of a 400 file run and then a rerun that finished only what was missing.

## Limits

- Not production. hatchet-lite is Hatchet's own development and low volume image, and this is one worker, one machine, one Postgres.
- The API and the worker must see the same spool folder: one machine or one shared volume. The unpacked files and the stage files go through it.
- A file is one run and one worker slot. There is no splitting of one huge file across workers, and the per file cap is the single upload's 10 MB.
- Archive documents are named by their path inside the zip. The file list's delete button works from files on disk, so it cannot delete them.
- Status is read from files, not from Hatchet. A run id nobody recorded is a 404, and a run whose worker is not running stays `queued`.
- The measurements use generated, invented text on one laptop class machine. They say how this code behaves, not how a real data room would.
- OCR is not wired into archive runs, so a scanned PDF with no text layer is a failed row.

## Tests

```bash
cd backend && pip install -r requirements-ci.txt -r requirements-hatchet.txt
python -m pytest tests/test_archive_guard.py tests/test_archive_pipeline.py tests/test_archive_api.py tests/test_source_view.py tests/test_hatchet_tasks.py
python eval/mutation_check.py --only-ingest              # the last line reads "N of N mutants caught"
python -m pytest -m hatchet tests/test_hatchet_integration.py   # real hatchet-lite, worker and Qdrant in Docker
```

- `test_archive_guard.py` has a case that must be refused or set aside and a case just inside the limit for every rule above.
- `test_archive_pipeline.py` runs the rules with a stub orchestrator (a few lines that run each file job in the test process) against the repo's own vector store on an in-memory Qdrant: idempotency, a rerun after a crash (the embedder is made to fail in the middle of one file), isolation of bad files, retries only for outages, a refused archive leaving nothing behind, the status files.
- `test_hatchet_tasks.py` runs the task functions through the SDK's `mock_run` and pins the retry settings.
- `test_hatchet_integration.py` is the same story against the containers: a mixed archive, the rerun, an outage retried with backoff, a file that exhausts its retries, a zip slip archive, and a worker killed in the middle of a run followed by a rerun. It is marked `hatchet` and skipped when Docker is absent. CI runs it in its own Linux job on every push to main and every pull request.
- `eval/mutation_check.py` breaks every guard above, one change at a time, in a copy of the code, and requires a failing test. The Hatchet task settings (which task retries, the backoff, which exception a bad file raises) are mutated too and need `requirements-hatchet.txt`. What no offline test can reach is only `fan_out` collecting failed runs as values; the integration test covers that, because one bad file would otherwise stop the archive.

## Notes on the software

- The SDK's `Hatchet()` needs a token to build a client, even offline. The tests build one from a made-up token shaped like Hatchet's.
- `hatchet-lite` seeds one tenant, `707d0855-80ab-4e1f-a156-f1c4546cbf52`. The compose file's healthcheck uses its `/api/ready`.
- The compose worker sets `HATCHET_CLIENT_HOST_PORT` and `HATCHET_CLIENT_SERVER_URL` to the container names, because the token carries `localhost` addresses that only work from the host.
