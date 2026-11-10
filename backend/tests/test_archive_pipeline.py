"""Archive ingestion rules (app/workflows/pipeline.py) run with a stub orchestrator and the repo's own vector store.

The orchestrator is a few lines of Python that run every file job in this process, so the rules are driven exactly as
the Hatchet tasks drive them, with no Hatchet. The vector store is the repo's VectorStoreService pointed at an
in-memory Qdrant with the hashing embedder; the loaders and the chunker are the repo's own. The cases:
idempotency (a rerun does no new work), a rerun after a crash finishing only what was missing, one bad file not stopping
the others, retries only for errors that are not the file's fault, and a refused archive leaving nothing behind.
eval/mutation_check.py breaks these rules one at a time and requires a failure from this file.
Run: pytest tests/test_archive_pipeline.py
"""
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services.chunking import TextChunker  # noqa: E402
from app.services.chunk_identity import chunk_point_id  # noqa: E402
from app.services.document_parser import DocumentParser  # noqa: E402
from app.workflows import pipeline  # noqa: E402
from app.workflows.archive_guard import ArchiveLimits  # noqa: E402
from app.workflows.run_store import InvalidRunId, RunStore, checked_run_id  # noqa: E402
from eval.archive_fixture import make_docx, make_encrypted_pdf, make_pdf, make_text, make_xlsx, mixed_entries  # noqa: E402
from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import in_memory_store  # noqa: E402

LIMITS = ArchiveLimits(max_files=500, max_total_bytes=50_000_000, max_file_bytes=2_000_000, allowed_extensions=frozenset(DocumentParser.SUPPORTED_EXTENSIONS))
RUN_A, RUN_B, RUN_C = "11111111-aaaa-4aaa-8aaa-000000000001", "11111111-aaaa-4aaa-8aaa-000000000002", "11111111-aaaa-4aaa-8aaa-000000000003"


class FlakyEmbeddings(HashingEmbeddings):
    """The hashing embedder that can be told to fail on a chosen call, as an embedding service that went down would."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.fail_on_call = None

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise ConnectionError("embedding service is down")
        return super().embed_documents(texts)


class CountingIndex:
    """The real vector store, counting how many chunks were written through it."""

    def __init__(self, store):
        self.store = store
        self.written = 0

    def is_indexed(self, doc_id):
        return self.store.is_indexed(doc_id)

    def add_documents(self, documents, ids=None):
        self.written += len(documents)
        return self.store.add_documents(documents, ids=ids)


@pytest.fixture
def embedder():
    return FlakyEmbeddings()


@pytest.fixture
def index(embedder):
    with in_memory_store(embedder, []) as store:
        yield CountingIndex(store)


@pytest.fixture
def spool(tmp_path):
    return tmp_path / "spool"


def build_zip(tmp_path, entries, name="archive.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        for entry_name, data in entries:
            archive.writestr(entry_name, data)
    return path


def stub_orchestrator(spool, store, index, calls=None):
    """Runs every job in this process. Returns one outcome per job: None, or the exception the job raised."""

    def parse(path):
        if calls is not None:
            calls.append(Path(path).name)
        return DocumentParser.parse(path)

    def spawn(jobs):
        outcomes = []
        for job in jobs:
            try:
                pipeline.ingest_job(job, spool, store, index, parse, TextChunker.chunk_documents)
                outcomes.append(None)
            except Exception as error:  # what Hatchet hands back with return_exceptions=True
                outcomes.append(error)
        return outcomes

    return spawn


def run(zip_path, run_id, spool, index, calls=None, limits=LIMITS):
    store = RunStore(spool)
    return pipeline.run_archive(zip_path, run_id, spool, limits, index, store, stub_orchestrator(spool, store, index, calls))


def stored_points(index):
    points, _ = index.store._client.scroll(settings.qdrant_collection, limit=10_000, with_payload=True)
    return points


def statuses(result):
    return {item["file"]: item["status"] for item in result["files"]}


# ---------------------------------------------------------------- the happy path


def test_every_readable_file_is_indexed_with_its_identity_in_the_metadata(tmp_path, spool, index):
    zip_path = build_zip(tmp_path, [("a/one.txt", make_text(1).encode()), ("b/two.md", make_text(2).encode())])
    result = run(zip_path, RUN_A, spool, index)
    assert result["state"] == "completed"
    assert statuses(result) == {"a/one.txt": "ingested", "b/two.md": "ingested"}
    assert result["counts"] == {"ingested": 2, "skipped_duplicate": 0, "failed": 0}
    points = stored_points(index)
    assert len(points) == sum(item["chunks"] for item in result["files"])
    for point in points:
        meta = point.payload["metadata"]
        assert len(meta["doc_id"]) == 64 and set(meta["doc_id"]) <= set("0123456789abcdef")
        assert meta["source"] == meta["file_name"], "the unpack folder path must not leak into the stored metadata"
        assert meta["file_name"] == meta["archive_path"] and meta["file_name"] in statuses(result)
        assert str(point.id) == chunk_point_id(meta["doc_id"], meta["chunk_seq"])
        assert 0 <= meta["chunk_seq"] < meta["doc_chunks_total"]


def test_the_repos_own_loaders_read_every_format_in_a_mixed_archive(tmp_path, spool, index):
    entries = [("t.txt", make_text(1).encode()), ("p.pdf", make_pdf(2)), ("d.docx", make_docx(3)), ("x.xlsx", make_xlsx(4))]
    result = run(build_zip(tmp_path, entries), RUN_A, spool, index)
    assert statuses(result) == {name: "ingested" for name, _ in entries}
    types = {p.payload["metadata"]["file_type"] for p in stored_points(index) if "file_type" in p.payload["metadata"]}
    assert types == {"txt", "pdf", "docx", "xlsx"}


def test_the_work_and_unpack_folders_are_removed_when_the_run_ends(tmp_path, spool, index):
    run(build_zip(tmp_path, [("a.txt", make_text(1).encode())]), RUN_A, spool, index)
    run_folder = spool / "runs" / RUN_A
    assert (run_folder / "manifest.json").exists() and (run_folder / "rows").is_dir()
    assert not (run_folder / "files").exists() and not (run_folder / "work").exists()


# ---------------------------------------------------------------- idempotency


def test_running_the_same_archive_again_does_no_new_work(tmp_path, spool, index):
    zip_path = build_zip(tmp_path, [(f"f{i}.txt", make_text(i).encode()) for i in range(5)])
    first = run(zip_path, RUN_A, spool, index)
    written, calls = index.written, []
    second = run(zip_path, RUN_B, spool, index, calls)
    assert first["counts"]["ingested"] == 5
    assert second["counts"] == {"ingested": 0, "skipped_duplicate": 5, "failed": 0}
    assert index.written == written, "the rerun wrote chunks"
    assert calls == [], "the rerun parsed files"
    assert second["to_process"] == 0
    assert all("already indexed" in item["reason"] for item in second["files"])


def test_a_file_renamed_in_a_new_archive_is_still_recognised_by_its_content(tmp_path, spool, index):
    run(build_zip(tmp_path, [("old-name.txt", make_text(7).encode())], "one.zip"), RUN_A, spool, index)
    second = run(build_zip(tmp_path, [("renamed/new-name.txt", make_text(7).encode())], "two.zip"), RUN_B, spool, index)
    assert statuses(second) == {"renamed/new-name.txt": "skipped_duplicate"}


def test_a_changed_file_is_indexed_again(tmp_path, spool, index):
    run(build_zip(tmp_path, [("a.txt", make_text(1).encode())], "one.zip"), RUN_A, spool, index)
    second = run(build_zip(tmp_path, [("a.txt", make_text(1).encode() + b"one more line\n")], "two.zip"), RUN_B, spool, index)
    assert statuses(second) == {"a.txt": "ingested"}


def test_two_copies_inside_one_archive_are_indexed_once(tmp_path, spool, index):
    same = make_text(3).encode()
    result = run(build_zip(tmp_path, [("first.txt", same), ("copy/second.txt", same)]), RUN_A, spool, index)
    assert statuses(result) == {"first.txt": "ingested", "copy/second.txt": "skipped_duplicate"}
    assert "first.txt" in [i for i in result["files"] if i["file"] == "copy/second.txt"][0]["reason"]
    assert len(stored_points(index)) == [i for i in result["files"] if i["file"] == "first.txt"][0]["chunks"]


def test_writing_a_chunk_twice_replaces_it_instead_of_adding_a_copy(tmp_path, spool, index):
    job_zip = build_zip(tmp_path, [("a.txt", make_text(1).encode())])
    run(job_zip, RUN_A, spool, index)
    before = len(stored_points(index))
    # the same chunks written again under the same ids, as a retried task would
    points = stored_points(index)
    from langchain_core.documents import Document

    docs = [Document(page_content=p.payload["page_content"], metadata=p.payload["metadata"]) for p in points]
    index.store.add_documents(docs, ids=[str(p.id) for p in points])
    assert len(stored_points(index)) == before


# ---------------------------------------------------------------- a rerun after a crash finishes only what was missing


def test_a_rerun_after_a_crash_finishes_only_the_file_that_was_cut_short(tmp_path, spool, index, embedder):
    big = make_text(11, paragraphs=80).encode()  # many chunks, so it is written in several batches
    small = [(f"s{i}.txt", make_text(20 + i).encode()) for i in range(3)]
    zip_path = build_zip(tmp_path, small + [("big.txt", big)])
    # the small files embed first (one call each), then the big one in two batches: the fifth call is its second batch
    embedder.fail_on_call = embedder.calls + 5
    first = run(zip_path, RUN_A, spool, index, calls=[])
    assert statuses(first)["big.txt"] == "failed"
    failed = [i for i in first["files"] if i["file"] == "big.txt"][0]
    assert "ConnectionError" in failed["reason"]
    partial = [p for p in stored_points(index) if p.payload["metadata"]["file_name"] == "big.txt"]
    assert 0 < len(partial) < partial[0].payload["metadata"]["doc_chunks_total"], "the crash should have left part of the file stored"

    embedder.fail_on_call = None
    calls, written_before = [], index.written
    second = run(zip_path, RUN_B, spool, index, calls)
    statuses_second = statuses(second)
    assert statuses_second["big.txt"] == "ingested"
    assert all(statuses_second[name] == "skipped_duplicate" for name, _ in small)
    assert len(calls) == 1, "only the unfinished file was parsed"
    full = [p for p in stored_points(index) if p.payload["metadata"]["file_name"] == "big.txt"]
    total = full[0].payload["metadata"]["doc_chunks_total"]
    assert len(full) == total, "every chunk is stored exactly once"
    assert sorted(p.payload["metadata"]["chunk_seq"] for p in full) == list(range(total))
    assert index.written - written_before == total, "the rerun wrote the whole unfinished file again, and nothing else"


def test_a_file_counts_as_indexed_only_when_all_its_chunks_are_stored(tmp_path, spool, index, embedder):
    zip_path = build_zip(tmp_path, [("big.txt", make_text(11, paragraphs=80).encode())])
    embedder.fail_on_call = embedder.calls + 2  # the second of its two batches
    run(zip_path, RUN_A, spool, index)
    doc_ids = {p.payload["metadata"]["doc_id"] for p in stored_points(index)}
    assert len(doc_ids) == 1
    found, total = index.store.indexed_chunk_info(next(iter(doc_ids)))
    assert 0 < found < total
    assert index.store.is_indexed(next(iter(doc_ids))) is False


# ---------------------------------------------------------------- one bad file does not stop the others


def bad_and_good_entries():
    return [
        ("good/one.txt", make_text(1).encode()),
        ("bad/corrupt.pdf", b"%PDF-1.4\nthis was cut short"),
        ("bad/locked.pdf", make_encrypted_pdf(2)),
        ("bad/empty.txt", b"   \n  \n"),
        ("bad/not-a-workbook.xlsx", b"just text"),
        ("good/two.md", make_text(2).encode()),
    ]


def test_corrupt_locked_and_empty_files_are_recorded_with_a_reason_and_the_rest_are_indexed(tmp_path, spool, index):
    result = run(build_zip(tmp_path, bad_and_good_entries()), RUN_A, spool, index)
    assert result["state"] == "completed"
    assert statuses(result) == {
        "good/one.txt": "ingested", "good/two.md": "ingested",
        "bad/corrupt.pdf": "failed", "bad/locked.pdf": "failed", "bad/empty.txt": "failed", "bad/not-a-workbook.xlsx": "failed",
    }
    reasons = {i["file"]: i["reason"] for i in result["files"]}
    assert all(reasons[name].startswith("unreadable_file:") for name in reasons if name.startswith("bad/"))
    assert "bad/corrupt.pdf" in reasons["bad/corrupt.pdf"], "a reason names the file as the archive calls it"
    assert "empty" in reasons["bad/empty.txt"]
    assert result["counts"] == {"ingested": 2, "skipped_duplicate": 0, "failed": 4}


def test_a_bad_file_is_parsed_once_and_never_retried(tmp_path, spool, index):
    calls = []
    run(build_zip(tmp_path, [("bad/corrupt.pdf", b"%PDF-1.4\ncut short")]), RUN_A, spool, index, calls)
    assert len(calls) == 1


def test_a_bad_file_is_not_stored_even_in_part(tmp_path, spool, index):
    run(build_zip(tmp_path, bad_and_good_entries()), RUN_A, spool, index)
    assert not [p for p in stored_points(index) if p.payload["metadata"]["file_name"].startswith("bad/")]


def test_a_bad_file_stays_failed_on_every_rerun_without_hiding_the_good_ones(tmp_path, spool, index):
    zip_path = build_zip(tmp_path, bad_and_good_entries())
    run(zip_path, RUN_A, spool, index)
    second = run(zip_path, RUN_B, spool, index)
    assert second["counts"] == {"ingested": 0, "skipped_duplicate": 2, "failed": 4}


def test_entries_set_aside_by_the_guard_are_rows_too(tmp_path, spool, index):
    entries = [("ok.txt", make_text(1).encode()), ("pic.png", b"png"), ("inner.zip", b"PK\x03\x04"), ("huge.txt", b"x" * 2_000_001)]
    result = run(build_zip(tmp_path, entries), RUN_A, spool, index)
    reasons = {i["file"]: i["reason"] for i in result["files"]}
    assert reasons["pic.png"].startswith("extension_not_allowed")
    assert reasons["inner.zip"].startswith("nested_archive")
    assert reasons["huge.txt"].startswith("file_too_large")
    assert result["counts"] == {"ingested": 1, "skipped_duplicate": 0, "failed": 3}


def test_a_task_that_reported_nothing_gets_a_failed_row_with_the_error(tmp_path, spool, index):
    store = RunStore(spool)
    zip_path = build_zip(tmp_path, [("a.txt", make_text(1).encode())])
    result = pipeline.run_archive(zip_path, RUN_A, spool, LIMITS, index, store, lambda jobs: [RuntimeError("worker lost") for _ in jobs])
    [item] = result["files"]
    assert item["status"] == "failed" and "RuntimeError: worker lost" in item["reason"]


# ---------------------------------------------------------------- retries are for errors that are not the file's fault


def make_job(tmp_path, spool, index, text=None):
    """A job whose parse and chunk stages are done, so only the upsert is left."""
    path = tmp_path / "x.txt"
    path.write_bytes((text or make_text(1)).encode())
    from app.services.chunk_identity import compute_doc_id

    job = pipeline.FileJob(RUN_C, "x.txt", str(path), compute_doc_id(path))
    store = RunStore(spool)
    pipeline.run_parse(job, spool, store, DocumentParser.parse)
    pipeline.run_chunk(job, spool, store, TextChunker.chunk_documents)
    return job, store


def test_an_error_before_the_last_attempt_is_raised_for_a_retry_and_leaves_no_failed_row(tmp_path, spool, index, embedder):
    job, store = make_job(tmp_path, spool, index)
    embedder.fail_on_call = embedder.calls + 1
    with pytest.raises(ConnectionError):
        pipeline.run_upsert(job, spool, store, index, attempt=1, max_attempts=4)
    assert store.rows(RUN_C) == []


def test_the_last_attempt_records_the_failure_and_still_raises(tmp_path, spool, index, embedder):
    job, store = make_job(tmp_path, spool, index)
    embedder.fail_on_call = embedder.calls + 1
    with pytest.raises(ConnectionError):
        pipeline.run_upsert(job, spool, store, index, attempt=4, max_attempts=4)
    [item] = store.rows(RUN_C)
    assert item["status"] == "failed" and item["reason"].startswith("gave_up_after_4_attempts: ConnectionError")


def test_a_retry_that_succeeds_leaves_one_ingested_row(tmp_path, spool, index, embedder):
    job, store = make_job(tmp_path, spool, index)
    embedder.fail_on_call = embedder.calls + 1
    with pytest.raises(ConnectionError):
        pipeline.run_upsert(job, spool, store, index, attempt=1, max_attempts=4)
    embedder.fail_on_call = None
    stored = pipeline.run_upsert(job, spool, store, index, attempt=2, max_attempts=4)
    [item] = store.rows(RUN_C)
    assert item["status"] == "ingested" and item["chunks"] == stored > 0


def test_parse_and_chunk_stages_turn_every_failure_of_the_file_into_bad_file(tmp_path, spool):
    path = tmp_path / "unpacked-4f2a.txt"
    path.write_bytes(b"text")
    job = pipeline.FileJob(RUN_C, "dir/x.txt", str(path), "0" * 64)

    def explode(_):
        raise OSError(f"cannot read {path.name}")

    with pytest.raises(pipeline.BadFile) as caught:
        pipeline.parse_stage(job, spool, explode)
    assert "dir/x.txt" in str(caught.value) and path.name not in str(caught.value)


# ---------------------------------------------------------------- a refused archive leaves nothing behind


def test_a_zip_slip_archive_is_refused_whole_and_nothing_is_indexed_or_unpacked(tmp_path, spool, index):
    zip_path = build_zip(tmp_path, [("fine.txt", make_text(1).encode()), ("../../escape.txt", b"bad")])
    result = run(zip_path, RUN_A, spool, index)
    assert result["state"] == "rejected" and result["error"]["code"] == "path_traversal"
    assert result["files"] == [] and index.written == 0
    assert not (spool / "runs" / RUN_A / "files").exists()
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()


@pytest.mark.parametrize(
    "limits,code",
    [
        (ArchiveLimits(2, 10**9, 10**9, LIMITS.allowed_extensions), "too_many_files"),
        (ArchiveLimits(100, 100, 10**9, LIMITS.allowed_extensions), "archive_too_large"),
    ],
)
def test_the_caps_come_from_the_limits_the_run_is_given(tmp_path, spool, index, limits, code):
    zip_path = build_zip(tmp_path, [(f"f{i}.txt", make_text(i).encode()) for i in range(3)])
    result = run(zip_path, RUN_A, spool, index, limits=limits)
    assert result["state"] == "rejected" and result["error"]["code"] == code and index.written == 0


def test_a_text_file_renamed_to_zip_is_refused(tmp_path, spool, index):
    path = tmp_path / "x.zip"
    path.write_bytes(b"not a zip")
    result = pipeline.run_archive(path, RUN_A, spool, LIMITS, index, RunStore(spool), lambda jobs: [])
    assert result["state"] == "rejected" and result["error"]["code"] == "not_a_zip"


# ---------------------------------------------------------------- the status files


@pytest.mark.parametrize("bad", ["", "../x", "..", "a/b", "a\\b", "short", "x" * 65, "12345678/../../etc", "run id with spaces"])
def test_a_run_id_that_could_name_another_folder_is_refused(bad):
    with pytest.raises(InvalidRunId):
        checked_run_id(bad)


def test_the_run_ids_hatchet_issues_are_accepted():
    assert checked_run_id("0d5a4b9e-6a33-4c0f-8e55-3b1f2c7a9d10") == "0d5a4b9e-6a33-4c0f-8e55-3b1f2c7a9d10"


def test_an_unknown_run_reads_as_none_and_a_queued_run_is_not_overwritten_by_a_later_submit(tmp_path):
    store = RunStore(tmp_path)
    assert store.load(RUN_A) is None
    store.update(RUN_A, state="running", archive="a.zip")
    store.start(RUN_A, "a.zip", state="queued", keep_existing=True)
    assert store.load(RUN_A)["state"] == "running"


def test_the_status_lists_each_path_once_even_when_two_paths_share_content(tmp_path):
    store = RunStore(tmp_path)
    store.add_row(RUN_A, pipeline.row("a.txt", "ingested", chunks=2, doc_id="d" * 64))
    store.add_row(RUN_A, pipeline.row("b.txt", "skipped_duplicate", "same content as a.txt", doc_id="d" * 64))
    store.add_row(RUN_A, pipeline.row("a.txt", "ingested", chunks=2, doc_id="d" * 64))  # a retried task rewrites its row
    store.start(RUN_A, "a.zip")
    status = store.load(RUN_A)
    assert [i["file"] for i in status["files"]] == ["a.txt", "b.txt"]
    assert status["counts"] == {"ingested": 1, "skipped_duplicate": 1, "failed": 0}


def test_the_mixed_fixture_archive_comes_out_as_documented(tmp_path, spool, index):
    entries = mixed_entries(40)
    result = run(build_zip(tmp_path, entries), RUN_A, spool, index)
    by_file = statuses(result)
    assert by_file["damaged/corrupt-report.pdf"] == "failed" and by_file["damaged/locked-report.pdf"] == "failed"
    assert by_file["damaged/not-a-workbook.xlsx"] == "failed" and by_file["damaged/nested.zip"] == "failed"
    assert by_file["images/site-photo.png"] == "failed"
    assert by_file["copies/doc-0000-copy.txt"] == "skipped_duplicate" and by_file["copies/doc-0005-copy.txt"] == "skipped_duplicate"
    assert result["counts"]["failed"] == 5 and result["counts"]["skipped_duplicate"] == 2
    assert result["counts"]["ingested"] == len(entries) - 7


# ---------------------------------------------------------------- more cases the guards need


def test_an_entry_that_is_damaged_inside_the_zip_is_a_failed_row_and_the_others_still_go_through(tmp_path, spool, index):
    path = build_zip(tmp_path, [("bad.txt", b"hello hello hello hello"), ("good.txt", make_text(1).encode())])
    data = bytearray(path.read_bytes())
    start = data.index(b"hello hello hello hello")
    data[start] ^= 0xFF  # a stored byte changes, so the CRC no longer matches
    path.write_bytes(bytes(data))
    result = run(path, RUN_A, spool, index)
    assert statuses(result) == {"bad.txt": "failed", "good.txt": "ingested"}
    assert [i for i in result["files"] if i["file"] == "bad.txt"][0]["reason"].startswith("corrupt_entry:")


def test_a_file_the_loader_reads_as_nothing_is_a_bad_file(tmp_path, spool):
    path = tmp_path / "unpacked-77aa.txt"
    path.write_bytes(b"text")
    job = pipeline.FileJob(RUN_C, "scan.pdf", str(path), "0" * 64)
    with pytest.raises(pipeline.BadFile) as caught:
        pipeline.parse_stage(job, spool, lambda _: [])  # a scanned PDF has no text layer, so the loader returns no pages
    assert "scan.pdf holds no readable text" in str(caught.value)


def test_a_file_that_gives_no_chunks_is_a_bad_file_and_nothing_is_stored_for_it(tmp_path, spool):
    path = tmp_path / "unpacked-88bb.txt"
    path.write_bytes(b"text")
    job = pipeline.FileJob(RUN_C, "odd.txt", str(path), "0" * 64)
    pipeline.parse_stage(job, spool, DocumentParser.parse)
    with pytest.raises(pipeline.BadFile) as caught:
        pipeline.chunk_stage(job, spool, lambda docs: [])
    assert "odd.txt gave no chunks" in str(caught.value)


def test_a_chunker_that_raises_is_reported_as_a_bad_file(tmp_path, spool):
    path = tmp_path / "unpacked-99cc.txt"
    path.write_bytes(b"text")
    job = pipeline.FileJob(RUN_C, "odd.txt", str(path), "0" * 64)
    pipeline.parse_stage(job, spool, DocumentParser.parse)

    def explode(_):
        raise RuntimeError("splitter broke")

    with pytest.raises(pipeline.BadFile) as caught:
        pipeline.chunk_stage(job, spool, explode)
    assert "splitter broke" in str(caught.value)


def test_the_limits_are_read_from_the_settings_in_megabytes(monkeypatch):
    monkeypatch.setattr(settings, "archive_max_files", 7)
    monkeypatch.setattr(settings, "archive_max_total_mb", 3)
    monkeypatch.setattr(settings, "max_file_size_mb", 2)
    limits = pipeline.limits_from_settings(settings, DocumentParser)
    assert (limits.max_files, limits.max_total_bytes, limits.max_file_bytes) == (7, 3 * 1024 * 1024, 2 * 1024 * 1024)
    assert limits.allowed_extensions == frozenset(DocumentParser.SUPPORTED_EXTENSIONS)


def test_ids_that_do_not_match_the_documents_are_refused(index):
    from langchain_core.documents import Document

    with pytest.raises(ValueError, match="1 ids for 2 documents"):
        index.store.add_documents([Document(page_content="a", metadata={}), Document(page_content="b", metadata={})], ids=["only-one"])


def test_a_row_can_only_carry_one_of_the_three_status_words():
    assert pipeline.row("a.txt", "ingested", chunks=1)["status"] == "ingested"
    with pytest.raises(ValueError):
        pipeline.row("a.txt", "weird")


@pytest.mark.skipif(os.name == "nt", reason="Windows refuses a rename onto a file another writer is replacing; the worker runs on POSIX")
def test_two_processes_writing_the_same_manifest_at_once_never_break_each_other(tmp_path):
    import threading

    store = RunStore(tmp_path)
    errors = []

    def writer(state):
        try:
            for _ in range(40):
                store.update(RUN_A, state=state)
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=writer, args=(name,)) for name in ("queued", "running")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert store.load(RUN_A)["state"] in ("queued", "running")
    assert not list((tmp_path / "runs" / RUN_A).glob("*.tmp")), "no temporary file is left behind"


def test_a_write_that_finds_the_folder_missing_is_tried_again(tmp_path, monkeypatch):
    real = Path.write_text
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise FileNotFoundError("the shared folder has not caught up")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", flaky)
    store = RunStore(tmp_path)
    store.start(RUN_B, "a.zip")
    assert store.load(RUN_B)["archive"] == "a.zip" and calls["n"] == 2
