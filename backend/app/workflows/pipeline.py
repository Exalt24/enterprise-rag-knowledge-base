"""
Archive ingestion, written as plain functions so it can run with or without Hatchet.

The Hatchet workflows in ingest.py are thin: each task calls one function from here. That keeps the rules (what is
refused, what counts as already indexed, which failures are retried) in one place that a test can drive with a stub
orchestrator and an in-memory index, and keeps the SDK out of the unit tests.

Flow for one archive:

    prepare_run   check the zip (archive_guard), unpack the fit files under their SHA-256, set aside the unfit ones,
                  skip a file whose bytes were already seen in this archive or are already indexed, and list the rest
    per file      parse_stage -> chunk_stage -> upsert_stage, using the repo's own DocumentParser, TextChunker and
                  VectorStoreService; each stage leaves a small file in the run's work folder for the next
    finish_run    fill in a row for any file nobody reported on, count the rows and clean the unpacked files up

Retries: parse_stage and chunk_stage turn every failure into BadFile, which the workflow never retries (the same file
fails the same way again). upsert_stage lets errors through as they are: those come from the embedding model or Qdrant
being unreachable, which a later attempt can fix.
"""
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

from langchain_core.documents import Document

from app.services.chunk_identity import annotate_chunks, point_ids
from app.workflows.archive_guard import ArchiveLimits, ArchiveRejected, plan_archive, unpack
from app.workflows.run_store import RunStore, row


class BadFile(Exception):
    """The file itself cannot be read. Never retried: the same bytes fail the same way."""


class Index(Protocol):
    def is_indexed(self, doc_id: str) -> bool: ...

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]: ...


@dataclass(frozen=True)
class FileJob:
    run_id: str
    file: str  # path inside the archive
    spool_path: str  # where the unpacked bytes are
    doc_id: str  # SHA-256 of the bytes


def limits_from_settings(settings, parser_cls) -> ArchiveLimits:
    return ArchiveLimits(
        max_files=settings.archive_max_files,
        max_total_bytes=settings.archive_max_total_mb * 1024 * 1024,
        max_file_bytes=settings.max_file_size_mb * 1024 * 1024,
        allowed_extensions=frozenset(parser_cls.SUPPORTED_EXTENSIONS),
    )


def run_dir(spool_dir, run_id: str) -> Path:
    return Path(spool_dir) / "runs" / run_id


# ---------------------------------------------------------------- the archive as a whole


def prepare_run(zip_path, run_id: str, spool_dir, limits: ArchiveLimits, index: Index, store: RunStore) -> List[FileJob]:
    """Check and unpack the archive and return the files that still need indexing. Raises ArchiveRejected."""
    store.update(run_id, state="validating", archive=Path(str(zip_path)).name, work_started_at=time.time())
    members, ignored = plan_archive(zip_path, limits)

    for member in members:
        if member.error_code is not None:
            store.add_row(run_id, row(member.path, "failed", f"{member.error_code}: {member.error_reason}"))

    first_seen: Dict[str, str] = {}
    jobs: List[FileJob] = []
    files_dir = run_dir(spool_dir, run_id) / "files"
    for member, path, outcome in unpack(zip_path, members, files_dir):
        if path is None:
            store.add_row(run_id, row(member.path, "failed", f"corrupt_entry: {outcome}"))
            continue
        doc_id = outcome
        if doc_id in first_seen:
            store.add_row(run_id, row(member.path, "skipped_duplicate", f"same content as {first_seen[doc_id]}", doc_id=doc_id))
            continue
        first_seen[doc_id] = member.path
        if index.is_indexed(doc_id):
            store.add_row(run_id, row(member.path, "skipped_duplicate", "already indexed (same content)", doc_id=doc_id))
            continue
        jobs.append(FileJob(run_id, member.path, str(path), doc_id))

    (run_dir(spool_dir, run_id) / "jobs.json").write_text(json.dumps([asdict(job) for job in jobs]), encoding="utf-8")
    store.update(run_id, state="running", entries=len(members), ignored=ignored, to_process=len(jobs))
    return jobs


def load_jobs(spool_dir, run_id: str) -> List[FileJob]:
    return [FileJob(**item) for item in json.loads((run_dir(spool_dir, run_id) / "jobs.json").read_text(encoding="utf-8"))]


def finish_run(run_id: str, jobs: Sequence[FileJob], outcomes: Sequence, spool_dir, store: RunStore) -> Dict:
    """Close the run: a job whose task reported nothing gets a failed row, then the counts are written.

    `seconds` runs from the start of prepare_run to here, so it covers checking and unpacking the archive as well as
    the file runs, and leaves out the time the run waited in the queue.
    """
    reported = {r["file"] for r in store.rows(run_id)}
    for job, outcome in zip(jobs, outcomes):
        if job.file not in reported:
            detail = f"{type(outcome).__name__}: {outcome}" if isinstance(outcome, BaseException) else "the task reported no result"
            store.add_row(run_id, row(job.file, "failed", f"task_failed: {detail}", doc_id=job.doc_id))
    shutil.rmtree(run_dir(spool_dir, run_id) / "files", ignore_errors=True)
    shutil.rmtree(run_dir(spool_dir, run_id) / "work", ignore_errors=True)
    began = (store.read_manifest(run_id) or {}).get("work_started_at", time.time())
    store.update(run_id, state="completed", seconds=round(time.time() - began, 2))
    return store.load(run_id)


def reject_run(run_id: str, error: ArchiveRejected, store: RunStore) -> None:
    store.update(run_id, state="rejected", error={"code": error.code, "reason": error.reason})


def fail_run(run_id: str, error: BaseException, store: RunStore) -> None:
    """An error nobody planned for ended the run: say so in the status, so it does not sit at "running" for ever."""
    store.update(run_id, state="failed", error={"code": "internal_error", "reason": f"{type(error).__name__}: {error}"[:500]})


# ---------------------------------------------------------------- one file


def _work_file(spool_dir, job: FileJob, kind: str) -> Path:
    return run_dir(spool_dir, job.run_id) / "work" / f"{job.doc_id}.{kind}.json"


def _dump(path: Path, docs: Sequence[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"page_content": d.page_content, "metadata": d.metadata} for d in docs]), encoding="utf-8")


def _load(path: Path) -> List[Document]:
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in json.loads(path.read_text(encoding="utf-8"))]


def _plain(job: FileJob, text: str) -> str:
    """The loader names the unpacked file in its messages; say the path inside the archive instead."""
    return text.replace(Path(job.spool_path).name, job.file)


def parse_stage(job: FileJob, spool_dir, parse: Callable[[str], List[Document]]) -> int:
    try:
        documents = parse(job.spool_path)
    except Exception as error:
        raise BadFile(_plain(job, str(error) or type(error).__name__)) from error
    documents = [d for d in documents if d.page_content and d.page_content.strip()]
    if not documents:
        raise BadFile(f"{job.file} holds no readable text")
    _dump(_work_file(spool_dir, job, "parsed"), documents)
    return len(documents)


def chunk_stage(job: FileJob, spool_dir, chunk: Callable[[List[Document]], List[Document]]) -> int:
    try:
        chunks = chunk(_load(_work_file(spool_dir, job, "parsed")))
    except Exception as error:
        raise BadFile(_plain(job, f"could not be split into chunks: {error}")) from error
    if not chunks:
        raise BadFile(f"{job.file} gave no chunks")
    for item in chunks:
        item.metadata["source"] = job.file
    annotate_chunks(chunks, job.doc_id, file_name=job.file, extra={"archive_path": job.file})
    _dump(_work_file(spool_dir, job, "chunks"), chunks)
    return len(chunks)


def upsert_stage(job: FileJob, spool_dir, index: Index) -> int:
    chunks = _load(_work_file(spool_dir, job, "chunks"))
    return len(index.add_documents(chunks, ids=point_ids(chunks)))


# Attempts the embed and upsert stage gets: the first try plus UPSERT_RETRIES retries, waiting 2, 4, 8 ... seconds up to
# BACKOFF_MAX_SECONDS between them. Only this stage retries: parse and chunk fail for reasons in the file.
UPSERT_RETRIES = 3
BACKOFF_FACTOR = 2.0
BACKOFF_MAX_SECONDS = 30


def record_success(job: FileJob, chunks: int, store: RunStore) -> None:
    store.add_row(job.run_id, row(job.file, "ingested", chunks=chunks, doc_id=job.doc_id))


def record_failure(job: FileJob, reason: str, store: RunStore) -> None:
    store.add_row(job.run_id, row(job.file, "failed", reason, doc_id=job.doc_id))


def run_parse(job: FileJob, spool_dir, store: RunStore, parse) -> int:
    """parse_stage that writes the failed row for a bad file before it raises BadFile."""
    try:
        return parse_stage(job, spool_dir, parse)
    except BadFile as error:
        record_failure(job, f"unreadable_file: {error}", store)
        raise


def run_chunk(job: FileJob, spool_dir, store: RunStore, chunk) -> int:
    """chunk_stage that writes the failed row for a bad file before it raises BadFile."""
    try:
        return chunk_stage(job, spool_dir, chunk)
    except BadFile as error:
        record_failure(job, f"unreadable_file: {error}", store)
        raise


def run_upsert(job: FileJob, spool_dir, store: RunStore, index: Index, attempt: int, max_attempts: int) -> int:
    """upsert_stage that records the outcome. An error is raised again so the orchestrator can retry it; the row for
    a failure is written only on the last attempt, so a file that succeeds on a retry never shows a failure."""
    try:
        stored = upsert_stage(job, spool_dir, index)
    except Exception as error:
        if attempt >= max_attempts:
            record_failure(job, f"gave_up_after_{attempt}_attempts: {type(error).__name__}: {error}", store)
        raise
    record_success(job, stored, store)
    return stored


def ingest_job(job: FileJob, spool_dir, store: RunStore, index: Index, parse, chunk) -> None:
    """All three stages in one call, for running without Hatchet. A bad file is recorded and not raised."""
    try:
        run_parse(job, spool_dir, store, parse)
        run_chunk(job, spool_dir, store, chunk)
    except BadFile:
        return
    run_upsert(job, spool_dir, store, index, attempt=1, max_attempts=1)


def run_archive(zip_path, run_id: str, spool_dir, limits: ArchiveLimits, index: Index, store: RunStore,
                spawn: Callable[[List[FileJob]], List]) -> Dict:
    """The whole archive with a pluggable orchestrator: `spawn(jobs)` must run every job and return one outcome each.

    Hatchet is one such orchestrator (ingest.py). The unit tests pass a stub that runs the jobs in this process.
    """
    store.start(run_id, Path(str(zip_path)).name, state="validating")
    try:
        jobs = prepare_run(zip_path, run_id, spool_dir, limits, index, store)
    except ArchiveRejected as error:
        reject_run(run_id, error, store)
        return store.load(run_id)
    outcomes = spawn(jobs) if jobs else []
    return finish_run(run_id, jobs, outcomes, spool_dir, store)
