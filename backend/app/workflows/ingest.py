"""
The Hatchet workflows for archive ingestion. The rules live in pipeline.py; the tasks here only call it.

    ingest_archive   validate     check the zip and unpack it (archive_guard, pipeline.prepare_run)
                     fan_out      start one ingest_file run per file still to index, wait for all of them, write the
                                  result table (pipeline.finish_run)
    ingest_file      parse -> chunk -> embed_and_upsert    one run per file

Run a worker for both with `python -m app.workflows worker` (see docs/ingestion.md). Importing this module needs
HATCHET_CLIENT_TOKEN in the environment because the SDK builds its client at once.
"""
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

from hatchet_sdk import Context, NonRetryableException
from pydantic import BaseModel

from app.core.config import settings
from app.services.chunking import TextChunker
from app.services.document_parser import DocumentParser
from app.services.vector_store import vector_store
from app.workflows import pipeline
from app.workflows.archive_guard import ArchiveRejected
from app.workflows.hatchet_client import get_hatchet
from app.workflows.run_store import RunStore
from app.workflows.testing_hooks import wrap_index

hatchet = get_hatchet()

SPAWN_BATCH = 100  # file runs started and awaited together; the default worker has 100 slots


class ArchiveInput(BaseModel):
    zip_path: str


class FileInput(BaseModel):
    run_id: str
    file: str
    spool_path: str
    doc_id: str


def spool_dir() -> Path:
    return Path(settings.ingest_spool_dir).resolve()


def run_store() -> RunStore:
    return RunStore(spool_dir())


def _job(input: FileInput) -> pipeline.FileJob:
    return pipeline.FileJob(input.run_id, input.file, input.spool_path, input.doc_id)


ingest_archive = hatchet.workflow(name="ingest_archive", input_validator=ArchiveInput)
ingest_file = hatchet.workflow(name="ingest_file", input_validator=FileInput)


# ---------------------------------------------------------------- the archive


@ingest_archive.task(name="validate", execution_timeout=timedelta(minutes=30), retries=0)
def validate(input: ArchiveInput, ctx: Context) -> Dict:
    run_id = ctx.workflow_run_id
    store = run_store()
    store.start(run_id, Path(input.zip_path).name, state="validating", keep_existing=True)
    limits = pipeline.limits_from_settings(settings, DocumentParser)
    try:
        jobs = pipeline.prepare_run(input.zip_path, run_id, spool_dir(), limits, wrap_index(vector_store), store)
    except ArchiveRejected as error:
        pipeline.reject_run(run_id, error, store)
        # a refused archive is the archive's fault, so asking again would only be refused again
        raise NonRetryableException(f"{error.code}: {error.reason}") from error
    except Exception as error:
        pipeline.fail_run(run_id, error, store)
        raise
    return {"run_id": run_id, "to_process": len(jobs)}


@ingest_archive.task(name="fan_out", parents=[validate], execution_timeout=timedelta(hours=2), retries=0)
async def fan_out(input: ArchiveInput, ctx: Context) -> Dict:
    run_id = ctx.workflow_run_id
    store = run_store()
    try:
        jobs = pipeline.load_jobs(spool_dir(), run_id)
        outcomes: List = []
        for start in range(0, len(jobs), SPAWN_BATCH):
            batch = jobs[start:start + SPAWN_BATCH]
            outcomes.extend(
                await ingest_file.aio_run_many(
                    [
                        ingest_file.create_bulk_run_item(
                            input=FileInput(run_id=run_id, file=j.file, spool_path=j.spool_path, doc_id=j.doc_id)
                        )
                        for j in batch
                    ],
                    return_exceptions=True,
                )
            )
        return pipeline.finish_run(run_id, jobs, outcomes, spool_dir(), store)
    except Exception as error:
        pipeline.fail_run(run_id, error, store)
        raise


# ---------------------------------------------------------------- one file


@ingest_file.task(name="parse", execution_timeout=timedelta(minutes=5), retries=0)
def parse(input: FileInput, ctx: Context) -> Dict:
    try:
        return {"documents": pipeline.run_parse(_job(input), spool_dir(), run_store(), DocumentParser.parse)}
    except pipeline.BadFile as error:
        raise NonRetryableException(str(error)) from error


@ingest_file.task(name="chunk", parents=[parse], execution_timeout=timedelta(minutes=5), retries=0)
def chunk(input: FileInput, ctx: Context) -> Dict:
    try:
        return {"chunks": pipeline.run_chunk(_job(input), spool_dir(), run_store(), TextChunker.chunk_documents)}
    except pipeline.BadFile as error:
        raise NonRetryableException(str(error)) from error


@ingest_file.task(
    name="embed_and_upsert",
    parents=[chunk],
    execution_timeout=timedelta(minutes=10),
    retries=pipeline.UPSERT_RETRIES,
    backoff_factor=pipeline.BACKOFF_FACTOR,
    backoff_max_seconds=pipeline.BACKOFF_MAX_SECONDS,
)
def embed_and_upsert(input: FileInput, ctx: Context) -> Dict:
    stored = pipeline.run_upsert(
        _job(input), spool_dir(), run_store(), wrap_index(vector_store),
        attempt=ctx.attempt_number, max_attempts=pipeline.UPSERT_RETRIES + 1,
    )
    return {"chunks_stored": stored}


def build_worker(slots: int = 100):
    """One worker that serves both workflows."""
    return hatchet.worker("rag-ingest-worker", slots=slots, workflows=[ingest_archive, ingest_file])
