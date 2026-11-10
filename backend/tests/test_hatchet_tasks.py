"""The Hatchet task definitions in app/workflows/ingest.py, run through the SDK's own mock_run (no engine, no Qdrant).

mock_run calls a task's function with a stand-in context, which is enough to prove how each task maps a bad file or a
transient error onto Hatchet's retry rules, with the real SDK exception classes. It also pins the retry settings. The
full flow against a real hatchet-lite container is tests/test_hatchet_integration.py.
Needs hatchet-sdk (pip install -r requirements-hatchet.txt); skipped without it.
Run: pytest tests/test_hatchet_tasks.py
"""
import base64
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("hatchet_sdk")


def _fake_token() -> str:
    """A token shaped like Hatchet's (three dot separated parts, the middle one JSON), enough to build a client offline."""
    def part(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    claims = {"grpc_broadcast_address": "localhost:7077", "server_url": "http://localhost:8888", "sub": "00000000-0000-4000-8000-000000000000"}
    return ".".join([part({"alg": "none"}), part(claims), "signature"])


_saved = {key: os.environ.get(key) for key in ("HATCHET_CLIENT_TOKEN", "HATCHET_CLIENT_TLS_STRATEGY")}
os.environ["HATCHET_CLIENT_TOKEN"] = _fake_token()
os.environ["HATCHET_CLIENT_TLS_STRATEGY"] = "none"
from hatchet_sdk import NonRetryableException  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.workflows import ingest, pipeline  # noqa: E402
from app.workflows.run_store import RunStore  # noqa: E402
from eval.archive_fixture import make_text  # noqa: E402
from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import in_memory_store  # noqa: E402

for _key, _value in _saved.items():  # the client is built; put the environment back the way the other tests expect it
    if _value is None:
        os.environ.pop(_key, None)
    else:
        os.environ[_key] = _value

RUN = "33333333-cccc-4ccc-8ccc-000000000001"


@pytest.fixture
def spool(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ingest_spool_dir", str(tmp_path / "spool"))
    return tmp_path / "spool"


def file_input(tmp_path, name, data: bytes) -> "ingest.FileInput":
    from app.services.chunk_identity import compute_doc_id

    path = tmp_path / f"unpacked-{abs(hash(name))}{Path(name).suffix}"
    path.write_bytes(data)
    return ingest.FileInput(run_id=RUN, file=name, spool_path=str(path), doc_id=compute_doc_id(path))


# ---------------------------------------------------------------- the retry settings


def test_parse_and_chunk_are_never_retried_and_only_the_upsert_is():
    assert ingest.parse.retries == 0 and ingest.chunk.retries == 0
    assert ingest.embed_and_upsert.retries == pipeline.UPSERT_RETRIES == 3
    assert ingest.embed_and_upsert.backoff_factor == pipeline.BACKOFF_FACTOR == 2.0
    assert ingest.embed_and_upsert.backoff_max_seconds == pipeline.BACKOFF_MAX_SECONDS == 30


def test_the_archive_level_tasks_are_not_retried_either():
    assert ingest.validate.retries == 0 and ingest.fan_out.retries == 0


def test_the_file_workflow_is_a_chain_and_the_archive_workflow_validates_before_it_fans_out():
    assert [t.name for t in ingest.ingest_file.tasks] == ["parse", "chunk", "embed_and_upsert"]
    assert [p.name for p in ingest.chunk.parents] == ["parse"]
    assert [p.name for p in ingest.embed_and_upsert.parents] == ["chunk"]
    assert [t.name for t in ingest.ingest_archive.tasks] == ["validate", "fan_out"]
    assert [p.name for p in ingest.fan_out.parents] == ["validate"]


# ---------------------------------------------------------------- a bad file is not retried


def test_parse_turns_a_corrupt_file_into_the_exception_hatchet_never_retries(tmp_path, spool):
    item = file_input(tmp_path, "bad/corrupt.pdf", b"%PDF-1.4\ncut short")
    with pytest.raises(NonRetryableException) as caught:
        ingest.parse.mock_run(item)
    assert "bad/corrupt.pdf" in str(caught.value)
    [failed] = RunStore(spool).rows(RUN)
    assert failed["file"] == "bad/corrupt.pdf" and failed["status"] == "failed" and failed["reason"].startswith("unreadable_file:")


def test_parse_then_chunk_then_upsert_ingests_a_good_file(tmp_path, spool):
    item = file_input(tmp_path, "good/note.txt", make_text(1).encode())
    with in_memory_store(HashingEmbeddings(), []) as store:
        assert ingest.parse.mock_run(item)["documents"] == 1
        chunks = ingest.chunk.mock_run(item, parent_outputs={"parse": {"documents": 1}})["chunks"]
        stored = ingest.embed_and_upsert.mock_run(item, parent_outputs={"chunk": {"chunks": chunks}})["chunks_stored"]
        assert stored == chunks == store._client.count("enterprise_rag").count
    [done] = RunStore(spool).rows(RUN)
    assert done["status"] == "ingested" and done["chunks"] == chunks


# ---------------------------------------------------------------- a transient error is retried


class DownEmbeddings(HashingEmbeddings):
    """Works while the store is built (the Qdrant wrapper probes it once), then goes down."""

    down = False

    def embed_documents(self, texts):
        if self.down:
            raise ConnectionError("embedding service is down")
        return super().embed_documents(texts)


def staged_file(tmp_path, spool):
    item = file_input(tmp_path, "good/note.txt", make_text(1).encode())
    ingest.parse.mock_run(item)
    ingest.chunk.mock_run(item, parent_outputs={"parse": {"documents": 1}})
    return item


def test_an_outage_on_an_early_attempt_is_raised_so_hatchet_retries_it_and_no_failure_is_recorded(tmp_path, spool):
    item = staged_file(tmp_path, spool)
    embedder = DownEmbeddings()
    with in_memory_store(embedder, []):
        embedder.down = True
        with pytest.raises(ConnectionError):
            ingest.embed_and_upsert.mock_run(item, parent_outputs={"chunk": {}}, retry_count=0)
    assert RunStore(spool).rows(RUN) == []


def test_an_outage_on_the_last_attempt_is_recorded_as_a_failed_row(tmp_path, spool):
    item = staged_file(tmp_path, spool)
    embedder = DownEmbeddings()
    with in_memory_store(embedder, []):
        embedder.down = True
        with pytest.raises(ConnectionError):
            ingest.embed_and_upsert.mock_run(item, parent_outputs={"chunk": {}}, retry_count=pipeline.UPSERT_RETRIES)
    [failed] = RunStore(spool).rows(RUN)
    assert failed["status"] == "failed" and "ConnectionError" in failed["reason"] and "4_attempts" in failed["reason"]


def test_the_last_attempt_is_counted_from_the_retries_the_task_is_allowed(tmp_path, spool):
    item = staged_file(tmp_path, spool)
    embedder = DownEmbeddings()
    with in_memory_store(embedder, []):
        embedder.down = True
        with pytest.raises(ConnectionError):  # attempt 3 of 4 (retry_count 2): Hatchet will still retry, so no failed row yet
            ingest.embed_and_upsert.mock_run(item, parent_outputs={"chunk": {}}, retry_count=pipeline.UPSERT_RETRIES - 1)
    assert RunStore(spool).rows(RUN) == []


# ---------------------------------------------------------------- the archive level task


def test_validate_refuses_a_zip_slip_archive_with_the_exception_hatchet_never_retries(tmp_path, spool):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(NonRetryableException) as caught:
        with in_memory_store(HashingEmbeddings(), []):
            ingest.validate.mock_run(ingest.ArchiveInput(zip_path=str(zip_path)))
    assert "path_traversal" in str(caught.value)
    runs = list((spool / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "rejected" and manifest["error"]["code"] == "path_traversal"


def test_an_error_nobody_planned_for_ends_the_run_as_failed_and_is_still_raised(tmp_path, spool, monkeypatch):
    zip_path = tmp_path / "ok.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("a.txt", "hello")

    def broken(*args, **kwargs):
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(pipeline, "prepare_run", broken)
    with pytest.raises(RuntimeError):
        with in_memory_store(HashingEmbeddings(), []):
            ingest.validate.mock_run(ingest.ArchiveInput(zip_path=str(zip_path)))
    [folder] = list((spool / "runs").iterdir())
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "failed" and manifest["error"]["reason"] == "RuntimeError: the disk is full"
