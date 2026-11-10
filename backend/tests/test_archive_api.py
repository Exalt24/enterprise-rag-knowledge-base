"""The two archive routes: POST /api/ingest/archive and GET /api/ingest/{run_id}.

The real FastAPI app through its test client, with the Hatchet call replaced by a stub that records what it was given,
so no Hatchet and no Qdrant are needed. The status route reads the same run files the worker writes (run_store.py),
which the tests seed through RunStore.
Run: pytest tests/test_archive_api.py
"""
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402
from app.workflows import trigger  # noqa: E402
from app.workflows.run_store import RunStore, row  # noqa: E402

RUN = "22222222-bbbb-4bbb-8bbb-000000000001"


@pytest.fixture
def spool(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ingest_spool_dir", str(tmp_path / "spool"))
    return tmp_path / "spool"


@pytest.fixture
def client():
    return TestClient(app)


def small_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("a.txt", "hello")
    return out.getvalue()


# ---------------------------------------------------------------- GET /api/ingest/{run_id}


def test_the_status_route_returns_the_result_table(spool, client):
    store = RunStore(spool)
    store.start(RUN, "room.zip", state="running")
    store.add_row(RUN, row("a.txt", "ingested", chunks=3, doc_id="a" * 64))
    store.add_row(RUN, row("b.txt", "skipped_duplicate", "already indexed (same content)", doc_id="b" * 64))
    store.add_row(RUN, row("c.pdf", "failed", "unreadable_file: PDF corrupted or encrypted"))
    store.update(RUN, state="completed", seconds=1.5, entries=3, ignored=0, to_process=2)
    body = client.get(f"/api/ingest/{RUN}").json()
    assert body["state"] == "completed" and body["archive"] == "room.zip" and body["seconds"] == 1.5
    assert body["counts"] == {"ingested": 1, "skipped_duplicate": 1, "failed": 1}
    assert [(f["file"], f["status"]) for f in body["files"]] == [("a.txt", "ingested"), ("b.txt", "skipped_duplicate"), ("c.pdf", "failed")]
    assert body["files"][2]["reason"].startswith("unreadable_file")


def test_a_refused_archive_shows_the_code_and_no_rows(spool, client):
    store = RunStore(spool)
    store.start(RUN, "evil.zip")
    store.update(RUN, state="rejected", error={"code": "path_traversal", "reason": "entry name '../x' climbs out"})
    body = client.get(f"/api/ingest/{RUN}").json()
    assert body["state"] == "rejected" and body["error"]["code"] == "path_traversal" and body["files"] == []


def test_an_unknown_run_is_a_404(spool, client):
    assert client.get(f"/api/ingest/{RUN}").status_code == 404


@pytest.mark.parametrize("bad", ["..", "short", "a%2F..%2Fb", "x" * 80])
def test_a_run_id_that_is_not_one_is_a_404_and_never_reads_another_folder(spool, client, bad, tmp_path):
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "manifest.json").write_text('{"run_id": "x", "state": "completed"}', encoding="utf-8")
    assert client.get(f"/api/ingest/{bad}").status_code == 404


# ---------------------------------------------------------------- POST /api/ingest/archive


def test_without_a_hatchet_token_the_upload_says_why_it_cannot_start(spool, client, monkeypatch):
    monkeypatch.delenv("HATCHET_CLIENT_TOKEN", raising=False)
    response = client.post("/api/ingest/archive", files={"file": ("room.zip", small_zip(), "application/zip")})
    assert response.status_code == 503 and response.json()["detail"].startswith("Archive ingestion is not set up: HATCHET_CLIENT_TOKEN")
    assert not (spool / "uploads").exists(), "the zip must not be saved when nothing can start the workflow"


def test_only_a_zip_name_is_accepted(spool, client, monkeypatch):
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "x")
    assert client.post("/api/ingest/archive", files={"file": ("room.rar", b"x", "application/octet-stream")}).status_code == 400


def test_an_upload_is_saved_in_the_spool_and_handed_to_the_workflow(spool, client, monkeypatch):
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "x")
    seen = {}

    def fake_submit(zip_path, store):
        seen["path"], seen["bytes"] = zip_path, Path(zip_path).read_bytes()
        return RUN

    monkeypatch.setattr(trigger, "submit_archive", fake_submit)
    response = client.post("/api/ingest/archive", files={"file": ("room.zip", small_zip(), "application/zip")})
    assert response.status_code == 202
    assert response.json() == {"run_id": RUN, "status_url": f"/api/ingest/{RUN}"}
    assert Path(seen["path"]).parent == (spool / "uploads").resolve() and seen["bytes"] == small_zip()


def test_an_upload_over_the_size_cap_is_refused_and_not_kept(spool, client, monkeypatch):
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "x")
    monkeypatch.setattr(settings, "archive_max_upload_mb", 0)
    monkeypatch.setattr(trigger, "submit_archive", lambda *a: pytest.fail("the workflow must not start"))
    response = client.post("/api/ingest/archive", files={"file": ("room.zip", small_zip(), "application/zip")})
    assert response.status_code == 413
    assert list((spool / "uploads").iterdir()) == []


def test_a_workflow_that_cannot_start_leaves_no_upload_behind(spool, client, monkeypatch):
    monkeypatch.setenv("HATCHET_CLIENT_TOKEN", "x")

    def broken(*args):
        raise ConnectionError("hatchet is down")

    monkeypatch.setattr(trigger, "submit_archive", broken)
    response = client.post("/api/ingest/archive", files={"file": ("room.zip", small_zip(), "application/zip")})
    assert response.status_code == 503 and "hatchet is down" in response.json()["detail"]
    assert list((spool / "uploads").iterdir()) == []
