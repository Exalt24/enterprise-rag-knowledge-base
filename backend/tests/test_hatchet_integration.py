"""Archive ingestion against a real hatchet-lite container, with a real worker, Qdrant and Postgres.

docker-compose.hatchet.yml starts the stack; this test creates a client token from the lite container, builds and starts
the worker container (the SDK's worker needs a POSIX system, so it runs in Docker on every platform), then drives the
workflows the way a user would: submit a zip, read the result table, submit it again, break things on purpose. The
embedder inside the worker is the hashing test double (RAG_INGEST_TEST_MODE=hashing) so nothing downloads a model.

Marked `hatchet` and left out of a plain `pytest` run (pytest.ini). CI runs it with `pytest -m hatchet`. It skips itself
when Docker is not available. It takes several minutes: the worker image is built, and two cases wait out retry backoffs.
Run: pytest -m hatchet tests/test_hatchet_integration.py
"""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("RENDER", "true")
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pytest  # noqa: E402

from eval.archive_fixture import build_archive, make_text  # noqa: E402

pytestmark = pytest.mark.hatchet

ROOT = BACKEND.parent
COMPOSE_FILE = ROOT / "docker-compose.hatchet.yml"
PROJECT = "rag-hatchet-test"
TENANT = "707d0855-80ab-4e1f-a156-f1c4546cbf52"  # the tenant hatchet-lite seeds on first start
FILES = 60  # a mixed archive: documents of every format and the bad entries listed in eval/archive_fixture.py
WAIT_SECONDS = 420


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


def compose(*args, env=None, check=True, timeout=900):
    command = ["docker", "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), *args]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    if check and done.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:6])} ... {' '.join(args)} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done


class Stack:
    def __init__(self, spool: Path, env: dict):
        self.spool, self.env = spool, env
        self.store = None
        self.qdrant = None

    def submit(self, name: str, wait: float = WAIT_SECONDS) -> dict:
        """Submit /spool/inbox/<name> (the zip as the worker sees it) and wait for the run to end."""
        from app.workflows.trigger import submit_archive, wait_for_run

        run_id = submit_archive(f"/spool/inbox/{name}", self.store)
        status = wait_for_run(self.store, run_id, wait)
        assert status is not None, f"run {run_id} did not finish in {wait} seconds"
        return status

    def points(self):
        found, offset = [], None
        while True:
            batch, offset = self.qdrant.scroll("enterprise_rag", limit=500, offset=offset, with_payload=True)
            found.extend(batch)
            if offset is None:
                return found

    def fault(self, count: int) -> None:
        (self.spool / "fault.txt").write_text(str(count), encoding="utf-8")

    def fault_left(self) -> int:
        return int((self.spool / "fault.txt").read_text(encoding="utf-8").strip())

    def restart_worker(self) -> None:
        compose("--profile", "worker", "up", "-d", "--force-recreate", "worker", env=self.env)
        self.wait_for_worker()

    def wait_for_worker(self, seconds: float = 90) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            logs = compose("logs", "worker", env=self.env, check=False).stdout
            if "started, waiting for tasks" in logs:
                return
            time.sleep(1)  # WAIT-OK: polls the worker log for its ready line
        raise RuntimeError("the worker did not report that it was ready")


@pytest.fixture(scope="module")
def stack(tmp_path_factory, request):
    if not docker_ready():
        pytest.skip("Docker is not available")
    spool = tmp_path_factory.mktemp("ingest-spool")
    (spool / "inbox").mkdir()
    env = {
        **os.environ,
        "INGEST_SPOOL_HOST": str(spool),
        "RAG_INGEST_TEST_MODE": "hashing",
        "RAG_INGEST_FAULT_FILE": "/spool/fault.txt",
    }
    (spool / "fault.txt").write_text("0", encoding="utf-8")
    try:
        compose("up", "-d", "--wait", "postgres", "hatchet-lite", "qdrant", env=env)
        token = compose("exec", "-T", "hatchet-lite", "/hatchet-admin", "token", "create", "--config", "/config", "--tenant-id", TENANT, env=env).stdout.strip()
        env["HATCHET_CLIENT_TOKEN"] = token
        compose("--profile", "worker", "up", "-d", "--build", "worker", env=env)
        item = Stack(spool, env)
        item.wait_for_worker()

        # this process is a client of the same Hatchet and Qdrant; it reads the run files the worker writes
        os.environ.update(HATCHET_CLIENT_TOKEN=token, HATCHET_CLIENT_TLS_STRATEGY="none", INGEST_SPOOL_DIR=str(spool))
        from qdrant_client import QdrantClient

        from app.workflows.run_store import RunStore

        item.store = RunStore(spool)
        item.qdrant = QdrantClient(url="http://localhost:6333")
        yield item
    finally:
        if request.session.testsfailed:  # keep what the containers said, for the CI step that prints it
            logs = compose("--profile", "worker", "logs", "--tail", "300", env=env, check=False).stdout
            (BACKEND / "hatchet-integration-logs.txt").write_text(logs, encoding="utf-8")
        compose("--profile", "worker", "down", "-v", "--remove-orphans", env=env, check=False)  # the worker sits behind a profile, so down must name it


def archive(stack, name, files=FILES, plain=False, seed=1):
    return build_archive(stack.spool / "inbox" / name, files=files, seed=seed, plain=plain)


def by_file(status):
    return {item["file"]: item for item in status["files"]}


# ---------------------------------------------------------------- the run


def test_a_mixed_archive_is_ingested_and_the_bad_files_are_isolated(stack):
    archive(stack, "room.zip")
    status = stack.submit("room.zip")
    rows = by_file(status)
    assert status["state"] == "completed"
    assert status["counts"] == {"ingested": FILES - 7, "skipped_duplicate": 2, "failed": 5}

    assert rows["copies/doc-0000-copy.txt"]["status"] == "skipped_duplicate"
    assert rows["damaged/corrupt-report.pdf"]["reason"].startswith("unreadable_file:")
    assert rows["damaged/locked-report.pdf"]["status"] == "failed" and "decrypted" in rows["damaged/locked-report.pdf"]["reason"]
    assert rows["damaged/nested.zip"]["reason"].startswith("nested_archive:")
    assert rows["images/site-photo.png"]["reason"].startswith("extension_not_allowed:")
    assert rows["damaged/not-a-workbook.xlsx"]["status"] == "failed"

    points = stack.points()
    assert len(points) == sum(item["chunks"] for item in status["files"])
    names = {p.payload["metadata"]["file_name"] for p in points}
    assert names == {item["file"] for item in status["files"] if item["status"] == "ingested"}, "only the readable files are stored"
    for point in points:
        meta = point.payload["metadata"]
        assert meta["file_name"] == meta["archive_path"] and 0 <= meta["chunk_seq"] < meta["doc_chunks_total"]


def test_the_status_route_serves_the_same_table(stack, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core.config import settings
    from app.main import app

    run = sorted((stack.spool / "runs").iterdir(), key=lambda p: p.stat().st_mtime)[-1].name
    monkeypatch.setattr(settings, "ingest_spool_dir", str(stack.spool))
    served = TestClient(app).get(f"/api/ingest/{run}").json()
    assert served["state"] == "completed" and served["counts"] == stack.store.load(run)["counts"]
    assert {f["file"] for f in served["files"]} == {f["file"] for f in stack.store.load(run)["files"]}


def test_running_the_same_archive_again_does_no_new_work(stack):
    before = len(stack.points())
    status = stack.submit("room.zip")
    assert status["state"] == "completed"
    assert status["to_process"] == 3, "only the three files the loaders cannot read are tried again, and they fail again"
    assert status["counts"] == {"ingested": 0, "skipped_duplicate": FILES - 5, "failed": 5}
    assert len(stack.points()) == before


# ---------------------------------------------------------------- retries are for outages, not for bad files


def test_an_outage_is_retried_with_backoff_and_the_file_goes_in(stack):
    with zipfile.ZipFile(stack.spool / "inbox" / "late.zip", "w") as z:
        z.writestr("late/one.txt", make_text(901))
    stack.fault(2)  # the next two upserts fail as an embedding service that is down would
    started = time.time()
    status = stack.submit("late.zip")
    elapsed = time.time() - started
    assert by_file(status)["late/one.txt"]["status"] == "ingested" and status["counts"]["failed"] == 0
    assert stack.fault_left() == 0, "both simulated outages were hit"
    assert elapsed >= 5, f"two retries wait about 2 and 4 seconds, and this run took {elapsed:.1f}"


def test_a_file_that_keeps_failing_is_given_up_on_after_the_retries_and_finishes_on_a_later_run(stack):
    with zipfile.ZipFile(stack.spool / "inbox" / "down.zip", "w") as z:
        z.writestr("down/one.txt", make_text(902))
    stack.fault(10)
    first = stack.submit("down.zip")
    row = by_file(first)["down/one.txt"]
    assert row["status"] == "failed" and row["reason"].startswith("gave_up_after_4_attempts: ConnectionError")
    assert stack.fault_left() == 10 - 4, "one attempt and three retries, no more"
    stack.fault(0)
    second = stack.submit("down.zip")
    assert by_file(second)["down/one.txt"]["status"] == "ingested"


# ---------------------------------------------------------------- a refused archive


def test_a_zip_slip_archive_is_refused_and_nothing_from_it_is_stored(stack):
    with zipfile.ZipFile(stack.spool / "inbox" / "evil.zip", "w") as z:
        z.writestr("fine.txt", make_text(903))
        z.writestr("../../escape.txt", "x")
    before = len(stack.points())
    from app.workflows.trigger import submit_archive, wait_for_run

    run_id = submit_archive("/spool/inbox/evil.zip", stack.store)
    status = wait_for_run(stack.store, run_id, WAIT_SECONDS)
    assert status["state"] == "rejected" and status["error"]["code"] == "path_traversal" and status["files"] == []
    assert len(stack.points()) == before


# ---------------------------------------------------------------- a worker that dies in the middle


def test_a_worker_killed_midway_leaves_a_rerun_that_finishes_only_what_was_missing(stack):
    total = 400
    archive(stack, "big.zip", files=total, plain=True, seed=5)
    from app.workflows.trigger import submit_archive

    first_id = submit_archive("/spool/inbox/big.zip", stack.store)
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        rows = len(stack.store.rows(first_id))
        if 25 <= rows < total - 25:
            break
        if rows >= total - 25:
            pytest.fail("the archive finished before the worker could be killed; raise the file count")
        time.sleep(0.1)  # WAIT-OK: polls the first run's row count
    compose("kill", "worker", env=stack.env)
    done_before_kill = len(stack.store.rows(first_id))
    assert 0 < done_before_kill < total

    stack.restart_worker()
    second = stack.submit("big.zip")
    counts = second["counts"]
    assert second["state"] == "completed" and counts["failed"] == 0
    assert counts["ingested"] + counts["skipped_duplicate"] == total
    assert 0 < second["to_process"] < total, "the rerun skipped what the first run had already stored"

    groups = {}
    for point in stack.points():
        meta = point.payload["metadata"]
        if meta["file_name"].startswith("bulk/"):
            groups.setdefault(meta["doc_id"], []).append(meta["chunk_seq"])
    assert len(groups) == total
    for seqs in groups.values():
        assert sorted(seqs) == list(range(len(seqs))), "every file is complete and no chunk is stored twice"
