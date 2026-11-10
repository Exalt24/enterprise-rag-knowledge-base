"""Start an archive run and read its result. Used by the upload route and by `python -m app.workflows run`."""
import time
from pathlib import Path
from typing import Dict, Optional

from app.workflows.run_store import RunStore

DONE_STATES = ("completed", "rejected", "failed")


def submit_archive(zip_path, store: RunStore) -> str:
    """Start the ingest_archive workflow and return its run id.

    zip_path is passed on as written, so it must be the path the WORKER sees (the same as this process's when both
    run on one machine; a container mounts the spool folder under its own name). The id is Hatchet's own run id, and
    it is what GET /api/ingest/{run_id} takes.
    """
    from app.workflows.ingest import ArchiveInput, ingest_archive

    reference = ingest_archive.run_no_wait(ArchiveInput(zip_path=str(zip_path)))
    run_id = reference.workflow_run_id
    store.start(run_id, Path(str(zip_path)).name, state="queued", keep_existing=True)
    return run_id


def wait_for_run(store: RunStore, run_id: str, timeout_seconds: float, poll_seconds: float = 0.5) -> Optional[Dict]:
    """Poll the run's status files until it is completed, rejected or failed; None if the time runs out."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = store.load(run_id)
        if status and status.get("state") in DONE_STATES:
            return status
        time.sleep(poll_seconds)  # WAIT-OK: polls the run's status files, which the worker updates
    return None
