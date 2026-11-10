"""
Run store: the per archive result table, kept as small JSON files so the API can read what the worker wrote.

    <spool>/runs/<run_id>/manifest.json   archive name, state, counts, the archive level error if there was one (state: queued, validating,
                                          running, completed, rejected or failed)
    <spool>/runs/<run_id>/rows/<n>.json   one row per file, written by whichever process finished that file

The Hatchet worker writes, the API only reads, and each writer owns its own file, so there is nothing to lock. A row is
{file, status, reason, chunks, doc_id} with status one of ingested, skipped_duplicate or failed.
"""
import contextlib
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

STATUSES = ("ingested", "skipped_duplicate", "failed")
RUN_ID = re.compile(r"[0-9A-Za-z][0-9A-Za-z-]{7,63}")


class InvalidRunId(ValueError):
    pass


def checked_run_id(run_id: str) -> str:
    """Run ids become folder names, so only the shapes Hatchet issues (UUIDs) are accepted."""
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise InvalidRunId(f"{run_id!r} is not a run id")
    return run_id


def row(file: str, status: str, reason: str = "", chunks: int = 0, doc_id: Optional[str] = None) -> Dict:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    return {"file": file, "status": status, "reason": reason, "chunks": chunks, "doc_id": doc_id}


def _write(path: Path, payload: Dict) -> None:
    """Write a file in one rename, so a reader never sees half of it.

    Two processes can write the same manifest at the same moment (the one that submitted the run and the worker that
    picked it up), so each writes its own temporary file. A shared folder on Docker Desktop can also answer "no such
    file" for a folder another process has just created, and Windows refuses a rename onto a file another writer is
    replacing, so a failed write is tried again.
    """
    for attempt in range(5):
        temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_text(json.dumps(payload), encoding="utf-8")
            temp.replace(path)
            return
        except (FileNotFoundError, PermissionError):  # PermissionError: Windows refuses a rename onto a file another writer is replacing
            with contextlib.suppress(OSError):
                temp.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))  # WAIT-OK: a short retry on a shared folder that has not caught up


class RunStore:
    def __init__(self, spool_dir):
        self.root = Path(spool_dir) / "runs"

    def _dir(self, run_id: str) -> Path:
        return self.root / checked_run_id(run_id)

    def start(self, run_id: str, archive: str, state: str = "queued", keep_existing: bool = False) -> None:
        """Write the first manifest. keep_existing leaves one alone: the worker can pick a run up before the process
        that submitted it has written this, and the worker's later state must not be overwritten by "queued"."""
        if keep_existing and self.read_manifest(run_id) is not None:
            return
        _write(self._dir(run_id) / "manifest.json", {"run_id": run_id, "archive": archive, "state": state, "started_at": time.time()})

    def update(self, run_id: str, **fields) -> None:
        manifest = self.read_manifest(run_id) or {"run_id": run_id}
        manifest.update(fields)
        _write(self._dir(run_id) / "manifest.json", manifest)

    def add_row(self, run_id: str, item: Dict) -> None:
        # named after the path inside the archive, so a retried task rewrites its own row instead of adding a second
        # and two paths with the same content (a duplicate) keep a row each
        key = hashlib.sha1(item["file"].encode("utf-8")).hexdigest()
        _write(self._dir(run_id) / "rows" / f"{key}.json", item)

    def read_manifest(self, run_id: str) -> Optional[Dict]:
        path = self._dir(run_id) / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def rows(self, run_id: str) -> List[Dict]:
        folder = self._dir(run_id) / "rows"
        if not folder.exists():
            return []
        found = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
        return sorted(found, key=lambda r: r["file"])

    def load(self, run_id: str) -> Optional[Dict]:
        """The manifest with the rows and the counts per status, or None for a run nobody recorded."""
        manifest = self.read_manifest(run_id)
        if manifest is None:
            return None
        files = self.rows(run_id)
        counts = {status: sum(1 for r in files if r["status"] == status) for status in STATUSES}
        return {**manifest, "counts": counts, "files": files}
