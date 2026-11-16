"""The two-PDF check as a FastAPI route, so another system (a workflow, a quoting tool) can call it.

`POST /compare` takes two PDF uploads (`document_a`, `document_b`) and the requirements as JSON text (`requirements`), and answers the structured report. It runs the same code as
the command line, with no judge (a `judge` requirement comes back `unclear`). A file it cannot read is a 422 with the reason, never a 500 and never a pass.

The approval checkpoint is not an endpoint on purpose: the reviewer's decision is a person's act, taken on the report's hash, and is written by `python -m app.compare review`.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .check import compare
from .pages import MAX_BYTES, CompareError, read_pdf
from .requirements import MAX_REQUIREMENTS, parse_requirements

router = APIRouter()

#: A requirements document is text; this is far more than MAX_REQUIREMENTS entries need.
MAX_REQUIREMENTS_BYTES = 1024 * 1024


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="%s is larger than %d MB" % (upload.filename or "a file", MAX_BYTES // (1024 * 1024)))
    return data


@router.post("/compare")
async def compare_documents(document_a: UploadFile = File(...), document_b: UploadFile = File(...), requirements: str = Form(...)) -> dict:
    if len(requirements.encode("utf-8")) > MAX_REQUIREMENTS_BYTES:
        raise HTTPException(status_code=413, detail="the requirements are larger than 1 MB (at most %d entries)" % MAX_REQUIREMENTS)
    data_a, data_b = await _read_upload(document_a), await _read_upload(document_b)

    def work() -> dict:
        import hashlib

        try:
            raw = json.loads(requirements)
        except ValueError:
            raise CompareError("the requirements are not valid JSON") from None
        parsed = parse_requirements(raw)
        report = compare(read_pdf(data_a, document_a.filename or "document A"), read_pdf(data_b, document_b.filename or "document B"), parsed,
                         hashlib.sha256(requirements.encode("utf-8")).hexdigest())
        return report.to_dict()

    try:
        return await run_in_threadpool(work)
    except CompareError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
