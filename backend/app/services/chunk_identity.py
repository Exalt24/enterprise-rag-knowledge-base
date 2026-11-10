"""
Chunk identity: how a stored chunk is named so it can be found again.

Three values are written into the metadata of every chunk at ingestion:

- doc_id          the SHA-256 of the file's bytes. The same bytes always give the same id, so the archive workflow can
                  ask "is this file already indexed" without comparing names, and a citation can name its document.
- chunk_seq       the chunk's position in the whole file, counted from 0 across pages and sheets. chunk_index already
                  exists, but the chunker restarts it for every page or sheet, so it cannot name one chunk of a file.
- doc_chunks_total  how many chunks the file produced. A file counts as indexed only when that many are stored, so a
                  crash half way through leaves a file that is retried, never one that is silently half searchable.

The point id of a chunk written by the archive workflow is derived from the first two (chunk_point_id), so writing the
same chunk twice replaces it instead of adding a copy.
"""
import hashlib
import uuid
from typing import Dict, List, Optional

from langchain_core.documents import Document

_POINT_NAMESPACE = uuid.UUID("c3f0a8f2-5a39-4f0e-9d0b-6f7c0f8a1d21")
_READ_BLOCK = 1024 * 1024


def compute_doc_id(path) -> str:
    """SHA-256 of the file's bytes, read in blocks so a large file is never held in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_point_id(doc_id: str, chunk_seq: int) -> str:
    """The Qdrant point id of one chunk: the same document and position always give the same id."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_seq}"))


def annotate_chunks(
    chunks: List[Document],
    doc_id: str,
    file_name: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> List[Document]:
    """Write doc_id, chunk_seq and doc_chunks_total into each chunk's metadata, in order. Returns the same list.

    file_name, when given, replaces the name the loader recorded (the archive workflow passes the path inside the zip).
    """
    total = len(chunks)
    for seq, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_seq"] = seq
        chunk.metadata["doc_chunks_total"] = total
        if file_name is not None:
            chunk.metadata["file_name"] = file_name
        if extra:
            chunk.metadata.update(extra)
    return chunks


def point_ids(chunks: List[Document]) -> List[str]:
    """Point ids for annotated chunks, in the same order."""
    return [chunk_point_id(chunk.metadata["doc_id"], chunk.metadata["chunk_seq"]) for chunk in chunks]


def display_location(metadata: dict) -> Dict[str, Optional[str]]:
    """Where a chunk sits in its file, as a reader would say it: a PDF page, a sheet and cell range, or a chunk number.

    The loaders already record `page` (PDF, one document per page) and `location` (xlsx, sheet!cells). A Word, text or
    Markdown file has neither, so it is named by chunk number.
    """
    meta = metadata or {}
    page = meta.get("page")
    if isinstance(page, int) and page > 0:
        return {"kind": "page", "label": f"Page {page}"}
    location = meta.get("location")
    if isinstance(location, str) and location:
        return {"kind": "cells", "label": location}
    seq = meta.get("chunk_seq")
    total = meta.get("doc_chunks_total")
    if isinstance(seq, int):
        suffix = f" of {total}" if isinstance(total, int) and total > 0 else ""
        return {"kind": "chunk", "label": f"Chunk {seq + 1}{suffix}"}
    return {"kind": "chunk", "label": "Chunk"}

