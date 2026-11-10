"""
Source view: the passage a citation points at, with the chunks around it.

The reviewer opens a citation and needs to read the cited text where it sits in its document, so the lookup returns the
chunk and up to `context` chunks on each side in file order. This module only shapes what the vector store returns;
the store does the lookup (VectorStoreService.get_chunk_window).
"""
from typing import Dict, List, Optional

from langchain_core.documents import Document

from app.services.chunk_identity import display_location

MAX_CONTEXT = 5
DEFAULT_CONTEXT = 2


def _passage(doc: Document) -> Dict:
    meta = doc.metadata or {}
    return {
        "chunk_id": meta.get("chunk_seq"),
        "text": doc.page_content,
        "page": meta.get("page"),
        "location": display_location(meta),
    }


def build_chunk_view(window: List[Document], doc_id: str, chunk_id: int) -> Optional[Dict]:
    """The response body for one chunk and its neighbours, or None when the chunk is not in the window."""
    cited = next((d for d in window if (d.metadata or {}).get("chunk_seq") == chunk_id), None)
    if cited is None:
        return None
    meta = cited.metadata or {}
    return {
        "doc_id": doc_id,
        "file_name": meta.get("file_name", "unknown"),
        "file_type": meta.get("file_type"),
        "total_chunks": meta.get("doc_chunks_total"),
        "chunk": _passage(cited),
        "before": [_passage(d) for d in window if d.metadata.get("chunk_seq") < chunk_id],
        "after": [_passage(d) for d in window if d.metadata.get("chunk_seq") > chunk_id],
    }
