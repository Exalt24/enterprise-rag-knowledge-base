"""Switches for the Hatchet integration test. They do nothing unless RAG_INGEST_TEST_MODE is set in the worker's environment.

- RAG_INGEST_TEST_MODE=hashing   embed with the hashing test double (eval/embedders.py) so CI downloads no model
- RAG_INGEST_FAULT_FILE=<path>   a text file holding a number N: the next N upserts raise a connection error, as an
                                 embedding service that is down would, and the file counts down; used to watch retries
"""
import os
from pathlib import Path
from typing import List, Optional


class FlakyIndex:
    """Wraps an index and fails its add_documents while the fault file still holds a positive count."""

    def __init__(self, inner, fault_file: Path):
        self.inner = inner
        self.fault_file = fault_file

    def is_indexed(self, doc_id: str) -> bool:
        return self.inner.is_indexed(doc_id)

    def add_documents(self, documents, ids: Optional[List[str]] = None):
        if self.fault_file.exists():
            remaining = int(self.fault_file.read_text().strip() or "0")
            if remaining > 0:
                self.fault_file.write_text(str(remaining - 1))
                raise ConnectionError("simulated embedding service outage")
        return self.inner.add_documents(documents, ids=ids)


def wrap_index(index):
    """The index the worker uses, with the test switches applied when they are set."""
    if os.environ.get("RAG_INGEST_TEST_MODE") == "hashing":
        from app.services.embeddings import EmbeddingService
        from eval.embedders import HashingEmbeddings

        EmbeddingService._embeddings = HashingEmbeddings()
    fault_file = os.environ.get("RAG_INGEST_FAULT_FILE")
    return FlakyIndex(index, Path(fault_file)) if fault_file else index
