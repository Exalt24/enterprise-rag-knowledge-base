"""The reviewer source view: chunk identity at ingestion, the chunk lookup, and GET /api/documents/{doc_id}/chunks/{chunk_id}.

Runs the repo's own ingestion service and vector store against an in-memory Qdrant with the hashing embedder, and the
real FastAPI app through its test client. No network, no model, no language model: where an answer is needed the
generation call is replaced by a stub, and the citations are still the chunks retrieval returned.
Run: pytest tests/test_source_view.py
"""
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.chunk_identity import annotate_chunks, chunk_point_id, display_location  # noqa: E402
from app.services.ingestion import IngestionService  # noqa: E402
from app.services.rag import rag_service  # noqa: E402
from app.services.vector_store import vector_store  # noqa: E402
from eval.archive_fixture import make_pdf, make_text, make_xlsx  # noqa: E402
from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import in_memory_store  # noqa: E402


@pytest.fixture
def store():
    with in_memory_store(HashingEmbeddings(), []) as s:
        yield s


@pytest.fixture
def client():
    return TestClient(app)


def ingest(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    result = IngestionService().ingest_file(str(path))
    assert result["success"], result
    return hashlib.sha256(data).hexdigest(), result["chunks_stored"]


# ---------------------------------------------------------------- identity written at ingestion


def test_ingestion_writes_the_file_hash_and_a_position_that_runs_across_pages(tmp_path, store):
    doc_id, stored = ingest(tmp_path, "report.pdf", make_pdf(4, pages=3))
    points, _ = store._client.scroll("enterprise_rag", limit=100, with_payload=True)
    metas = sorted((p.payload["metadata"] for p in points), key=lambda m: m["chunk_seq"])
    assert len(metas) == stored
    assert {m["doc_id"] for m in metas} == {doc_id}
    assert [m["chunk_seq"] for m in metas] == list(range(stored))
    assert {m["doc_chunks_total"] for m in metas} == {stored}
    pages = [m["page"] for m in metas]
    assert pages == sorted(pages) and set(pages) == {1, 2, 3}, "the PDF loader keeps the page, and chunk_seq keeps counting across pages"
    assert [m["chunk_index"] for m in metas].count(0) == 3, "chunk_index restarts on every page, which is why chunk_seq exists"


def test_the_same_bytes_always_give_the_same_document_id_and_different_bytes_do_not(tmp_path):
    one, two = tmp_path / "a.txt", tmp_path / "b.txt"
    one.write_bytes(b"same")
    two.write_bytes(b"same")
    from app.services.chunk_identity import compute_doc_id

    assert compute_doc_id(one) == compute_doc_id(two) == hashlib.sha256(b"same").hexdigest()
    two.write_bytes(b"different")
    assert compute_doc_id(two) != compute_doc_id(one)


def test_point_ids_depend_on_the_document_and_the_position():
    assert chunk_point_id("a" * 64, 0) == chunk_point_id("a" * 64, 0)
    assert chunk_point_id("a" * 64, 0) != chunk_point_id("a" * 64, 1)
    assert chunk_point_id("a" * 64, 0) != chunk_point_id("b" * 64, 0)


def test_a_location_is_a_page_then_cells_then_a_chunk_number():
    assert display_location({"page": 3, "chunk_seq": 9}) == {"kind": "page", "label": "Page 3"}
    assert display_location({"location": "Budget!A2:C26", "chunk_seq": 1}) == {"kind": "cells", "label": "Budget!A2:C26"}
    assert display_location({"chunk_seq": 3, "doc_chunks_total": 7}) == {"kind": "chunk", "label": "Chunk 4 of 7"}


def test_annotating_renames_the_file_only_when_asked():
    from langchain_core.documents import Document

    chunks = [Document(page_content="a", metadata={"file_name": "x.txt"}), Document(page_content="b", metadata={"file_name": "x.txt"})]
    annotate_chunks(chunks, "d" * 64)
    assert [c.metadata["file_name"] for c in chunks] == ["x.txt", "x.txt"]
    annotate_chunks(chunks, "d" * 64, file_name="folder/x.txt")
    assert [c.metadata["file_name"] for c in chunks] == ["folder/x.txt", "folder/x.txt"]


# ---------------------------------------------------------------- the lookup


def test_the_window_holds_the_chunk_and_its_neighbours_in_file_order(tmp_path, store):
    doc_id, stored = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    assert stored >= 8
    window = store.get_chunk_window(doc_id, 4, 2)
    assert [d.metadata["chunk_seq"] for d in window] == [2, 3, 4, 5, 6]


def test_the_window_stops_at_the_start_and_the_end_of_the_document(tmp_path, store):
    doc_id, stored = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    assert [d.metadata["chunk_seq"] for d in store.get_chunk_window(doc_id, 0, 2)] == [0, 1, 2]
    assert [d.metadata["chunk_seq"] for d in store.get_chunk_window(doc_id, stored - 1, 2)] == [stored - 3, stored - 2, stored - 1]
    assert [d.metadata["chunk_seq"] for d in store.get_chunk_window(doc_id, 3, 0)] == [3]


def test_a_chunk_that_does_not_exist_gives_an_empty_window(tmp_path, store):
    doc_id, stored = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    assert store.get_chunk_window(doc_id, stored + 5, 2) == []
    assert store.get_chunk_window(doc_id, stored, 2) == [], "the neighbours of a chunk that is not there are not returned"
    assert store.get_chunk_window("0" * 64, 0, 2) == []


def test_the_window_never_mixes_in_another_document(tmp_path, store):
    first, _ = ingest(tmp_path, "one.txt", make_text(5, paragraphs=40).encode())
    second, _ = ingest(tmp_path, "two.txt", make_text(6, paragraphs=40).encode())
    assert {d.metadata["doc_id"] for d in store.get_chunk_window(first, 4, 2)} == {first}
    assert {d.metadata["doc_id"] for d in store.get_chunk_window(second, 4, 2)} == {second}


def test_a_document_uploaded_twice_still_gives_one_copy_of_each_chunk_in_the_window(tmp_path, store):
    data = make_text(5, paragraphs=40).encode()
    doc_id, _ = ingest(tmp_path, "one.txt", data)
    ingest(tmp_path, "one-again.txt", data)  # random point ids, so the second upload adds a second copy
    assert [d.metadata["chunk_seq"] for d in store.get_chunk_window(doc_id, 4, 2)] == [2, 3, 4, 5, 6]


# ---------------------------------------------------------------- the route


def fetch(client, doc_id, chunk_id, context=None):
    suffix = "" if context is None else f"?context={context}"
    return client.get(f"/api/documents/{doc_id}/chunks/{chunk_id}{suffix}")


def test_the_route_returns_the_passage_exactly_as_stored_with_neighbours_above_and_below(tmp_path, store, client):
    doc_id, stored = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    body = fetch(client, doc_id, 4, 2).json()
    points, _ = store._client.scroll("enterprise_rag", limit=100, with_payload=True)
    stored_text = {p.payload["metadata"]["chunk_seq"]: p.payload["page_content"] for p in points}
    assert body["doc_id"] == doc_id and body["file_name"] == "long.txt" and body["total_chunks"] == stored
    assert body["chunk"]["chunk_id"] == 4 and body["chunk"]["text"] == stored_text[4]
    assert [c["chunk_id"] for c in body["before"]] == [2, 3] and [c["text"] for c in body["before"]] == [stored_text[2], stored_text[3]]
    assert [c["chunk_id"] for c in body["after"]] == [5, 6] and [c["text"] for c in body["after"]] == [stored_text[5], stored_text[6]]
    assert body["chunk"]["location"] == {"kind": "chunk", "label": f"Chunk 5 of {stored}"}


def test_the_default_context_is_two_chunks_each_side(tmp_path, store, client):
    doc_id, _ = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    body = fetch(client, doc_id, 5).json()
    assert len(body["before"]) == 2 and len(body["after"]) == 2


def test_a_pdf_passage_is_located_by_page(tmp_path, store, client):
    doc_id, _ = ingest(tmp_path, "report.pdf", make_pdf(4, pages=3))
    points, _ = store._client.scroll("enterprise_rag", limit=100, with_payload=True)
    pages = {p.payload["metadata"]["chunk_seq"]: p.payload["metadata"]["page"] for p in points}
    first_of_page_two = min(seq for seq, page in pages.items() if page == 2)
    body = fetch(client, doc_id, first_of_page_two).json()
    assert body["chunk"]["page"] == 2 and body["chunk"]["location"] == {"kind": "page", "label": "Page 2"}
    assert body["before"][-1]["location"] == {"kind": "page", "label": "Page 1"}, "the chunk just before it is the end of page 1"
    assert body["after"][0]["page"] in (2, 3)


def test_a_workbook_passage_is_located_by_sheet_and_cells(tmp_path, store, client):
    doc_id, _ = ingest(tmp_path, "plan.xlsx", make_xlsx(4))
    body = fetch(client, doc_id, 0).json()
    assert body["chunk"]["location"]["kind"] == "cells" and body["chunk"]["location"]["label"].startswith("Schedule!A")


@pytest.mark.parametrize("doc_id", ["nonsense", "../etc/passwd", "A" * 64, "a" * 63])
def test_something_that_is_not_a_document_id_is_a_404(store, client, doc_id):
    assert fetch(client, doc_id, 0).status_code == 404


def test_a_missing_document_or_chunk_is_a_404(tmp_path, store, client):
    doc_id, stored = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    assert fetch(client, "0" * 64, 0).status_code == 404
    assert fetch(client, doc_id, stored).status_code == 404


@pytest.mark.parametrize("chunk_id,context", [(-1, 2), (0, -1), (0, 6)])
def test_out_of_range_numbers_are_a_422(tmp_path, store, client, chunk_id, context):
    doc_id, _ = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    assert fetch(client, doc_id, chunk_id, context).status_code == 422


def test_the_route_reads_only_the_configured_collection(tmp_path, store, client, monkeypatch):
    doc_id, _ = ingest(tmp_path, "long.txt", make_text(5, paragraphs=40).encode())
    from qdrant_client.models import Distance, VectorParams

    from app.core.config import settings

    store._client.create_collection("some_other_collection", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    assert fetch(client, doc_id, 4).status_code == 200
    monkeypatch.setattr(settings, "qdrant_collection", "some_other_collection")
    assert fetch(client, doc_id, 4).status_code == 404


# ---------------------------------------------------------------- citations carry the ids


def stub_generation(monkeypatch):
    stub = SimpleNamespace(answer="stub answer", model_used="stub")
    monkeypatch.setattr(rag_service.generation, "generate", lambda question, context: stub)
    monkeypatch.setattr(rag_service.generation, "generate_stream", lambda question, context: ("stub", iter(["stub ", "answer"])))


def test_a_citation_carries_the_ids_that_open_the_exact_chunk(tmp_path, store, client, monkeypatch):
    stub_generation(monkeypatch)
    doc_id, _ = ingest(tmp_path, "lease.txt", make_text(7, paragraphs=40).encode())
    answer = client.post("/api/query", json={"question": "site lease parcel option to extend notice", "k": 3}).json()
    assert answer["sources"], "retrieval returned nothing"
    for source in answer["sources"]:
        assert source["doc_id"] == doc_id and isinstance(source["chunk_id"], int)
        opened = fetch(client, source["doc_id"], source["chunk_id"]).json()
        assert opened["chunk"]["text"].startswith(source["content_preview"].removesuffix("..."))


def test_the_streamed_citations_carry_the_same_ids(tmp_path, store, client, monkeypatch):
    stub_generation(monkeypatch)
    doc_id, _ = ingest(tmp_path, "lease.txt", make_text(7, paragraphs=40).encode())
    sources, _ = rag_service.query_stream("site lease parcel option to extend notice", k=3)
    assert sources and all(s["doc_id"] == doc_id and isinstance(s["chunk_id"], int) for s in sources)


def test_a_chunk_stored_before_the_source_view_existed_has_no_ids_and_the_citation_still_works(store, client, monkeypatch):
    stub_generation(monkeypatch)
    from langchain_core.documents import Document

    vector_store.add_documents([Document(page_content="Legacy parcel lease text about an option to extend.", metadata={"file_name": "old.txt", "chunk_index": 0})])
    answer = client.post("/api/query", json={"question": "legacy parcel lease option to extend", "k": 1}).json()
    assert answer["sources"][0]["file_name"] == "old.txt"
    assert answer["sources"][0]["doc_id"] is None and answer["sources"][0]["chunk_id"] is None


# ---------------------------------------------------------------- the existing upload route is unchanged


def test_the_single_document_upload_route_still_answers_the_same_way(tmp_path, store, client, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response = client.post("/api/ingest", files={"file": ("note.txt", make_text(9).encode(), "text/plain")})
    body = response.json()
    assert response.status_code == 200
    assert body["success"] is True and body["message"] == "Document ingested successfully"
    assert body["file_name"] == "note.txt" and body["chunks_stored"] > 0
    assert set(body) == {"success", "message", "file_name", "chunks_stored", "error"}
    assert (tmp_path / "data" / "documents" / "note.txt").exists()
    refused = client.post("/api/ingest", files={"file": ("note.exe", b"x", "application/octet-stream")}).json()
    assert refused["success"] is False and refused["message"] == "Unsupported file format"
