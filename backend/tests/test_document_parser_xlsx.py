"""The workbook (.xlsx) loader in document_parser.py.

Every case builds its own small workbook with openpyxl in a temporary folder, so nothing needs Qdrant, a model or a
network. The happy path checks the text and the citation location (sheet and cell range) of every block; the failure
cases check that a damaged, password protected, oversized or empty workbook is refused with a plain message and never
half read. eval/mutation_check.py breaks the loader one guard at a time and requires a failure from this file.
Run: pytest tests/test_document_parser_xlsx.py
"""
import os
import re
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from openpyxl import Workbook  # noqa: E402

from app.api.schemas import Source  # noqa: E402
from app.services.chunking import TextChunker  # noqa: E402
from app.services.document_parser import DocumentParser  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


def _save(wb: Workbook, tmp_path: Path, name: str = "book.xlsx") -> Path:
    path = tmp_path / name
    wb.save(path)
    return path


def _people_workbook(tmp_path: Path, rows: int = 3) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Staff"
    ws.append(["Name", "Team", "Salary"])
    for i in range(rows):
        ws.append([f"Person {i}", "Ops" if i % 2 else "Data", 1000 + i])
    return _save(wb, tmp_path)


def _with_cached_formula_value(path: Path, cell: str, value: str) -> None:
    """openpyxl writes a formula with no saved result, like a file nobody opened in Excel. Put the result in, as Excel does."""
    rewritten = path.with_suffix(".tmp")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(rewritten, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                text = data.decode("utf-8")
                pattern = re.compile(r'(<c r="%s"[^>]*>\s*<f>[^<]*</f>)\s*(<v\s*/>|<v></v>)' % cell)
                text, count = pattern.subn(lambda m: m.group(1) + "<v>" + value + "</v>", text)
                assert count == 1, "the test helper could not find the formula cell to give a saved result"
                data = text.encode("utf-8")
            target.writestr(item, data)
    rewritten.replace(path)


# ---------------------------------------------------------------- the happy path


def test_xlsx_is_a_supported_format_and_xls_is_not(tmp_path):
    assert DocumentParser.is_supported("a.xlsx")
    assert DocumentParser.is_supported("A.XLSX")
    assert not DocumentParser.is_supported("a.xls")
    old = tmp_path / "a.xls"
    old.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentParser.parse(str(old))


def test_rows_become_one_block_with_sheet_headers_and_cell_range(tmp_path):
    docs = DocumentParser.parse(str(_people_workbook(tmp_path, rows=3)))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["file_type"] == "xlsx"
    assert doc.metadata["sheet"] == "Staff"
    assert doc.metadata["cell_range"] == "A2:C4"
    assert doc.metadata["location"] == "Staff!A2:C4"
    assert (doc.metadata["row_start"], doc.metadata["row_end"]) == (2, 4)
    assert doc.page_content.splitlines()[:3] == ["Sheet: Staff", "Columns: Name | Team | Salary", "Rows 2 to 4 (A2:C4)"]
    assert "Name: Person 1; Team: Ops; Salary: 1001" in doc.page_content
    assert doc.metadata["char_count"] == len(doc.page_content)


def test_each_sheet_gets_its_own_blocks_and_a_sheet_name_with_a_space_is_quoted(tmp_path):
    wb = Workbook()
    first = wb.active
    first.title = "Q1 Budget"
    first.append(["Item", "Cost"])
    first.append(["Rent", 500])
    second = wb.create_sheet("Notes")
    second.append(["Topic", "Detail"])
    second.append(["Owner", "Finance"])
    docs = DocumentParser.parse(str(_save(wb, tmp_path)))
    assert [d.metadata["location"] for d in docs] == ["'Q1 Budget'!A2:B2", "Notes!A2:B2"]
    assert "Item: Rent; Cost: 500" in docs[0].page_content
    assert "Topic: Owner; Detail: Finance" in docs[1].page_content


def test_a_blank_row_keeps_the_real_row_numbers(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Score"])
    ws.append(["Ana", 1])
    ws.append([None, None])
    ws.append([None, None])
    ws.append(["Ben", 2])
    doc = DocumentParser.parse(str(_save(wb, tmp_path)))[0]
    assert doc.metadata["cell_range"] == "A2:B5"
    assert (doc.metadata["row_start"], doc.metadata["row_end"]) == (2, 5)
    assert "Name: Ben; Score: 2" in doc.page_content


def test_a_range_starts_at_the_first_filled_column(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws["C1"] = "Item"
    ws["D1"] = "Cost"
    ws["C2"] = "Rent"
    ws["D2"] = 500
    doc = DocumentParser.parse(str(_save(wb, tmp_path)))[0]
    assert doc.metadata["cell_range"] == "C2:D2"
    assert doc.metadata["location"] == "Sheet!C2:D2"


def test_long_sheets_are_cut_into_blocks_that_fit_the_chunker_and_cover_every_row(tmp_path):
    docs = DocumentParser.parse(str(_people_workbook(tmp_path, rows=60)))
    assert len(docs) >= 3
    covered = []
    for doc in docs:
        start, end = doc.metadata["row_start"], doc.metadata["row_end"]
        assert end - start + 1 <= DocumentParser.XLSX_BLOCK_ROWS
        assert len(doc.page_content) <= 500  # the chunker's default size, so one block stays one chunk
        covered.extend(range(start, end + 1))
    assert covered == list(range(2, 62))
    chunks = TextChunker.chunk_documents(docs)
    assert len(chunks) == len(docs)
    assert [c.metadata["location"] for c in chunks] == [d.metadata["location"] for d in docs]


def test_a_sheet_without_a_text_header_row_is_read_as_plain_rows(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append([1, 2, 3])
    ws.append([4, 5, 6])
    doc = DocumentParser.parse(str(_save(wb, tmp_path)))[0]
    assert doc.metadata["cell_range"] == "A1:C2"
    assert "Columns:" not in doc.page_content
    assert "1 | 2 | 3" in doc.page_content


def test_cell_values_are_shown_as_a_person_reads_them(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["When", "Paid", "Amount", "Time"])
    ws.append([date(2026, 3, 9), True, 12.0, datetime(2026, 3, 9, 14, 30)])
    ws.append([date(2026, 3, 10), False, 12.5, None])
    text = DocumentParser.parse(str(_save(wb, tmp_path)))[0].page_content
    assert "When: 2026-03-09; Paid: TRUE; Amount: 12; Time: 2026-03-09 14:30:00" in text
    assert "When: 2026-03-10; Paid: FALSE; Amount: 12.5" in text


def test_a_very_long_cell_is_cut_and_marked(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Note"])
    ws.append(["word " * 200])
    text = DocumentParser.parse(str(_save(wb, tmp_path)))[0].page_content
    line = [ln for ln in text.splitlines() if ln.startswith("Note: ")][0]
    assert line.endswith("...")
    assert len(line) <= len("Note: ") + DocumentParser.XLSX_CELL_CHARS


def test_formulas_are_read_as_the_value_excel_saved(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Item", "Cost"])
    ws.append(["Rent", 500])
    ws.append(["Power", 250])
    ws.append(["Total", "=SUM(B2:B3)"])
    path = _save(wb, tmp_path)
    _with_cached_formula_value(path, "B4", "750")
    text = DocumentParser.parse(str(path))[0].page_content
    assert "Item: Total; Cost: 750" in text
    assert "SUM" not in text


def test_a_formula_with_no_saved_value_comes_out_empty_not_as_formula_text(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.append(["Item", "Cost"])
    ws.append(["Rent", 500])
    ws.append(["Total", "=SUM(B2:B2)"])
    text = DocumentParser.parse(str(_save(wb, tmp_path)))[0].page_content
    assert "SUM" not in text
    assert "Item: Total" in text


def test_a_hidden_sheet_is_left_out(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Shown"
    ws.append(["A", "B"])
    ws.append(["x", "y"])
    secret = wb.create_sheet("Secret")
    secret.append(["A", "B"])
    secret.append(["hidden", "data"])
    secret.sheet_state = "hidden"
    docs = DocumentParser.parse(str(_save(wb, tmp_path)))
    assert [d.metadata["sheet"] for d in docs] == ["Shown"]
    assert all("hidden" not in d.page_content for d in docs)


def test_the_citation_schema_carries_the_location():
    source = Source(file_name="book.xlsx", location="Staff!A2:C4", content_preview="...")
    assert source.model_dump()["location"] == "Staff!A2:C4"
    assert Source(file_name="a.pdf", page=2, content_preview="...").location is None


# ---------------------------------------------------------------- refusals


def test_a_missing_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentParser.parse(str(tmp_path / "nope.xlsx"))


@pytest.mark.parametrize(
    "content",
    [b"", b"this is not a workbook", b"PK\x03\x04 cut off in the middle"],
    ids=["empty-file", "plain-text", "truncated-zip"],
)
def test_a_damaged_file_is_refused_with_a_clear_error(tmp_path, content):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="not a valid .xlsx workbook"):
        DocumentParser.parse(str(path))


def test_a_zip_that_is_not_a_workbook_is_refused(tmp_path):
    path = tmp_path / "other.xlsx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hello.txt", "hi")
    with pytest.raises(ValueError, match="Failed to parse XLSX"):
        DocumentParser.parse(str(path))


def test_a_password_protected_workbook_is_refused_and_says_so(tmp_path):
    # Excel stores an encrypted workbook in an OLE container, not a zip: it starts with these eight bytes.
    path = tmp_path / "locked.xlsx"
    path.write_bytes(bytes.fromhex("d0cf11e0a1b11ae1") + b"\x00" * 512)
    with pytest.raises(ValueError, match="password protected"):
        DocumentParser.parse(str(path))


def test_a_zip_with_encrypted_entries_is_refused_and_says_so(tmp_path):
    path = _people_workbook(tmp_path)
    data = bytearray(path.read_bytes())
    marker = data.index(b"PK\x01\x02")  # first central directory entry; its flag field is 8 bytes in
    data[marker + 8] |= 0x1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="password protected"):
        DocumentParser.parse(str(path))


def test_a_workbook_that_unpacks_too_large_is_refused_before_it_is_opened(tmp_path, monkeypatch):
    path = _people_workbook(tmp_path)
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_UNCOMPRESSED_MB", 0)
    with pytest.raises(ValueError, match="unpacks to more than"):
        DocumentParser.parse(str(path))


def test_a_workbook_over_the_row_limit_is_refused_and_one_at_the_limit_is_read(tmp_path, monkeypatch):
    path = _people_workbook(tmp_path, rows=9)  # 10 rows with the header
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_ROWS", 10)
    assert DocumentParser.parse(str(path))
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_ROWS", 9)
    with pytest.raises(ValueError, match="more than 9 rows"):
        DocumentParser.parse(str(path))


def test_the_row_limit_counts_rows_across_sheets(tmp_path, monkeypatch):
    wb = Workbook()
    for title in ("One", "Two"):
        ws = wb.active if title == "One" else wb.create_sheet(title)
        ws.title = title
        ws.append(["A"])
        ws.append(["x"])
        ws.append(["y"])
    path = _save(wb, tmp_path)
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_ROWS", 5)
    with pytest.raises(ValueError, match="more than 5 rows"):
        DocumentParser.parse(str(path))


def test_a_workbook_over_the_column_limit_is_refused_and_one_at_the_limit_is_read(tmp_path, monkeypatch):
    wb = Workbook()
    ws = wb.active
    ws.append([f"c{i}" for i in range(6)])
    ws.append([f"v{i}" for i in range(6)])
    path = _save(wb, tmp_path)
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_COLUMNS", 6)
    assert DocumentParser.parse(str(path))
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_COLUMNS", 5)
    with pytest.raises(ValueError, match="more than 5 columns"):
        DocumentParser.parse(str(path))


def test_a_workbook_over_the_sheet_limit_is_refused_and_one_at_the_limit_is_read(tmp_path, monkeypatch):
    wb = Workbook()
    for i in range(3):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.title = f"S{i}"
        ws.append(["A"])
        ws.append(["x"])
    path = _save(wb, tmp_path)
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_SHEETS", 3)
    assert len(DocumentParser.parse(str(path))) == 3
    monkeypatch.setattr(DocumentParser, "XLSX_MAX_SHEETS", 2)
    with pytest.raises(ValueError, match="has 3 sheets"):
        DocumentParser.parse(str(path))


def test_the_block_row_limit_decides_where_a_block_ends(tmp_path, monkeypatch):
    path = _people_workbook(tmp_path, rows=6)
    monkeypatch.setattr(DocumentParser, "XLSX_BLOCK_ROWS", 3)
    docs = DocumentParser.parse(str(path))
    assert [d.metadata["cell_range"] for d in docs] == ["A2:C4", "A5:C7"]


@pytest.mark.parametrize("kind", ["no-sheets-with-text", "only-hidden", "only-a-header"])
def test_a_workbook_with_no_text_is_refused_as_empty(tmp_path, kind):
    wb = Workbook()
    ws = wb.active
    if kind == "only-hidden":
        ws.append(["A"])
        ws.append(["x"])
        ws.sheet_state = "hidden"
        wb.create_sheet("Other")  # a workbook needs one visible sheet to save
    elif kind == "only-a-header":
        ws.append(["Name", "Team"])
    with pytest.raises(ValueError, match="XLSX file is empty"):
        DocumentParser.parse(str(_save(wb, tmp_path)))


# ---------------------------------------------------------------- the lists that must agree


def test_the_upload_route_and_the_upload_box_accept_what_the_parser_reads():
    """Three hand-typed lists name the formats: the parser, the /ingest route and the front end. They must stay one list."""
    wanted = DocumentParser.SUPPORTED_EXTENSIONS

    routes = (BACKEND / "app" / "api" / "routes.py").read_text(encoding="utf-8")
    in_route = set(re.findall(r"'(\.\w+)'", re.search(r"suffix\.lower\(\) not in \{([^}]*)\}", routes).group(1)))
    assert in_route == wanted

    box = (BACKEND.parent / "frontend" / "src" / "components" / "DocumentUpload.tsx").read_text(encoding="utf-8")
    in_list = set(re.findall(r'"(\.\w+)"', re.search(r"allowedTypes = \[([^\]]*)\]", box).group(1)))
    in_accept = set(re.findall(r"\.\w+", re.search(r'accept="([^"]*)"', box).group(1)))
    assert in_list == wanted
    assert in_accept == wanted


# ---------------------------------------------------------------- through the real store


def test_a_workbook_row_is_found_by_search_and_cited_with_its_cells(tmp_path):
    """Workbook -> parser -> chunker -> the repo's vector store (in memory, hashing embedder) -> search -> context text."""
    from eval.embedders import HashingEmbeddings
    from eval.harness import in_memory_store
    from app.services.retrieval import retrieval_service

    wb = Workbook()
    ws = wb.active
    ws.title = "Vendors"
    ws.append(["Vendor", "Contact", "Terms"])
    ws.append(["Northwind Freight", "dispatch desk", "net 30 days"])
    ws.append(["Harbor Packaging", "account manager", "prepaid in full"])
    docs = DocumentParser.parse(str(_save(wb, tmp_path, "vendors.xlsx")))
    chunks = TextChunker.chunk_documents(docs)
    with in_memory_store(HashingEmbeddings(), chunks) as store:
        found = store.search("Harbor Packaging prepaid in full", k=1)
    assert found and found[0].metadata["location"] == "Vendors!A2:C3"
    assert "Harbor Packaging" in found[0].page_content
    assert "(Vendors!A2:C3)" in retrieval_service.format_context(found)
