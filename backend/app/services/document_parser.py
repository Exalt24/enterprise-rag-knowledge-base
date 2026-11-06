"""
Document Parser Service

Extracts text from various document formats:
- PDF (using pypdf)
- DOCX (using python-docx)
- XLSX (using openpyxl, formulas read as the values Excel last saved)
- TXT, MD (plain text)
- OCR support for scanned PDFs (local only)

Returns standardized Document objects with rich metadata.

"""

from typing import List, Optional
from pathlib import Path
from datetime import date, datetime
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document as DocxDocument
from langchain_core.documents import Document
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import os
import re
import zipfile

# OCR support (local only - Render has 512MB RAM limit)
try:
    if not os.getenv("RENDER"):
        import pytesseract
        from PIL import Image
        import pdf2image

        OCR_AVAILABLE = True
        print("[i] OCR support enabled (local environment)")
    else:
        OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False
    if not os.getenv("RENDER"):
        print("[!] OCR unavailable (install: pip install pytesseract pdf2image)")


class DocumentParser:
    """
    Parse documents and extract text content with rich metadata.

    Supports: PDF, DOCX, XLSX, TXT, MD
    Returns: LangChain Document objects with metadata

    Features:
    - Configurable PDF splitting (per-page or combined)
    - OCR for scanned/image PDFs (local only)
    - Rich metadata (word count, upload date, file size)
    - Specific error handling
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".md"}

    # Workbook limits. A workbook over any of them is refused with a clear error, never cut short without saying so.
    # The block size sits under the chunker's 500 characters so a block is stored as one chunk with one cell range.
    XLSX_MAX_SHEETS = 20
    XLSX_MAX_ROWS = 20000  # rows read across the workbook, empty ones included
    XLSX_MAX_COLUMNS = 50
    XLSX_MAX_UNCOMPRESSED_MB = 50  # the XML inside the zip, checked before it is opened (zip bombs)
    XLSX_CELL_CHARS = 200
    XLSX_BLOCK_CHARS = 480  # the whole block text, its sheet, columns and range lines included
    XLSX_BLOCK_ROWS = 25
    _OLE_MAGIC = bytes.fromhex("d0cf11e0a1b11ae1")  # an encrypted .xlsx is stored in this container, not a zip

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file format is supported"""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def parse(
        cls, file_path: str, split_by_page: bool = True, use_ocr: bool = False
    ) -> List[Document]:
        """
        Parse document and return LangChain Document objects.

        Args:
            file_path: Path to document file
            split_by_page: For PDFs, split into one doc per page (True) or combine (False)
            use_ocr: Use OCR for image-only pages (local only, requires tesseract)

        Returns:
            List of Document objects

        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not cls.is_supported(file_path):
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        # Route to appropriate parser
        extension = path.suffix.lower()

        if extension == ".pdf":
            return cls._parse_pdf(path, split_by_page, use_ocr)
        elif extension == ".docx":
            return cls._parse_docx(path)
        elif extension == ".xlsx":
            return cls._parse_xlsx(path)
        elif extension in {".txt", ".md"}:
            return cls._parse_text(path)

        raise ValueError(f"No parser for {extension}")

    @classmethod
    def _parse_pdf(
        cls, path: Path, split_by_page: bool = True, use_ocr: bool = False
    ) -> List[Document]:
        """
        Parse PDF file with flexible splitting and optional OCR.

        Args:
            split_by_page: True = one doc per page, False = combine all pages
            use_ocr: Try OCR on empty pages (local only)

        Returns:
            List of Document objects
        """
        try:
            reader = PdfReader(str(path))
            file_stat = path.stat()
            upload_date = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            if split_by_page:
                documents = []

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    # Try OCR if page is empty and OCR available
                    if not text.strip() and use_ocr and OCR_AVAILABLE:
                        text = cls._ocr_pdf_page(path, page_num)
                        if text:
                            print(f"[i] OCR extracted text from page {page_num + 1}")

                    # Skip if still empty
                    if not text.strip():
                        continue

                    # Create Document with RICH metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "file_type": "pdf",
                            "page": page_num + 1,
                            "total_pages": len(reader.pages),
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "upload_date": upload_date,
                            "file_size_kb": round(file_stat.st_size / 1024, 2),
                        },
                    )
                    documents.append(doc)

                return documents

            else:
                all_text = []
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()

                    # Try OCR if empty
                    if not text.strip() and use_ocr and OCR_AVAILABLE:
                        text = cls._ocr_pdf_page(path, page_num)

                    if text.strip():
                        all_text.append(text)

                combined_text = "\n\n".join(all_text)

                return [
                    Document(
                        page_content=combined_text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "file_type": "pdf",
                            "total_pages": len(reader.pages),
                            "char_count": len(combined_text),
                            "word_count": len(combined_text.split()),
                            "upload_date": upload_date,
                            "file_size_kb": round(file_stat.st_size / 1024, 2),
                            "combined": True,
                        },
                    )
                ]

        except PdfReadError as e:
            raise ValueError(f"PDF corrupted or encrypted: {path.name} - {e}")
        except PermissionError:
            raise ValueError(f"Permission denied: {path.name}")
        except Exception as e:
            raise ValueError(f"Failed to parse PDF {path.name}: {e}")

    @classmethod
    def _ocr_pdf_page(cls, path: Path, page_num: int) -> str:
        """
        OCR a single PDF page (local only).

        Returns extracted text or empty string if OCR fails.
        """
        if not OCR_AVAILABLE:
            return ""

        try:
            # Convert PDF page to image
            images = pdf2image.convert_from_path(
                str(path), first_page=page_num + 1, last_page=page_num + 1
            )

            if not images:
                return ""

            # OCR the image
            text = pytesseract.image_to_string(images[0])
            return text.strip()

        except Exception as e:
            print(f"[!] OCR failed for page {page_num + 1}: {e}")
            return ""

    @classmethod
    def _parse_docx(cls, path: Path) -> List[Document]:
        """
        Parse DOCX file.

        Returns single Document with all content and rich metadata.
        """
        try:
            doc = DocxDocument(str(path))

            # Extract all paragraphs
            text = "\n".join([para.text for para in doc.paragraphs])

            if not text.strip():
                raise ValueError(f"DOCX file is empty: {path.name}")

            # Rich metadata
            file_stat = path.stat()

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "file_type": "docx",
                        "paragraphs": len(doc.paragraphs),
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "upload_date": datetime.fromtimestamp(
                            file_stat.st_mtime
                        ).isoformat(),
                        "file_size_kb": round(file_stat.st_size / 1024, 2),
                    },
                )
            ]

        except Exception as e:
            raise ValueError(f"Failed to parse DOCX {path.name}: {e}")

    @classmethod
    def _xlsx_cell_text(cls, value) -> str:
        """One cell as short text; an empty cell is an empty string."""
        if value is None:
            return ""
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, datetime):
            if value.time() == datetime.min.time():
                return value.date().isoformat()
            return value.isoformat(sep=" ")
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        text = " ".join(str(value).split())
        if len(text) > cls.XLSX_CELL_CHARS:
            text = text[: cls.XLSX_CELL_CHARS - 3] + "..."
        return text

    @classmethod
    def _xlsx_check_container(cls, path: Path) -> None:
        """Refuse a file that is not a readable, unprotected workbook before openpyxl sees it."""
        with open(path, "rb") as f:
            head = f.read(len(cls._OLE_MAGIC))
        if head == cls._OLE_MAGIC:
            raise ValueError(
                f"{path.name} is password protected (or is an old .xls file renamed). "
                "Remove the password and save it as .xlsx, then upload it again."
            )
        if not zipfile.is_zipfile(path):
            raise ValueError(
                f"{path.name} is not a valid .xlsx workbook (it is damaged or is a different kind of file)."
            )
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise ValueError(f"{path.name} is password protected. Remove the password and upload it again.")
            if sum(entry.file_size for entry in entries) > cls.XLSX_MAX_UNCOMPRESSED_MB * 1024 * 1024:
                raise ValueError(
                    f"{path.name} unpacks to more than {cls.XLSX_MAX_UNCOMPRESSED_MB} MB, which is over the limit."
                )

    @classmethod
    def _xlsx_preamble_chars(cls, sheet: str, header: List[str]) -> int:
        """Characters the sheet, columns and range lines add to a block (the range line is counted at its widest)."""
        columns = len("Columns: " + " | ".join(h for h in header if h)) + 1 if header else 0
        return len(f"Sheet: {sheet}") + 1 + columns + 45

    @classmethod
    def _xlsx_block(cls, path: Path, file_stat, sheet: str, header: List[str], block: list) -> Document:
        """One block of rows as one Document that carries its sheet and cell range."""
        first_col = min(b[2] for b in block)
        last_col = max(b[3] for b in block)
        first_row, last_row = block[0][0], block[-1][0]
        cell_range = f"{get_column_letter(first_col)}{first_row}:{get_column_letter(last_col)}{last_row}"
        quoted = sheet if re.fullmatch(r"\w+", sheet) else "'" + sheet.replace("'", "''") + "'"
        lines = [f"Sheet: {sheet}"]
        if header:
            lines.append("Columns: " + " | ".join(h for h in header if h))
        lines.append(f"Rows {first_row} to {last_row} ({cell_range})")
        lines.extend(b[1] for b in block)
        text = "\n".join(lines)
        return Document(
            page_content=text,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": "xlsx",
                "sheet": sheet,
                "cell_range": cell_range,
                "location": f"{quoted}!{cell_range}",
                "row_start": first_row,
                "row_end": last_row,
                "char_count": len(text),
                "word_count": len(text.split()),
                "upload_date": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "file_size_kb": round(file_stat.st_size / 1024, 2),
            },
        )

    @classmethod
    def _parse_xlsx(cls, path: Path) -> List[Document]:
        """
        Parse XLSX file.

        Each sheet becomes blocks of rows. A block is one Document holding the sheet name, the column headers
        and the row text, with the sheet and the cell range (for example Budget!A2:C26) in its metadata so a
        citation can point at the cells. Formulas are read as the value Excel last saved; a workbook that was
        never opened in Excel has none, so those cells come out empty. Hidden sheets are left out.

        Raises ValueError (with a plain message) for a damaged or password protected workbook, a workbook over
        the sheet, row, column or size limits, and a workbook with no text in it.
        """
        try:
            cls._xlsx_check_container(path)
            file_stat = path.stat()
            workbook = load_workbook(str(path), read_only=True, data_only=True)
            try:
                visible = [ws for ws in workbook.worksheets if ws.sheet_state == "visible"]
                if len(visible) > cls.XLSX_MAX_SHEETS:
                    raise ValueError(f"{path.name} has {len(visible)} sheets; the limit is {cls.XLSX_MAX_SHEETS}.")
                documents: List[Document] = []
                rows_read = 0
                for ws in visible:
                    header: Optional[List[str]] = None  # None until the first filled row has been seen
                    block: list = []  # (row number, text, first column, last column)
                    block_chars = 0  # characters of row text in the open block
                    rows = ws.iter_rows(values_only=True, max_col=cls.XLSX_MAX_COLUMNS + 1)
                    for row_number, row in enumerate(rows, start=1):
                        rows_read += 1
                        if rows_read > cls.XLSX_MAX_ROWS:
                            raise ValueError(f"{path.name} has more than {cls.XLSX_MAX_ROWS} rows, which is over the limit.")
                        cells = [cls._xlsx_cell_text(v) for v in row]
                        if len(cells) > cls.XLSX_MAX_COLUMNS and cells[cls.XLSX_MAX_COLUMNS]:
                            raise ValueError(
                                f"{path.name} sheet {ws.title} has more than {cls.XLSX_MAX_COLUMNS} columns, "
                                "which is over the limit."
                            )
                        cells = cells[: cls.XLSX_MAX_COLUMNS]
                        filled = [i for i, c in enumerate(cells) if c]
                        if not filled:
                            continue
                        if header is None:
                            # the first filled row is the header when every filled cell in it is text
                            if all(isinstance(row[i], str) for i in filled):
                                header = cells
                                continue
                            header = []
                        if header:
                            labels = [header[i] if i < len(header) and header[i] else get_column_letter(i + 1) for i in filled]
                            text = "; ".join(f"{label}: {cells[i]}" for label, i in zip(labels, filled))
                        else:
                            text = " | ".join(cells[i] for i in filled)
                        used = cls._xlsx_preamble_chars(ws.title, header) + block_chars
                        if block and (len(block) >= cls.XLSX_BLOCK_ROWS or used + len(text) > cls.XLSX_BLOCK_CHARS):
                            documents.append(cls._xlsx_block(path, file_stat, ws.title, header, block))
                            block, block_chars = [], 0
                        block.append((row_number, text, filled[0] + 1, filled[-1] + 1))
                        block_chars += len(text) + 1
                    if block:
                        documents.append(cls._xlsx_block(path, file_stat, ws.title, header or [], block))
            finally:
                workbook.close()

            if not documents:
                raise ValueError(f"XLSX file is empty: {path.name}")
            return documents

        except ValueError:
            raise
        except PermissionError:
            raise ValueError(f"Permission denied: {path.name}")
        except Exception as e:
            raise ValueError(f"Failed to parse XLSX {path.name}: {e}")

    @classmethod
    def _parse_text(cls, path: Path) -> List[Document]:
        """
        Parse plain text file (TXT, MD).

        Returns single Document with all content and rich metadata.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                raise ValueError(f"Text file is empty: {path.name}")

            # Rich metadata
            file_stat = path.stat()

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "file_type": path.suffix[1:],  # Remove the dot
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "upload_date": datetime.fromtimestamp(
                            file_stat.st_mtime
                        ).isoformat(),
                        "file_size_kb": round(file_stat.st_size / 1024, 2),
                    },
                )
            ]

        except UnicodeDecodeError:
            # Try different encoding
            try:
                with open(path, "r", encoding="latin-1") as f:
                    text = f.read()

                file_stat = path.stat()

                return [
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "file_type": path.suffix[1:],
                            "encoding": "latin-1",
                            "char_count": len(text),
                            "word_count": len(text.split()),
                            "upload_date": datetime.fromtimestamp(
                                file_stat.st_mtime
                            ).isoformat(),
                            "file_size_kb": round(file_stat.st_size / 1024, 2),
                        },
                    )
                ]
            except Exception as e:
                raise ValueError(f"Failed to parse text file {path.name}: {e}")

        except Exception as e:
            raise ValueError(f"Failed to parse text file {path.name}: {e}")


# =============================================================================
# Test Parser
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Document Parser Test (With Optimizations)")
    print("=" * 70)

    # Test with sample text file
    test_content = """
    RAG (Retrieval-Augmented Generation) is a technique for enhancing LLM responses.

    It works by:
    1. Retrieving relevant documents from a knowledge base
    2. Passing those documents as context to the LLM
    3. Generating an answer based on the retrieved context

    This improves accuracy and allows LLMs to access specific knowledge.
    """

    # Create sample file
    test_file = Path("../data/test_document.txt")
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(test_content)

    # Test parser
    print(f"\nParsing: {test_file.name}")
    print("-" * 70)

    documents = DocumentParser.parse(str(test_file))

    for doc in documents:
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print()

    # Cleanup
    test_file.unlink()

    print("=" * 70)
    print("[OK] Parser working with optimizations!")
    print("=" * 70)
    print("\nOptimizations enabled:")
    print("  ✅ Configurable PDF splitting")
    print("  ✅ Rich metadata (word count, file size, upload date)")
    print("  ✅ Specific error handling")
    print(f"  {'✅' if OCR_AVAILABLE else '⏸️'} OCR support (local only)")
