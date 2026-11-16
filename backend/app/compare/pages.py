"""Read a PDF into numbered pages, and say which pages gave no text.

A requirement check is only as honest as the text it reads. Two things matter more here than in a search service:

  * A page with no text layer (a scan, a drawing saved as a picture) is NOT an empty page. `unreadable` lists it, so a later "this phrase is missing" can be turned into
    "this phrase is missing from the pages that could be read, and page 3 was not". There is no OCR in this module; a person is told, never a silent skip.
  * A file that cannot be read at all (encrypted, corrupt, empty, too large, not a PDF) is refused with the reason, never read as "no text".

Page numbers are the PDF's own, counted from 1, because a QA report is read by a person who opens the page.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

#: The most pages and bytes one document may hold. Bounds the work one request can ask for; a larger file is refused, not cut short.
MAX_PAGES = 500
MAX_BYTES = 50 * 1024 * 1024


class CompareError(Exception):
    """A document or a requirements file that cannot be used, with the reason a person can act on."""


@dataclass(frozen=True)
class PageText:
    number: int
    text: str


@dataclass(frozen=True)
class PdfText:
    name: str
    sha256: str
    pages: tuple[PageText, ...]
    #: Page numbers with no extractable text (a scan, or a picture of a drawing).
    unreadable: tuple[int, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, number: int) -> PageText | None:
        return self.pages[number - 1] if 1 <= number <= len(self.pages) else None


def read_pdf(source: str | Path | bytes, name: str | None = None) -> PdfText:
    """Read a PDF from a path or from bytes. Raises CompareError for anything that is not a readable, unencrypted PDF."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CompareError("cannot open %s: %s" % (path.name, exc.strerror or "unreadable")) from None
        label = name or path.name
    else:
        data, label = bytes(source), name or "document.pdf"
    if not data:
        raise CompareError("%s is empty" % label)
    if len(data) > MAX_BYTES:
        raise CompareError("%s is larger than %d MB, which is the most one document may be" % (label, MAX_BYTES // (1024 * 1024)))
    if not data.lstrip()[:5] == b"%PDF-":
        raise CompareError("%s is not a PDF file" % label)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CompareError("%s is password protected, and this tool does not open protected files" % label)
        count = len(reader.pages)
        if count == 0:
            raise CompareError("%s has no pages" % label)
        if count > MAX_PAGES:
            raise CompareError("%s has %d pages, more than the %d one document may hold" % (label, count, MAX_PAGES))
        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            pages.append(PageText(index, page.extract_text() or ""))
    except CompareError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, AttributeError, OSError, RecursionError, NotImplementedError):
        raise CompareError("%s could not be read as a PDF (it may be damaged)" % label) from None
    unreadable = tuple(p.number for p in pages if not p.text.strip())
    return PdfText(label, hashlib.sha256(data).hexdigest(), tuple(pages), unreadable)
