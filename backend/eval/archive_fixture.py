"""Generated data room archives for the ingestion tests and the measurements in docs/ingestion.md.

Everything here is invented text. The text of a file depends only on its number and the seed. Word and Excel files
carry a creation time and the locked PDF a random salt, so two builds differ in those bytes; a rerun means submitting the
same zip file again.

    python eval/archive_fixture.py out.zip --files 300       a mixed archive: txt, md, docx, xlsx, pdf and a few bad entries
    python eval/archive_fixture.py out.zip --files 3000 --plain   3000 small text files, nothing else

The mixed archive also holds, on purpose: a corrupt PDF, a password protected PDF, a nested zip, a .png, two exact
copies of other files, and one workbook that is not a workbook. docs/ingestion.md says what each one must come out as.
"""
import argparse
import io
import random
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from docx import Document as DocxDocument
from openpyxl import Workbook
from pypdf import PdfReader, PdfWriter

FIXED_TIME = (2026, 1, 1, 0, 0, 0)
TOPICS = [
    ("Site lease", "The lessor grants the lessee the right to use parcel {n} for the term of {a} years, with an option to extend by {b} years if notice is given {c} days before expiry."),
    ("Interconnection study", "The study found that feeder {n} can accept {a} megawatts before a network upgrade is needed, and the upgrade is estimated at {b} thousand dollars."),
    ("Permit record", "Permit {n} was filed on day {a} and approved on day {b}, subject to {c} conditions that the owner must meet before construction starts."),
    ("Equipment schedule", "Order {n} lists {a} units delivered in {b} batches, with the last batch due {c} days after the notice to proceed."),
    ("Inspection note", "Inspection {n} covered {a} items. {b} items passed, and the remaining items were returned to the contractor with a {c} day deadline."),
    ("Payment terms", "Invoice {n} is due {a} days after receipt. A late payment adds {b} percent a month, and disputes must be raised within {c} days."),
    ("Warranty clause", "The warranty on item {n} runs {a} years from commissioning. A defect reported in the first {b} months is repaired at no charge, and later ones within {c} days."),
    ("Meeting minutes", "Meeting {n} agreed that {a} open actions would close by the next review. {b} people attended and the next meeting is in {c} weeks."),
]
CLOSING = [
    "This document is a generated sample and describes no real project.",
    "Any figure in this text was invented for the test and means nothing outside it.",
]


def _paragraphs(number: int, seed: int, count: int) -> List[str]:
    rng = random.Random(seed * 100003 + number)
    title, template = TOPICS[number % len(TOPICS)]
    out = [f"{title} {number:04d}"]
    for i in range(count):
        out.append(template.format(n=f"{number}-{i}", a=rng.randint(2, 40), b=rng.randint(1, 30), c=rng.randint(5, 120)))
    out.append(CLOSING[number % len(CLOSING)])
    return out


def make_text(number: int, seed: int = 1, paragraphs: int = 8) -> str:
    return "\n\n".join(_paragraphs(number, seed, paragraphs)) + "\n"


def make_docx(number: int, seed: int = 1) -> bytes:
    doc = DocxDocument()
    for text in _paragraphs(number, seed, 6):
        doc.add_paragraph(text)
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def make_xlsx(number: int, seed: int = 1) -> bytes:
    rng = random.Random(seed * 7919 + number)
    wb = Workbook()
    ws = wb.active
    ws.title = "Schedule"
    ws.append(["Item", "Quantity", "Unit cost", "Due day"])
    for i in range(30):
        ws.append([f"Item {number}-{i}", rng.randint(1, 50), rng.randint(10, 900), rng.randint(5, 200)])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def make_pdf(number: int, seed: int = 1, pages: int = 1) -> bytes:
    """A small valid PDF with `pages` pages of text, written by hand so no PDF library has to create it."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for page in range(pages):
        stream = ["BT", "/F1 11 Tf", "50 760 Td", "14 TL"]
        for line in (w for t in _paragraphs(number * 100 + page, seed, 5) for w in textwrap.wrap(t, 88)):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.append(f"({escaped}) Tj T*")
        stream.append("ET")
        content = "\n".join(stream).encode("latin-1", "replace")
        page_id, content_id = len(objects) + 1, len(objects) + 2
        kids.append(f"{page_id} 0 R")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>".encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def make_encrypted_pdf(number: int, seed: int = 1) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(number, seed))))
    writer.encrypt("owner-only-password", algorithm="AES-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _tiny_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("inner.txt", "inside")
    return out.getvalue()


def _add(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data)


def mixed_entries(files: int, seed: int = 1) -> List[Tuple[str, bytes]]:
    """`files` entries: mostly readable documents spread over folders, plus the bad entries listed in the module docstring."""
    makers = [
        ("txt", lambda n: make_text(n, seed).encode()),
        ("md", lambda n: ("# " + make_text(n, seed)).encode()),
        ("pdf", lambda n: make_pdf(n, seed)),
        ("docx", lambda n: make_docx(n, seed)),
        ("xlsx", lambda n: make_xlsx(n, seed)),
    ]
    folders = ["leases", "permits", "studies", "contracts", "inspections", "finance"]
    bad = 7  # the corrupt pdf, locked pdf, fake workbook, nested zip and png, plus the two copies below
    good = max(files - bad, 0)
    entries: List[Tuple[str, bytes]] = []
    for n in range(good):
        ext, build = makers[n % len(makers)]
        entries.append((f"{folders[n % len(folders)]}/doc-{n:04d}.{ext}", build(n)))
    if good >= 2:
        entries.append(("copies/doc-0000-copy.txt", entries[0][1] if entries[0][0].endswith("txt") else make_text(0, seed).encode()))
        entries.append(("copies/doc-0005-copy.txt", entries[5][1] if len(entries) > 5 and entries[5][0].endswith("txt") else make_text(5, seed).encode()))
    entries.append(("damaged/corrupt-report.pdf", b"%PDF-1.4\nthis file was cut short and is not a pdf any reader can open"))
    entries.append(("damaged/locked-report.pdf", make_encrypted_pdf(9001, seed)))
    entries.append(("damaged/not-a-workbook.xlsx", b"not a workbook, just text"))
    entries.append(("damaged/nested.zip", _tiny_zip()))
    entries.append(("images/site-photo.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64))
    return entries[:files] if len(entries) > files else entries


def plain_entries(files: int, seed: int = 1) -> List[Tuple[str, bytes]]:
    return [(f"bulk/part-{n // 500:02d}/note-{n:05d}.txt", make_text(n, seed, paragraphs=3).encode()) for n in range(files)]


def build_archive(path, files: int = 300, seed: int = 1, plain: bool = False) -> Dict[str, int]:
    entries = plain_entries(files, seed) if plain else mixed_entries(files, seed)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            _add(archive, name, data)
    return {"entries": len(entries), "bytes": Path(path).stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--files", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plain", action="store_true", help="only small text files")
    args = parser.parse_args()
    print(build_archive(args.out, args.files, args.seed, args.plain))
