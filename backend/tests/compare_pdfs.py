# LIFESPAN: permanent (the test PDF writer for the compare package and the sample documents under samples/compare)
"""A small, deterministic PDF writer for the compare tests and for the sample documents.

Each page is a list of text lines, or None for a page with NO text layer (a rectangle is drawn instead, which is what a scanned page looks like to a text extractor). The same input
always gives the same bytes: no clock, no id, no producer string, so a sample PDF can be committed and a test can check the committed bytes are what this writes.

Only ASCII text is written (Helvetica, WinAnsi), which is all the tests need.
"""
from __future__ import annotations


def _escape(line: str) -> str:
    line.encode("ascii")
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str] | None]) -> bytes:
    if not pages:
        raise ValueError("a PDF needs at least one page")
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")            # 1, filled in once the page tree number is known
    tree = add(b"")               # 2
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")   # 3
    kids = []
    for lines in pages:
        if lines is None:
            stream = b"0.5 w 50 50 300 120 re S"
        else:
            body = ["BT /F1 11 Tf 56 760 Td 15 TL"]
            for line in lines:
                body.append("(%s) Tj T*" % _escape(line))
            body.append("ET")
            stream = "\n".join(body).encode("ascii")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page = add(("<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (tree, font, content)).encode("ascii"))
        kids.append(page)
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % tree
    objects[tree - 1] = ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join("%d 0 R" % k for k in kids), len(kids))).encode("ascii")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1) + b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, catalog, xref)
    return bytes(out)
