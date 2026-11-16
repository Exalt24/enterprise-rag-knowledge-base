"""Check two documents against the requirements, and say what is wrong with page and quote.

The rules that decide a status, in the order they bite:

  * A figure is read ONLY as a plain number: `1,200.5` and `1200.5` are the same, `1.200,5` and `1,5` are not understood and make the finding `unclear`. A guess here would
    turn a wrong figure into a pass.
  * One document holding two different figures for the same thing (page 2 says 45 m/s, page 6 says 40 m/s) is `unclear`, never a pick of one: the author has to settle it.
  * A requirement whose item is in no readable page is `missing` only when every page was readable. If a page had no text layer the finding is `unclear`, because the item may
    be on the page nobody read.
  * A judged statement is `met` or `discrepancy` only when the model cites a page and quotes words that are actually on that page; anything else is `unclear`.
  * A missing judge is `unclear`, never `met`.

The text of a page is read with its whitespace collapsed, so a phrase broken over two lines is still found; case is ignored; a quote is a window of the collapsed text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, Sequence

from .pages import PageText, PdfText
from .report import DocumentInfo, Evidence, Finding, Report, seal
from .requirements import Requirement

QUOTE_CONTEXT = 50
MAX_QUOTE_CHARS = 200
_NUMBER = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
_SPACE = re.compile(r"\s+")
#: U+00AD, a hyphen that a PDF puts inside a word at a line break and that a search must not see.
SOFT_HYPHEN = chr(0xAD)


class Judge(Protocol):
    """Anything that can judge a statement against pages. `judge` returns {"verdict": met|not_met|unclear, "page": int, "quote": str}."""

    name: str

    def judge(self, statement: str, pages: Sequence[PageText]) -> dict: ...


def collapse(text: str) -> str:
    return _SPACE.sub(" ", text.replace(SOFT_HYPHEN, "")).strip()


def parse_number(text: str) -> Decimal | None:
    """A plain number with comma thousands and a point decimal, else None."""
    cleaned = text.strip()
    if not _NUMBER.match(cleaned):
        return None
    return Decimal(cleaned.replace(",", ""))


def _window(text: str, start: int, end: int) -> str:
    lo, hi = max(0, start - QUOTE_CONTEXT), min(len(text), end + QUOTE_CONTEXT)
    quote = text[lo:hi].strip()
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[:MAX_QUOTE_CHARS].rstrip()
    return ("..." if lo > 0 else "") + quote + ("..." if hi < len(text) else "")


@dataclass(frozen=True)
class Hit:
    document: str
    page: int
    quote: str
    value: Decimal | None
    unit: str | None


def _find_phrase(doc: PdfText, key: str, phrase: str) -> list[Evidence]:
    rx = re.compile(re.escape(collapse(phrase)), re.IGNORECASE)
    found = []
    for page in doc.pages:
        text = collapse(page.text)
        m = rx.search(text)
        if m:
            found.append(Evidence(key, page.number, _window(text, m.start(), m.end())))
    return found


def _find_values(doc: PdfText, key: str, pattern: re.Pattern[str]) -> list[Hit]:
    hits = []
    for page in doc.pages:
        text = collapse(page.text)
        for m in pattern.finditer(text):
            unit = m.group("unit") if "unit" in pattern.groupindex else None
            hits.append(Hit(key, page.number, _window(text, m.start(), m.end()), parse_number(m.group("value")),
                            collapse(unit).casefold() if unit and unit.strip() else None))
    return hits


def _absent(req: Requirement, docs: dict[str, PdfText], keys: Sequence[str], what: str) -> Finding:
    """The item is in no readable page of the documents in `keys`: missing, or unclear when a page could not be read."""
    unread = [k for k in keys if docs[k].unreadable]
    if unread:
        pages = "; ".join("document %s page %s" % (k.upper(), ", ".join(str(p) for p in docs[k].unreadable)) for k in unread)
        return Finding(req.id, req.title, req.severity, "unclear", "%s was not found, but there is no text to read on %s, so it may be there" % (what, pages))
    return Finding(req.id, req.title, req.severity, "missing", "%s is not in %s" % (what, " or ".join("document %s" % k.upper() for k in keys)))


def _keys(req: Requirement) -> list[str]:
    return ["a", "b"] if req.document == "both" else [req.document]


def _check_present(req: Requirement, docs: dict[str, PdfText]) -> Finding:
    evidence: list[Evidence] = []
    for key in _keys(req):
        found = _find_phrase(docs[key], key, req.phrase or "")
        if not found:
            return _absent(req, docs, [key], '"%s"' % req.phrase)
        evidence.append(found[0])
    return Finding(req.id, req.title, req.severity, "met", "found", tuple(evidence))


def _check_forbidden(req: Requirement, docs: dict[str, PdfText]) -> Finding:
    evidence: list[Evidence] = []
    for key in _keys(req):
        evidence += _find_phrase(docs[key], key, req.phrase or "")
    if evidence:
        return Finding(req.id, req.title, req.severity, "discrepancy", '"%s" must not appear and does' % req.phrase, tuple(evidence))
    unread = [k for k in _keys(req) if docs[k].unreadable]
    if unread:
        return Finding(req.id, req.title, req.severity, "unclear", '"%s" was not found, but %s has pages with no text to read' % (req.phrase, " and ".join("document %s" % k.upper() for k in unread)))
    return Finding(req.id, req.title, req.severity, "met", "not found")


def _settle(req: Requirement, docs: dict[str, PdfText], key: str, hits: list[Hit]) -> tuple[Hit | None, Finding | None]:
    """The one figure a document gives, or the finding that says why it gives none."""
    if not hits:
        return None, _absent(req, docs, [key], "the figure")
    bad = [h for h in hits if h.value is None]
    if bad:
        return None, Finding(req.id, req.title, req.severity, "unclear", "document %s page %d has a figure that is not a plain number (1,200.5 and 1200.5 are read, 1.200,5 is not)" % (key.upper(), bad[0].page),
                             tuple(Evidence(key, h.page, h.quote) for h in bad[:3]))
    distinct = {(h.value, h.unit) for h in hits}
    if len(distinct) > 1:
        first = {}
        for h in hits:
            first.setdefault((h.value, h.unit), h)
        return None, Finding(req.id, req.title, req.severity, "unclear", "document %s gives more than one figure (pages %s), so it disagrees with itself" % (
            key.upper(), ", ".join(str(h.page) for h in first.values())), tuple(Evidence(key, h.page, h.quote) for h in first.values()))
    return hits[0], None


def _check_match(req: Requirement, docs: dict[str, PdfText]) -> Finding:
    assert req.pattern is not None
    chosen: dict[str, Hit] = {}
    for key in ("a", "b"):
        hit, problem = _settle(req, docs, key, _find_values(docs[key], key, req.pattern))
        if problem is not None:
            return problem
        assert hit is not None
        chosen[key] = hit
    a, b = chosen["a"], chosen["b"]
    evidence = (Evidence("a", a.page, a.quote), Evidence("b", b.page, b.quote))
    if a.unit != b.unit:
        return Finding(req.id, req.title, req.severity, "discrepancy", "the units differ: %s says %s and %s says %s" % ("document A", a.unit or "nothing", "document B", b.unit or "nothing"), evidence)
    assert a.value is not None and b.value is not None
    if abs(a.value - b.value) > req.tolerance:
        return Finding(req.id, req.title, req.severity, "discrepancy", "document A says %s and document B says %s%s" % (
            a.value, b.value, "" if req.tolerance == 0 else " (tolerance %s)" % req.tolerance), evidence)
    return Finding(req.id, req.title, req.severity, "met", "both documents say %s" % a.value, evidence)


def _check_limit(req: Requirement, docs: dict[str, PdfText]) -> Finding:
    assert req.pattern is not None
    key = req.document
    hit, problem = _settle(req, docs, key, _find_values(docs[key], key, req.pattern))
    if problem is not None:
        return problem
    assert hit is not None and hit.value is not None
    evidence = (Evidence(key, hit.page, hit.quote),)
    if req.unit is not None:
        if hit.unit is None:
            return Finding(req.id, req.title, req.severity, "unclear", "the figure has no unit and the requirement is in %s" % req.unit, evidence)
        if hit.unit != req.unit.casefold():
            return Finding(req.id, req.title, req.severity, "discrepancy", "the figure is in %s and the requirement is in %s" % (hit.unit, req.unit), evidence)
    if req.minimum is not None and hit.value < req.minimum:
        return Finding(req.id, req.title, req.severity, "discrepancy", "%s is below the minimum %s" % (hit.value, req.minimum), evidence)
    if req.maximum is not None and hit.value > req.maximum:
        return Finding(req.id, req.title, req.severity, "discrepancy", "%s is above the maximum %s" % (hit.value, req.maximum), evidence)
    return Finding(req.id, req.title, req.severity, "met", "%s is within the limit" % hit.value, evidence)


def _check_judge(req: Requirement, docs: dict[str, PdfText], judge: Judge | None) -> Finding:
    key = req.document
    doc = docs[key]
    if judge is None:
        return Finding(req.id, req.title, req.severity, "unclear", "no judge is configured, so a statement in words is not checked")
    readable = [p for p in doc.pages if p.text.strip()]
    if not readable:
        return Finding(req.id, req.title, req.severity, "unclear", "document %s has no readable page" % key.upper())
    try:
        answer = judge.judge(req.text or "", readable)
    except Exception as exc:  # the judge is a network call in production: its failure is a finding, never a crash or a pass
        return Finding(req.id, req.title, req.severity, "unclear", "the judge failed (%s)" % type(exc).__name__)
    verdict = answer.get("verdict") if isinstance(answer, dict) else None
    if verdict == "unclear":
        return Finding(req.id, req.title, req.severity, "unclear", "the judge could not decide")
    if verdict not in ("met", "not_met"):
        return Finding(req.id, req.title, req.severity, "unclear", "the judge's answer was not met, not_met or unclear")
    page_no, quote = answer.get("page"), answer.get("quote")
    page = doc.page(page_no) if isinstance(page_no, int) and not isinstance(page_no, bool) else None
    if page is None or not isinstance(quote, str) or not collapse(quote):
        return Finding(req.id, req.title, req.severity, "unclear", "the judge named no page and words, so its answer is not accepted")
    text = collapse(page.text)
    where = text.casefold().find(collapse(quote).casefold())
    if where < 0:
        return Finding(req.id, req.title, req.severity, "unclear", "the words the judge quoted are not on page %d, so its answer is not accepted" % page.number)
    evidence = (Evidence(key, page.number, _window(text, where, where + len(collapse(quote)))),)
    if verdict == "met":
        return Finding(req.id, req.title, req.severity, "met", "the judge found it and the quote is on the page", evidence)
    return Finding(req.id, req.title, req.severity, "discrepancy", "the judge found the statement is not met and quoted the page", evidence)


def compare(a: PdfText, b: PdfText, requirements: Sequence[Requirement], requirements_sha256: str, judge: Judge | None = None) -> Report:
    """Check document A and document B against every requirement, in the order given."""
    docs = {"a": a, "b": b}
    findings: list[Finding] = []
    for req in requirements:
        if req.kind == "present":
            findings.append(_check_present(req, docs))
        elif req.kind == "forbidden":
            findings.append(_check_forbidden(req, docs))
        elif req.kind == "match":
            findings.append(_check_match(req, docs))
        elif req.kind == "limit":
            findings.append(_check_limit(req, docs))
        else:
            findings.append(_check_judge(req, docs, judge))
    info = {k: DocumentInfo(d.name, d.sha256, d.page_count, d.unreadable) for k, d in docs.items()}
    return seal(Report(info, requirements_sha256, tuple(findings), getattr(judge, "name", None) if judge is not None else None))
