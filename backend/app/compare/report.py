"""The structured QA report: findings with source references, a content hash, and a plain text rendering.

A finding is one requirement's result: `met`, `discrepancy` (the documents disagree, or break a limit, or hold something forbidden), `missing` (a required item is in no readable
page) or `unclear` (the tool cannot say: a scan, a model that could not cite its words, two figures in one document). Every finding that is not `met` carries evidence, and
every piece of evidence names the document, the page and the words copied from that page.

The report is hashed. The hash covers the documents' own SHA-256, the requirements file's SHA-256 and every finding, and nothing that changes between runs (no clock), so the
same two files and requirements always give the same hash, and an edit to a finding changes it. The approval step binds a reviewer's decision to that hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

STATUSES = ("met", "discrepancy", "missing", "unclear")
REPORT_VERSION = 1


@dataclass(frozen=True)
class Evidence:
    document: str       # "a" or "b"
    page: int
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {"document": self.document, "page": self.page, "quote": self.quote}


@dataclass(frozen=True)
class Finding:
    id: str
    title: str
    severity: str
    status: str
    reason: str
    evidence: tuple[Evidence, ...] = ()

    @property
    def open_blocking(self) -> bool:
        return self.severity == "blocking" and self.status != "met"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "severity": self.severity, "status": self.status, "reason": self.reason,
                "evidence": [e.to_dict() for e in self.evidence]}


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    sha256: str
    pages: int
    unreadable_pages: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "pages": self.pages, "unreadable_pages": list(self.unreadable_pages)}


@dataclass(frozen=True)
class Report:
    documents: dict[str, DocumentInfo]
    requirements_sha256: str
    findings: tuple[Finding, ...]
    judge: str | None = None
    sha256: str = field(default="")

    def body(self) -> dict[str, Any]:
        """Everything the hash covers."""
        return {"version": REPORT_VERSION, "documents": {k: v.to_dict() for k, v in sorted(self.documents.items())},
                "requirements_sha256": self.requirements_sha256, "judge": self.judge, "findings": [f.to_dict() for f in self.findings]}

    def to_dict(self) -> dict[str, Any]:
        counts = {s: sum(1 for f in self.findings if f.status == s) for s in STATUSES}
        return {**self.body(), "summary": {**counts, "open_blocking": [f.id for f in self.findings if f.open_blocking]}, "report_sha256": self.sha256}


def canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def body_hash(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(body)).hexdigest()


def seal(report: Report) -> Report:
    """The same report with its hash filled in."""
    return Report(report.documents, report.requirements_sha256, report.findings, report.judge, body_hash(report.body()))


def render_text(report: Report) -> str:
    """The report as a person reads it: the verdict per requirement, then the page and the words behind each one."""
    out = ["QA REPORT %s" % report.sha256[:12], ""]
    for key in sorted(report.documents):
        d = report.documents[key]
        note = "" if not d.unreadable_pages else ", no text on page %s (not read)" % ", ".join(str(p) for p in d.unreadable_pages)
        out.append("Document %s: %s (%d pages%s)" % (key.upper(), d.name, d.pages, note))
    out.append("")
    for f in report.findings:
        out.append("%s  [%s] %s: %s" % (f.id, f.severity, f.status.upper(), f.title))
        if f.status != "met":
            out.append("    %s" % f.reason)
        for e in f.evidence:
            out.append("    %s p.%d: \"%s\"" % (e.document.upper(), e.page, e.quote))
    summary = report.to_dict()["summary"]
    out += ["", "met %d, discrepancy %d, missing %d, unclear %d; open blocking: %s" % (
        summary["met"], summary["discrepancy"], summary["missing"], summary["unclear"], ", ".join(summary["open_blocking"]) or "none")]
    return "\n".join(out) + "\n"


def report_from_dict(raw: Any) -> Report:
    """Rebuild a Report from its JSON form, and refuse one whose hash does not match its content."""
    from .pages import CompareError

    try:
        docs = {k: DocumentInfo(v["name"], v["sha256"], int(v["pages"]), tuple(int(p) for p in v["unreadable_pages"])) for k, v in raw["documents"].items()}
        findings = tuple(Finding(f["id"], f["title"], f["severity"], f["status"], f["reason"],
                                 tuple(Evidence(e["document"], int(e["page"]), e["quote"]) for e in f["evidence"])) for f in raw["findings"])
        report = Report(docs, raw["requirements_sha256"], findings, raw.get("judge"), raw["report_sha256"])
    except (KeyError, TypeError, ValueError, AttributeError):
        raise CompareError("the report file is not a report this tool wrote") from None
    if body_hash(report.body()) != report.sha256:
        raise CompareError("the report's hash does not match its content: it was changed after it was written")
    return report
