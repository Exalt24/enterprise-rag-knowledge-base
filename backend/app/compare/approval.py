"""The human approval checkpoint.

A QA report does not clear a package by itself. Every finding that is both blocking and not met needs a reviewer's decision (`accept` it as a real problem to fix, `reject` it as
a false alarm, or `waive` it with a reason) before the package is approved. The approved file records who decided, what they decided and why, and it is bound to the report by
the report's hash: change one finding, one decision or the reviewer's name afterwards and `verify_approved` says which part was touched.

Rules, each pinned by a test:
  * a decision names a finding that exists and is not `met` (a decision about a clean finding is noise and is refused);
  * `reject` and `waive` carry a reason of at least 10 characters (an `accept` needs none: it agrees with the tool);
  * one decision per finding;
  * every open blocking finding is decided, and the refusal lists the ones that are not;
  * the reviewer is a name, not an empty string.
`approved_at` is passed in (it defaults to the clock at the call) and sits outside the report's hash, so the report stays reproducible.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .pages import CompareError
from .report import Report, body_hash, canonical, report_from_dict

DECISIONS = ("accept", "reject", "waive")
MIN_REASON_CHARS = 10


def _approval_hash(report_sha256: str, reviewer: str, decisions: list[dict[str, str]]) -> str:
    return hashlib.sha256(canonical({"report_sha256": report_sha256, "reviewer": reviewer, "decisions": decisions})).hexdigest()


def approve(report: Report, reviewer: Any, decisions: Any, approved_at: str | None = None) -> dict[str, Any]:
    """Approve a report with a reviewer's decisions. Raises CompareError, naming every problem at once, when the checkpoint is not passed."""
    problems: list[str] = []
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2:
        problems.append("the reviewer's name is missing")
    if not isinstance(decisions, list):
        raise CompareError("the decisions must be a list")
    by_id = {f.id: f for f in report.findings}
    clean: list[dict[str, str]] = []
    decided: set[str] = set()
    for position, d in enumerate(decisions, start=1):
        if not isinstance(d, dict) or not isinstance(d.get("finding"), str):
            problems.append("decision number %d names no finding" % position)
            continue
        fid = d["finding"]
        finding = by_id.get(fid)
        if finding is None:
            problems.append("decision for %s: there is no such finding" % fid)
            continue
        if fid in decided:
            problems.append("decision for %s: decided twice" % fid)
            continue
        decided.add(fid)
        if finding.status == "met":
            problems.append("decision for %s: the finding is met, there is nothing to decide" % fid)
            continue
        kind, reason = d.get("decision"), d.get("reason", "")
        if kind not in DECISIONS:
            problems.append("decision for %s: decision must be accept, reject or waive" % fid)
            continue
        if kind != "accept" and (not isinstance(reason, str) or len(reason.strip()) < MIN_REASON_CHARS):
            problems.append("decision for %s: a %s needs a reason of at least %d characters" % (fid, kind, MIN_REASON_CHARS))
            continue
        clean.append({"finding": fid, "decision": kind, "reason": reason.strip() if isinstance(reason, str) else ""})
    open_ids = [f.id for f in report.findings if f.open_blocking and f.id not in decided]
    if open_ids:
        problems.append("blocking findings with no decision: %s" % ", ".join(open_ids))
    if problems:
        raise CompareError("not approved: " + "; ".join(problems))
    reviewer_name = reviewer.strip()
    clean.sort(key=lambda d: d["finding"])
    return {"report": report.to_dict(), "reviewer": reviewer_name, "decisions": clean,
            "approved_at": approved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approval_sha256": _approval_hash(report.sha256, reviewer_name, clean)}


def verify_approved(raw: Any) -> Report:
    """Check an approved file: the report matches its hash, the approval matches the report, and the checkpoint is still passed. Returns the report."""
    if not isinstance(raw, dict):
        raise CompareError("the approved file is not an approval this tool wrote")
    report = report_from_dict(raw.get("report"))
    reviewer, decisions = raw.get("reviewer"), raw.get("decisions")
    if not isinstance(reviewer, str) or not isinstance(decisions, list):
        raise CompareError("the approved file is not an approval this tool wrote")
    if raw.get("approval_sha256") != _approval_hash(report.sha256, reviewer, decisions):
        raise CompareError("the approval does not match the report and decisions it names: something was changed after approval")
    approve(report, reviewer, decisions, raw.get("approved_at"))
    return report


def load_json(path: str) -> Any:
    try:
        with open(path, "rb") as handle:
            return json.loads(handle.read().decode("utf-8"))
    except OSError as exc:
        raise CompareError("cannot open %s: %s" % (path, exc.strerror or "unreadable")) from None
    except (UnicodeDecodeError, ValueError):
        raise CompareError("%s is not valid JSON" % path) from None


__all__ = ["approve", "verify_approved", "load_json", "body_hash", "DECISIONS"]
