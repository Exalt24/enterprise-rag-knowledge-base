"""The predefined requirements a pair of documents is checked against.

A requirements file is JSON: `{"requirements": [ {...}, ... ]}`. Every entry has an `id` (unique), a `title` (what a reader sees), a `kind` and a `severity`. The kinds:

  present    `phrase` must appear in `document` (a, b or both).
  forbidden  `phrase` must NOT appear in `document` (a stamp such as "DRAFT", a superseded clause).
  match      `pattern` (a regular expression with a `value` group and optionally a `unit` group) is read from BOTH documents, and the two figures must agree
             (`tolerance`, a decimal, default 0).
  limit      `pattern` is read from `document`, and the figure must sit inside `min` and/or `max` (inclusive), with `unit` if one is named.
  judge      `text` is a statement in words that a model is asked to judge against `document`; the model must cite a page and quote the words. Without a judge it is
             reported as unclear, never as met.

Severity is `blocking` (the package cannot be approved while it is open), `major` or `minor`.

Everything is validated here, so a typo in a requirement is a clear error before any document is read: a pattern that is not a regular expression, a `value` group that is
missing, a `min` that is not a number, a duplicate id. A pattern is also compiled with a size limit, because a requirements file may come from anyone.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .pages import CompareError

KINDS = ("present", "forbidden", "match", "limit", "judge")
SEVERITIES = ("blocking", "major", "minor")
DOCUMENTS = ("a", "b", "both")
MAX_REQUIREMENTS = 500
MAX_PATTERN_CHARS = 300
MAX_TEXT_CHARS = 1000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


@dataclass(frozen=True)
class Requirement:
    id: str
    title: str
    kind: str
    severity: str
    document: str = "both"
    phrase: str | None = None
    pattern: re.Pattern[str] | None = None
    text: str | None = None
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    unit: str | None = None
    tolerance: Decimal = Decimal(0)


def _decimal(raw: Any, field: str, rid: str) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise CompareError("requirement %s: %s must be a number written as text or a whole number" % (rid, field))
    try:
        value = Decimal(str(raw).strip())
    except InvalidOperation:
        raise CompareError("requirement %s: %s is not a number" % (rid, field)) from None
    if not value.is_finite():
        raise CompareError("requirement %s: %s is not a finite number" % (rid, field))
    return value


def _text(entry: dict, field: str, rid: str, *, limit: int = MAX_TEXT_CHARS) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CompareError("requirement %s: %s is required and must be text" % (rid, field))
    if len(value) > limit:
        raise CompareError("requirement %s: %s is longer than %d characters" % (rid, field, limit))
    return value.strip()


def parse_requirements(raw: Any) -> list[Requirement]:
    """Validate the decoded JSON and build the requirements, in the order given."""
    entries = raw.get("requirements") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise CompareError('the requirements file must be {"requirements": [ ... ]} with at least one entry')
    if len(entries) > MAX_REQUIREMENTS:
        raise CompareError("more than %d requirements in one file" % MAX_REQUIREMENTS)
    out: list[Requirement] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise CompareError("requirement number %d is not an object" % position)
        rid = entry.get("id")
        if not isinstance(rid, str) or not _ID_RE.match(rid):
            raise CompareError("requirement number %d has no usable id (letters, digits, dot, dash, underscore; at most 40)" % position)
        if rid in seen:
            raise CompareError("requirement id %s appears twice" % rid)
        seen.add(rid)
        title = _text(entry, "title", rid, limit=200)
        kind = entry.get("kind")
        if kind not in KINDS:
            raise CompareError("requirement %s: kind must be one of %s" % (rid, ", ".join(KINDS)))
        severity = entry.get("severity", "major")
        if severity not in SEVERITIES:
            raise CompareError("requirement %s: severity must be one of %s" % (rid, ", ".join(SEVERITIES)))
        document = entry.get("document", "both")
        if document not in DOCUMENTS:
            raise CompareError("requirement %s: document must be a, b or both" % rid)
        phrase = pattern = text = None
        minimum = maximum = None
        unit = None
        tolerance = Decimal(0)
        if kind in ("present", "forbidden"):
            phrase = _text(entry, "phrase", rid, limit=MAX_PATTERN_CHARS)
        elif kind in ("match", "limit"):
            source = _text(entry, "pattern", rid, limit=MAX_PATTERN_CHARS)
            try:
                pattern = re.compile(source, re.IGNORECASE)
            except re.error as exc:
                raise CompareError("requirement %s: the pattern is not a regular expression (%s)" % (rid, exc)) from None
            if "value" not in pattern.groupindex:
                raise CompareError("requirement %s: the pattern needs a (?P<value>...) group" % rid)
            if kind == "match":
                if document != "both":
                    raise CompareError("requirement %s: a match compares both documents, so document must be both" % rid)
                if "tolerance" in entry:
                    tolerance = _decimal(entry["tolerance"], "tolerance", rid)
                    if tolerance < 0:
                        raise CompareError("requirement %s: tolerance must not be negative" % rid)
            else:
                if document == "both":
                    raise CompareError("requirement %s: a limit is read from one document, so document must be a or b" % rid)
                if "min" not in entry and "max" not in entry:
                    raise CompareError("requirement %s: a limit needs min, max or both" % rid)
                if "min" in entry:
                    minimum = _decimal(entry["min"], "min", rid)
                if "max" in entry:
                    maximum = _decimal(entry["max"], "max", rid)
                if minimum is not None and maximum is not None and minimum > maximum:
                    raise CompareError("requirement %s: min is above max" % rid)
                if "unit" in entry:
                    unit = _text(entry, "unit", rid, limit=20)
        else:
            text = _text(entry, "text", rid)
            if document == "both":
                raise CompareError("requirement %s: a judged statement is read from one document, so document must be a or b" % rid)
        if kind in ("present", "forbidden") and document not in DOCUMENTS:
            raise CompareError("requirement %s: document must be a, b or both" % rid)
        out.append(Requirement(rid, title, kind, severity, document, phrase, pattern, text, minimum, maximum, unit, tolerance))
    return out


def load_requirements(path: str | Path) -> tuple[list[Requirement], str]:
    """Read and validate a requirements file. Returns the requirements and the SHA-256 of the file's bytes, which the report records."""
    import hashlib

    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise CompareError("cannot open %s: %s" % (p.name, exc.strerror or "unreadable")) from None
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise CompareError("%s is not valid JSON" % p.name) from None
    return parse_requirements(raw), hashlib.sha256(data).hexdigest()
