"""Draft a quotation from an enquiry, a fee schedule, business rules and a template, with every doubt flagged for a person.

The drafting is plain arithmetic and plain rules, so a figure is never a model's guess:

  * A line is `quantity x rate` in `Decimal`, rounded half up to the currency's cent, and the totals are sums of those rounded lines.
  * A service the enquiry names that is not in the fee schedule gets NO price. It becomes a flag, so a person prices it.
  * A quantity outside the service's minimum and maximum, a service marked as needing review, a requested discount above the authority the schedule gives, and a total under the
    minimum quote are each a flag with the reason. The discount applied is never more than the authority.
  * The template is filled by name, and a placeholder the template uses that this module does not know is an error, not a blank in a client's document.

The enquiry is read by `parse_enquiry`: lines such as `3 x site inspection` or `site inspection x 3`, matched on the schedule's own names and aliases, case ignored. Free text a model
has read can be handed in through an `Extractor`; its answer counts only when each service quotes words that are really in the enquiry, so a model cannot add a service the client
never asked for.

A draft is a draft. `approve` is the human step: every flag needs a named reviewer's acknowledgement with a reason, and the approval is bound to the draft's hash, so changing a
line afterwards is found by `verify`. Nothing here sends anything.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, Sequence

from app.compare.pages import CompareError

#: The public surface: a service wires `extract_with_model` to its own chat model for free text enquiries; the command line uses the rest.
__all__ = ["Quote", "Service", "Schedule", "Requested", "Line", "Flag", "Extractor", "parse_schedule", "parse_enquiry", "extract_with_model", "draft", "approve", "verify", "load_json", "money", "recompute_hash"]

CENT = Decimal("0.01")
MAX_SERVICES = 500
MAX_ENQUIRY_LINES = 200
MAX_TEXT_CHARS = 20000
MIN_REASON_CHARS = 10

_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,29}$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
KNOWN_PLACEHOLDERS = frozenset({"client", "reference", "currency", "lines", "subtotal", "discount", "tax", "total", "flags"})


@dataclass(frozen=True)
class Service:
    code: str
    name: str
    unit: str
    rate: Decimal
    aliases: tuple[str, ...]
    minimum: Decimal
    maximum: Decimal | None
    needs_review: bool


@dataclass(frozen=True)
class Schedule:
    currency: str
    tax_rate: Decimal
    discount_authority_percent: Decimal
    minimum_total: Decimal
    services: tuple[Service, ...]


@dataclass(frozen=True)
class Requested:
    """One thing the enquiry asks for, with the words it came from."""

    text: str
    quantity: Decimal
    source: str


@dataclass(frozen=True)
class Line:
    code: str | None
    description: str
    quantity: Decimal
    unit: str
    rate: Decimal | None
    amount: Decimal | None
    source: str


@dataclass(frozen=True)
class Flag:
    id: str
    reason: str


class Extractor(Protocol):
    name: str

    def extract(self, enquiry: str, schedule: Schedule) -> list[dict]: ...


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def _decimal(raw: Any, what: str, *, allow_zero: bool = True) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise CompareError("%s must be a number written as text or a whole number" % what)
    try:
        value = Decimal(str(raw).strip())
    except InvalidOperation:
        raise CompareError("%s is not a number" % what) from None
    if not value.is_finite() or value < 0 or (value == 0 and not allow_zero):
        raise CompareError("%s must be a finite number above %s" % (what, "or equal to 0" if allow_zero else "0"))
    return value


def parse_schedule(raw: Any) -> Schedule:
    """Validate the decoded fee schedule JSON."""
    if not isinstance(raw, dict):
        raise CompareError("the fee schedule must be a JSON object")
    currency = raw.get("currency")
    if not isinstance(currency, str) or not _CURRENCY_RE.match(currency):
        raise CompareError("the fee schedule needs an ISO currency such as AUD")
    tax = _decimal(raw.get("tax_rate", "0"), "tax_rate")
    if tax > 1:
        raise CompareError("tax_rate is a fraction such as 0.10, not a percent")
    rules = raw.get("rules", {})
    if not isinstance(rules, dict):
        raise CompareError("rules must be an object")
    authority = _decimal(rules.get("discount_authority_percent", "0"), "discount_authority_percent")
    if authority > 100:
        raise CompareError("discount_authority_percent is above 100")
    minimum_total = _decimal(rules.get("minimum_quote_total", "0"), "minimum_quote_total")
    entries = raw.get("services")
    if not isinstance(entries, list) or not entries:
        raise CompareError("the fee schedule needs at least one service")
    if len(entries) > MAX_SERVICES:
        raise CompareError("more than %d services in one schedule" % MAX_SERVICES)
    services: list[Service] = []
    codes: set[str] = set()
    names: dict[str, str] = {}
    for position, e in enumerate(entries, start=1):
        if not isinstance(e, dict):
            raise CompareError("service number %d is not an object" % position)
        code = e.get("code")
        if not isinstance(code, str) or not _CODE_RE.match(code):
            raise CompareError("service number %d has no usable code" % position)
        if code in codes:
            raise CompareError("service code %s appears twice" % code)
        codes.add(code)
        name, unit = e.get("name"), e.get("unit")
        if not isinstance(name, str) or not name.strip() or not isinstance(unit, str) or not unit.strip():
            raise CompareError("service %s needs a name and a unit" % code)
        rate = _decimal(e.get("rate"), "the rate of %s" % code)
        minimum = _decimal(e.get("min_qty", "0"), "min_qty of %s" % code)
        maximum = _decimal(e["max_qty"], "max_qty of %s" % code) if "max_qty" in e else None
        if maximum is not None and maximum < minimum:
            raise CompareError("service %s has max_qty below min_qty" % code)
        aliases = e.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
            raise CompareError("aliases of %s must be a list of words" % code)
        flag = e.get("needs_review", False)
        if not isinstance(flag, bool):
            raise CompareError("needs_review of %s must be true or false" % code)
        words = tuple(dict.fromkeys(w.strip().casefold() for w in [name, *aliases]))
        for w in words:
            if w in names and names[w] != code:
                raise CompareError("the words %r name both %s and %s, so an enquiry could not be matched" % (w, names[w], code))
            names[w] = code
        services.append(Service(code, name.strip(), unit.strip(), rate, words, minimum, maximum, flag))
    return Schedule(currency, tax, authority, minimum_total, tuple(services))


def load_json(path: str | Path) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_bytes().decode("utf-8"))
    except OSError as exc:
        raise CompareError("cannot open %s: %s" % (p.name, exc.strerror or "unreadable")) from None
    except (UnicodeDecodeError, ValueError):
        raise CompareError("%s is not valid JSON" % p.name) from None


def _service_for(text: str, schedule: Schedule) -> Service | None:
    wanted = re.sub(r"\s+", " ", text).strip().casefold()
    for s in schedule.services:
        if wanted in s.aliases:
            return s
    return None


_LINE_A = re.compile(r"^\s*(?P<q>\d+(?:\.\d+)?)\s*(?:x|×)\s*(?P<t>.+?)\s*$", re.IGNORECASE)
_LINE_B = re.compile(r"^\s*(?P<t>.+?)\s*(?:x|×)\s*(?P<q>\d+(?:\.\d+)?)\s*$", re.IGNORECASE)


def parse_enquiry(text: str) -> list[Requested]:
    """The requested things in an enquiry, one per line: `3 x site inspection` or `site inspection x 3`. A line that is neither is returned with quantity 0 for a person to read."""
    if len(text) > MAX_TEXT_CHARS:
        raise CompareError("the enquiry is longer than %d characters" % MAX_TEXT_CHARS)
    lines = [l for l in (raw.strip() for raw in text.splitlines()) if l]
    if not lines:
        raise CompareError("the enquiry is empty")
    if len(lines) > MAX_ENQUIRY_LINES:
        raise CompareError("the enquiry has more than %d lines" % MAX_ENQUIRY_LINES)
    out = []
    for line in lines:
        m = _LINE_A.match(line) or _LINE_B.match(line)
        out.append(Requested(m.group("t"), Decimal(m.group("q")), line) if m else Requested(line, Decimal(0), line))
    return out


def extract_with_model(extractor: Extractor, enquiry: str, schedule: Schedule) -> list[Requested]:
    """Things a model read from free text. Each must name a service of the schedule, carry a plain number as its quantity and quote words that are in the enquiry."""
    try:
        answer = extractor.extract(enquiry, schedule)
    except Exception as exc:  # a model call: its failure must not become a quote
        raise CompareError("the extractor failed (%s), so no services were read from the enquiry" % type(exc).__name__) from None
    if not isinstance(answer, list):
        raise CompareError("the extractor did not return a list")
    haystack = re.sub(r"\s+", " ", enquiry).casefold()
    out = []
    for item in answer:
        if not isinstance(item, dict) or not isinstance(item.get("service"), str) or not isinstance(item.get("quote"), str):
            raise CompareError("an extracted item has no service and quote")
        if _service_for(item["service"], schedule) is None:
            raise CompareError("the extractor named %r, which is not in the fee schedule" % item["service"][:40])
        quote = re.sub(r"\s+", " ", item["quote"]).strip().casefold()
        if not quote or quote not in haystack:
            raise CompareError("the extractor quoted words that are not in the enquiry (%r)" % item["quote"][:40])
        qty = item.get("quantity")
        if isinstance(qty, bool) or not isinstance(qty, (int, str)):
            raise CompareError("an extracted quantity is not a plain number")
        out.append(Requested(item["service"], _decimal(qty, "an extracted quantity", allow_zero=False), item["quote"]))
    return out


@dataclass(frozen=True)
class Quote:
    client: str
    reference: str
    currency: str
    lines: tuple[Line, ...]
    subtotal: Decimal
    discount_percent: Decimal
    discount: Decimal
    tax: Decimal
    total: Decimal
    flags: tuple[Flag, ...]
    body: str
    sha256: str

    def data(self) -> dict[str, Any]:
        return {"client": self.client, "reference": self.reference, "currency": self.currency,
                "lines": [{"code": l.code, "description": l.description, "quantity": str(l.quantity), "unit": l.unit,
                           "rate": None if l.rate is None else str(l.rate), "amount": None if l.amount is None else str(l.amount), "source": l.source} for l in self.lines],
                "subtotal": str(self.subtotal), "discount_percent": str(self.discount_percent), "discount": str(self.discount), "tax": str(self.tax), "total": str(self.total),
                "flags": [{"id": f.id, "reason": f.reason} for f in self.flags], "body": self.body}

    def to_dict(self) -> dict[str, Any]:
        return {**self.data(), "quote_sha256": self.sha256}


def _canonical(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def recompute_hash(quote: Quote) -> str:
    """The hash a quote's own content gives, for a command that reads a quote back from a file and must not trust the hash written in it."""
    return hashlib.sha256(_canonical(quote.data())).hexdigest()


def _render(template: str, values: dict[str, str]) -> str:
    unknown = sorted({m for m in _PLACEHOLDER.findall(template) if m not in KNOWN_PLACEHOLDERS})
    if unknown:
        raise CompareError("the template uses placeholders this tool does not know: %s" % ", ".join("{%s}" % u for u in unknown))
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def draft(client: str, reference: str, requested: Sequence[Requested], schedule: Schedule, template: str, *, discount_percent: str | int = 0) -> Quote:
    """Price the requested things from the schedule, flag every doubt and fill the template."""
    if not isinstance(client, str) or not client.strip() or not isinstance(reference, str) or not reference.strip():
        raise CompareError("a quote needs a client and a reference")
    if len(template) > MAX_TEXT_CHARS:
        raise CompareError("the template is longer than %d characters" % MAX_TEXT_CHARS)
    if not requested:
        raise CompareError("nothing was requested")
    asked_discount = _decimal(discount_percent, "the requested discount")
    if asked_discount > 100:
        raise CompareError("the requested discount is above 100 percent")
    lines: list[Line] = []
    flags: list[Flag] = []
    for position, r in enumerate(requested, start=1):
        fid = "L%d" % position
        service = _service_for(r.text, schedule)
        if service is None:
            lines.append(Line(None, r.text, r.quantity, "", None, None, r.source))
            flags.append(Flag(fid, "%r is not in the fee schedule, so it has no price" % r.text[:60]))
            continue
        if r.quantity <= 0:
            lines.append(Line(service.code, service.name, r.quantity, service.unit, service.rate, None, r.source))
            flags.append(Flag(fid, "%s has no quantity in the enquiry line %r" % (service.name, r.source[:60])))
            continue
        lines.append(Line(service.code, service.name, r.quantity, service.unit, service.rate, money(r.quantity * service.rate), r.source))
        if r.quantity < service.minimum:
            flags.append(Flag(fid, "%s: quantity %s is below the minimum %s" % (service.name, r.quantity, service.minimum)))
        if service.maximum is not None and r.quantity > service.maximum:
            flags.append(Flag(fid, "%s: quantity %s is above the maximum %s" % (service.name, r.quantity, service.maximum)))
        if service.needs_review:
            flags.append(Flag(fid, "%s is marked in the fee schedule as needing a person's review" % service.name))
    subtotal = sum((l.amount for l in lines if l.amount is not None), Decimal(0))
    applied = min(asked_discount, schedule.discount_authority_percent)
    if asked_discount > schedule.discount_authority_percent:
        flags.append(Flag("DISCOUNT", "a discount of %s percent was asked for and the schedule allows %s, so %s was applied" % (asked_discount, schedule.discount_authority_percent, applied)))
    discount = money(subtotal * applied / 100)
    taxable = subtotal - discount
    tax = money(taxable * schedule.tax_rate)
    total = taxable + tax
    if total < schedule.minimum_total:
        flags.append(Flag("MINIMUM", "the total %s is below the minimum quote of %s" % (total, schedule.minimum_total)))
    shown = "\n".join("%-8s %-34s %8s %-6s x %10s = %12s" % (l.code or "-", l.description[:34], l.quantity, l.unit, "-" if l.rate is None else l.rate,
                                                              "NO PRICE" if l.amount is None else l.amount) for l in lines)
    values = {"client": client.strip(), "reference": reference.strip(), "currency": schedule.currency, "lines": shown, "subtotal": str(subtotal), "discount": str(discount),
              "tax": str(tax), "total": str(total), "flags": "\n".join("%s: %s" % (f.id, f.reason) for f in flags) or "none"}
    body = _render(template, values)
    partial = Quote(client.strip(), reference.strip(), schedule.currency, tuple(lines), subtotal, applied, discount, tax, total, tuple(flags), body, "")
    return Quote(**{**partial.__dict__, "sha256": recompute_hash(partial)})


def approve(quote: Quote, reviewer: Any, acknowledgements: Any, approved_at: str | None = None) -> dict[str, Any]:
    """The human step. Every flag needs a named reviewer's acknowledgement with a reason, and the approval is bound to the draft's hash."""
    from datetime import datetime, timezone

    problems: list[str] = []
    if not isinstance(reviewer, str) or len(reviewer.strip()) < 2:
        problems.append("the reviewer's name is missing")
    if not isinstance(acknowledgements, list):
        raise CompareError("the acknowledgements must be a list")
    known = {f.id for f in quote.flags}
    done: dict[str, str] = {}
    for position, a in enumerate(acknowledgements, start=1):
        if not isinstance(a, dict) or not isinstance(a.get("flag"), str):
            problems.append("acknowledgement number %d names no flag" % position)
            continue
        if a["flag"] not in known:
            problems.append("acknowledgement for %s: there is no such flag" % a["flag"])
        elif a["flag"] in done:
            problems.append("acknowledgement for %s: given twice" % a["flag"])
        else:
            reason = a.get("reason")
            if not isinstance(reason, str) or len(reason.strip()) < MIN_REASON_CHARS:
                problems.append("acknowledgement for %s needs a reason of at least %d characters" % (a["flag"], MIN_REASON_CHARS))
            else:
                done[a["flag"]] = reason.strip()
    open_flags = [f.id for f in quote.flags if f.id not in done]
    if open_flags:
        problems.append("flags with no acknowledgement: %s" % ", ".join(open_flags))
    if problems:
        raise CompareError("not approved: " + "; ".join(problems))
    acks = [{"flag": k, "reason": v} for k, v in sorted(done.items())]
    name = reviewer.strip()
    return {"quote": quote.to_dict(), "reviewer": name, "acknowledgements": acks,
            "approved_at": approved_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approval_sha256": hashlib.sha256(_canonical({"quote_sha256": quote.sha256, "reviewer": name, "acknowledgements": acks})).hexdigest()}


def verify(raw: Any) -> str:
    """Check an approved quote file. Returns the quote's hash, or raises naming what was touched."""
    try:
        quote_part, reviewer, acks = raw["quote"], raw["reviewer"], raw["acknowledgements"]
        stored = quote_part["quote_sha256"]
        body = {k: v for k, v in quote_part.items() if k != "quote_sha256"}
        flags = [f["id"] for f in body["flags"]]
    except (KeyError, TypeError, AttributeError):
        raise CompareError("the file is not an approved quote this tool wrote") from None
    if hashlib.sha256(_canonical(body)).hexdigest() != stored:
        raise CompareError("the quote does not match its hash: a line, a total or a flag was changed after it was drafted")
    if raw.get("approval_sha256") != hashlib.sha256(_canonical({"quote_sha256": stored, "reviewer": reviewer, "acknowledgements": acks})).hexdigest():
        raise CompareError("the approval does not match the quote and acknowledgements it names: something was changed after approval")
    missing = [f for f in flags if f not in {a.get("flag") for a in acks if isinstance(a, dict)}]
    if missing:
        raise CompareError("not approved: flags with no acknowledgement: %s" % ", ".join(missing))
    return stored
