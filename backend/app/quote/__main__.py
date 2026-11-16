"""Command line for the quotation drafter.

    python -m app.quote draft --enquiry enquiry.txt --fees fees.json --template template.txt --client "Marlow Council" --reference Q-1001 --out quote.json [--text quote.txt] [--discount 5]
    python -m app.quote approve quote.json --reviewer "Dana Planner" --acknowledgements acks.json --out approved.json
    python -m app.quote verify approved.json

`draft` writes the draft and prints it, with every flag listed; it exits 0 whether or not there are flags (a flag is the point of the draft, not a failure). `approve` exits 2 and writes
nothing while a flag has no acknowledgement. `verify` exits 0 for an intact approval and 2 for one that was touched. Nothing is ever sent.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from app.compare.pages import CompareError

from .quote import Flag, Line, Quote, approve, draft, load_json, parse_enquiry, parse_schedule, recompute_hash, verify


def _write_json(path: str, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _quote_from_dict(raw: dict) -> Quote:
    lines = tuple(Line(l["code"], l["description"], Decimal(l["quantity"]), l["unit"], None if l["rate"] is None else Decimal(l["rate"]),
                       None if l["amount"] is None else Decimal(l["amount"]), l["source"]) for l in raw["lines"])
    return Quote(raw["client"], raw["reference"], raw["currency"], lines, Decimal(raw["subtotal"]), Decimal(raw["discount_percent"]), Decimal(raw["discount"]),
                 Decimal(raw["tax"]), Decimal(raw["total"]), tuple(Flag(f["id"], f["reason"]) for f in raw["flags"]), raw["body"], raw["quote_sha256"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.quote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--enquiry", required=True)
    d.add_argument("--fees", required=True)
    d.add_argument("--template", required=True)
    d.add_argument("--client", required=True)
    d.add_argument("--reference", required=True)
    d.add_argument("--discount", default="0")
    d.add_argument("--out", required=True)
    d.add_argument("--text")
    a = sub.add_parser("approve")
    a.add_argument("quote")
    a.add_argument("--reviewer", required=True)
    a.add_argument("--acknowledgements", required=True)
    a.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("approved")
    args = parser.parse_args(argv)
    try:
        if args.command == "draft":
            schedule = parse_schedule(load_json(args.fees))
            try:
                enquiry, template = Path(args.enquiry).read_text(encoding="utf-8"), Path(args.template).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise CompareError("cannot read %s" % getattr(exc, "filename", "an input file")) from None
            quote = draft(args.client, args.reference, parse_enquiry(enquiry), schedule, template, discount_percent=args.discount)
            _write_json(args.out, quote.to_dict())
            if args.text:
                Path(args.text).write_text(quote.body, encoding="utf-8")
            sys.stdout.write(quote.body + "\n")
            return 0
        if args.command == "approve":
            raw = load_json(args.quote)
            acks = load_json(args.acknowledgements)
            try:
                quote = _quote_from_dict(raw)
            except (KeyError, TypeError, ValueError, ArithmeticError):
                raise CompareError("the file is not a quote this tool wrote") from None
            if recompute_hash(quote) != raw.get("quote_sha256") or quote.to_dict() != raw:
                raise CompareError("the quote file does not match its hash: it was changed after it was drafted")
            approved = approve(quote, args.reviewer, acks.get("acknowledgements") if isinstance(acks, dict) else acks)
            _write_json(args.out, approved)
            print("approved by %s, quote %s" % (approved["reviewer"], quote.sha256[:12]))
            return 0
        print("approval intact, quote %s" % verify(load_json(args.approved))[:12])
        return 0
    except CompareError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
