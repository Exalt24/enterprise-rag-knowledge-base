"""Command line for the two-PDF requirement check.

    python -m app.compare compare design_basis.pdf calc_report.pdf --requirements requirements.json --out report.json [--text report.txt] [--strict]
    python -m app.compare review  report.json --reviewer "Dana Planner" --decisions decisions.json --out approved.json
    python -m app.compare verify  approved.json

`compare` writes the structured report (and a plain text one with `--text`) and prints the same text. It exits 0 unless `--strict` is given and a blocking finding is not met
(exit 1), and 2 for a file it cannot use. `review` is the approval checkpoint: exit 2 and nothing written while a blocking finding has no decision. `verify` exits 0 for an
approval that is intact and 2 for one that was touched.

The judge for statements in words is off here: a `judge` requirement is reported as unclear. A service wires `ChatModelJudge` to its own chat model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .approval import approve, load_json, verify_approved
from .check import compare
from .pages import CompareError, read_pdf
from .report import render_text, report_from_dict
from .requirements import load_requirements


def _write_json(path: str, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="check two PDFs against a requirements file")
    c.add_argument("document_a")
    c.add_argument("document_b")
    c.add_argument("--requirements", required=True)
    c.add_argument("--out", required=True, help="where the JSON report is written")
    c.add_argument("--text", help="also write the plain text report here")
    c.add_argument("--strict", action="store_true", help="exit 1 when a blocking finding is not met")
    r = sub.add_parser("review", help="the approval checkpoint")
    r.add_argument("report")
    r.add_argument("--reviewer", required=True)
    r.add_argument("--decisions", required=True)
    r.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="check an approved file has not been touched")
    v.add_argument("approved")
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            requirements, req_hash = load_requirements(args.requirements)
            report = compare(read_pdf(args.document_a), read_pdf(args.document_b), requirements, req_hash)
            _write_json(args.out, report.to_dict())
            text = render_text(report)
            if args.text:
                Path(args.text).write_text(text, encoding="utf-8")
            sys.stdout.write(text)
            return 1 if args.strict and report.to_dict()["summary"]["open_blocking"] else 0
        if args.command == "review":
            report = report_from_dict(load_json(args.report))
            decisions = load_json(args.decisions)
            approved = approve(report, args.reviewer, decisions.get("decisions") if isinstance(decisions, dict) else decisions)
            _write_json(args.out, approved)
            print("approved by %s, report %s" % (approved["reviewer"], report.sha256[:12]))
            return 0
        report = verify_approved(load_json(args.approved))
        print("approval intact, report %s" % report.sha256[:12])
        return 0
    except CompareError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
