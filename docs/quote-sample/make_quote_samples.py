# LIFESPAN: permanent (writes the committed sample outputs of the quotation drafter; tests/test_quote.py checks the files on disk are what this writes)
"""Write the sample quotation outputs from the invented inputs in this folder.

    python docs/quote-sample/make_quote_samples.py            rewrite quote.json, quote.txt and approved.json
    python docs/quote-sample/make_quote_samples.py --check    exit 1 if a file on disk differs from what this would write

The inputs (fee_schedule.json, enquiry.txt, template.txt, acknowledgements.json) are made up: a small consultancy's fee schedule and an enquiry with a service that is not on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))

from app.quote.quote import approve, draft, load_json, parse_enquiry, parse_schedule  # noqa: E402

APPROVED_AT = "2026-11-17T00:00:00Z"


def _json(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def build() -> dict[str, bytes]:
    schedule = parse_schedule(load_json(HERE / "fee_schedule.json"))
    quote = draft("Marlow Council", "Q-1001", parse_enquiry((HERE / "enquiry.txt").read_text(encoding="utf-8")), schedule,
                  (HERE / "template.txt").read_text(encoding="utf-8"), discount_percent="15")
    acks = load_json(HERE / "acknowledgements.json")["acknowledgements"]
    return {"quote.json": _json(quote.to_dict()), "quote.txt": quote.body.encode("utf-8"), "approved.json": _json(approve(quote, "Dana Planner", acks, APPROVED_AT))}


def main(argv: list[str]) -> int:
    files = build()
    if "--check" in argv:
        stale = [n for n, d in files.items() if not (HERE / n).exists() or (HERE / n).read_bytes() != d]
        print("stale: " + ", ".join(stale) if stale else "all %d sample files are current" % len(files))
        return 1 if stale else 0
    for name, data in files.items():
        (HERE / name).write_bytes(data)
    print("wrote %d files to %s" % (len(files), HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
