# LIFESPAN: permanent (writes the committed sample documents and their outputs; tests/test_compare_samples.py checks the files on disk are what this writes)
"""Write the sample documents for the two-PDF requirement check, and what the check says about them.

    python docs/compare-sample/make_samples.py            rewrite the files in this folder
    python docs/compare-sample/make_samples.py --check    exit 1 if a file on disk differs from what this would write

Everything here is invented: a made-up footbridge design basis (document A), a made-up calculation report that is meant to be checked against it (document B), and a requirements
file. Document B has a scanned page (page 3 has no text layer), a draft stamp, a wind speed and a concrete grade that disagree with the basis, and one figure that agrees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parents[1] / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "tests"))

from app.compare.approval import approve  # noqa: E402
from app.compare.check import compare  # noqa: E402
from app.compare.pages import read_pdf  # noqa: E402
from app.compare.report import render_text  # noqa: E402
from app.compare.requirements import load_requirements  # noqa: E402
from compare_pdfs import make_pdf  # noqa: E402

DESIGN_BASIS = [
    ["DESIGN BASIS - MARLOW CREEK FOOTBRIDGE", "Design wind speed: 45 m/s", "Design life: 50 years", "Concrete grade: 40 MPa", "Deck live load: 5.0 kPa"],
    ["Fee estimate: 18,400 AUD", "All calculations are to be checked by a second engineer."],
]
CALC_REPORT = [
    ["CALCULATION REPORT - MARLOW CREEK FOOTBRIDGE", "Status: DRAFT - NOT FOR ISSUE", "Design wind speed: 40 m/s"],
    ["Design life: 50 years", "Concrete grade: 32 MPa", "Deck live load: 5.0 kPa"],
    None,
    ["Fee estimate: 18,400 AUD"],
]
REQUIREMENTS = {"requirements": [
    {"id": "R1", "title": "Design wind speed agrees with the design basis", "kind": "match", "severity": "blocking",
     "pattern": "design wind speed:? (?P<value>[0-9][0-9,.]*) (?P<unit>m/s|km/h)"},
    {"id": "R2", "title": "Design life agrees with the design basis", "kind": "match", "severity": "major",
     "pattern": "design life:? (?P<value>[0-9][0-9,.]*) (?P<unit>years)"},
    {"id": "R3", "title": "Concrete grade is at least the basis grade", "kind": "limit", "severity": "blocking", "document": "b",
     "pattern": "concrete grade:? (?P<value>[0-9][0-9,.]*) (?P<unit>MPa)", "min": "40", "unit": "MPa"},
    {"id": "R4", "title": "The report is not stamped as a draft", "kind": "forbidden", "severity": "blocking", "document": "b", "phrase": "DRAFT - NOT FOR ISSUE"},
    {"id": "R5", "title": "The report records an independent check", "kind": "present", "severity": "blocking", "document": "b", "phrase": "checked by a second engineer"},
    {"id": "R6", "title": "Deck live load agrees with the design basis", "kind": "match", "severity": "major",
     "pattern": "deck live load:? (?P<value>[0-9][0-9,.]*) (?P<unit>kPa)"},
    {"id": "R7", "title": "Fee estimate agrees with the design basis", "kind": "match", "severity": "major",
     "pattern": "fee estimate:? (?P<value>[0-9][0-9,.]*) (?P<unit>AUD)"},
    {"id": "R8", "title": "The basis is issued at a named revision", "kind": "present", "severity": "major", "document": "a", "phrase": "Revision:"},
    {"id": "R9", "title": "The report says how the deck was checked for pedestrian vibration", "kind": "judge", "severity": "minor", "document": "b",
     "text": "The report states how the footbridge deck was checked for pedestrian vibration."},
]}
DECISIONS = {"decisions": [
    {"finding": "R1", "decision": "accept"},
    {"finding": "R3", "decision": "accept"},
    {"finding": "R4", "decision": "waive", "reason": "The stamp is the template footer and the issue copy is made without it."},
    {"finding": "R5", "decision": "reject", "reason": "The scanned page 3 is the signed checking sheet, checked by hand."},
]}
REVIEWER = "Dana Planner"
APPROVED_AT = "2026-11-17T00:00:00Z"


def _json(data: object) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def build() -> dict[str, bytes]:
    """Every file this folder holds (except this script), by name."""
    files: dict[str, bytes] = {
        "design_basis.pdf": make_pdf(DESIGN_BASIS),
        "calculation_report.pdf": make_pdf(CALC_REPORT),
        "requirements.json": _json(REQUIREMENTS),
        "decisions.json": _json(DECISIONS),
    }
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "r.json").write_bytes(files["requirements.json"])
        requirements, digest = load_requirements(Path(tmp) / "r.json")
    report = compare(read_pdf(files["design_basis.pdf"], "design_basis.pdf"), read_pdf(files["calculation_report.pdf"], "calculation_report.pdf"), requirements, digest)
    files["report.json"] = _json(report.to_dict())
    files["report.txt"] = render_text(report).encode("utf-8")
    files["approved.json"] = _json(approve(report, REVIEWER, DECISIONS["decisions"], APPROVED_AT))
    return files


def main(argv: list[str]) -> int:
    files = build()
    if "--check" in argv:
        stale = [name for name, data in files.items() if not (HERE / name).exists() or (HERE / name).read_bytes() != data]
        print("stale: " + ", ".join(stale) if stale else "all %d sample files are current" % len(files))
        return 1 if stale else 0
    for name, data in files.items():
        (HERE / name).write_bytes(data)
    print("wrote %d files to %s" % (len(files), HERE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
