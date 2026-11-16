"""Seeded worlds for the two-PDF check, against an oracle written from the plan each document was built from.

The dimensions varied together in every world: which figure each document holds (a value, none, two that disagree, a format that is not understood, or a different unit), where it
sits (any readable page, repeated on several), how the phrases are written (random case, broken over two lines), how many pages have no text layer (0 to 2, anywhere), how long the
documents are, and noise lines that look like the requirements without being them. The oracle never reads a PDF: it is the plan, turned into the status each rule must give.
"""
from __future__ import annotations

import random
from decimal import Decimal

import pytest

from app.compare.check import compare
from app.compare.pages import read_pdf
from app.compare.requirements import parse_requirements
from compare_pdfs import make_pdf

REQUIREMENTS = parse_requirements({"requirements": [
    {"id": "W", "title": "wind", "kind": "match", "severity": "blocking", "pattern": "wind speed:? (?P<value>[0-9][0-9,.]*) (?P<unit>m/s|km/h)"},
    {"id": "C", "title": "concrete", "kind": "limit", "severity": "blocking", "document": "b", "pattern": "concrete grade:? (?P<value>[0-9][0-9,.]*) (?P<unit>MPa)",
     "min": "32", "unit": "MPa"},
    {"id": "S", "title": "no stamp", "kind": "forbidden", "severity": "blocking", "document": "b", "phrase": "not for issue"},
    {"id": "P", "title": "second engineer", "kind": "present", "severity": "major", "document": "b", "phrase": "checked by a second engineer"},
    {"id": "F", "title": "fee", "kind": "match", "severity": "major", "pattern": "fee estimate:? (?P<value>[0-9][0-9,.]*) (?P<unit>AUD)"},
]})

NOISE = ["General notes apply to all sheets.", "Wind speed is covered in section four.", "Concrete is placed in the dry season.", "The fee schedule is attached.",
         "Checked drawings are filed with the registry.", "Nothing here is a figure."]


def fmt(n: int, rng: random.Random) -> str:
    return "{:,}".format(n) if rng.random() < 0.5 else str(n)


def plan_figure(rng: random.Random, low: int, high: int) -> tuple:
    kind = rng.choice(["val", "val", "val", "absent", "conflict", "bad"])
    if kind == "val":
        return ("val", rng.randint(low, high))
    if kind == "conflict":
        a = rng.randint(low, high)
        return ("conflict", a, a + rng.randint(1, 9))
    return (kind,)


def lines_for(plan: tuple, label: str, unit: str, rng: random.Random) -> list[str]:
    """The lines a document holds for one figure. One line per occurrence, so a figure can be repeated or contradicted."""
    if plan[0] == "absent":
        return []
    if plan[0] == "bad":
        return ["%s: 1.200,5 %s" % (label, unit)]
    if plan[0] == "conflict":
        return ["%s: %s %s" % (label, fmt(plan[1], rng), unit), "%s: %s %s" % (label, fmt(plan[2], rng), unit)]
    return ["%s: %s %s" % (label, fmt(plan[1], rng), unit)] * rng.randint(1, 3)


def spread(lines: list[str], rng: random.Random, unread: int) -> list[list[str] | None]:
    """Put the lines on random readable pages among noise, with `unread` text-free pages inserted at random positions."""
    readable = rng.randint(1, 4)
    pages: list[list[str] | None] = [[rng.choice(NOISE)] for _ in range(readable)]
    for line in lines:
        target = pages[rng.randrange(readable)]
        target.insert(rng.randint(0, len(target)), line)
    for _ in range(unread):
        pages.insert(rng.randint(0, len(pages)), None)
    return pages


def status_of(kind_plan: tuple, unreadable: bool) -> str | None:
    """The status a single document's figure gives on its own, or None when it is a usable value. `bad` and `conflict` are unclear; an absent one is missing, or unclear when a page is unread."""
    if kind_plan[0] == "bad" or kind_plan[0] == "conflict":
        return "unclear"
    if kind_plan[0] == "absent":
        return "unclear" if unreadable else "missing"
    return None


def case_wiggle(text: str, rng: random.Random) -> str:
    return "".join(c.upper() if rng.random() < 0.3 else c for c in text)


def build_world(seed: int):
    """One seeded world: the findings the check gave and the statuses the plan says it must give."""
    rng = random.Random(seed)
    unread_a, unread_b = rng.choice([0, 0, 1, 2]), rng.choice([0, 0, 1, 2])
    wind_a, wind_b = plan_figure(rng, 20, 60), plan_figure(rng, 20, 60)
    wind_unit_b = rng.choice(["m/s", "m/s", "m/s", "km/h"])
    concrete_b = plan_figure(rng, 20, 50)
    fee_a, fee_b = plan_figure(rng, 1000, 90000), plan_figure(rng, 1000, 90000)
    stamp = rng.random() < 0.4
    engineer = rng.choice(["yes", "wrapped", "no"])
    if rng.random() < 0.3 and wind_a[0] == "val" and wind_b[0] == "val":
        wind_b = wind_a                                           # agreeing documents must be common enough to see the met path
    if rng.random() < 0.3 and fee_a[0] == "val" and fee_b[0] == "val":
        fee_b = fee_a

    a_lines = lines_for(wind_a, "Wind speed", "m/s", rng) + lines_for(fee_a, "Fee estimate", "AUD", rng)
    b_lines = lines_for(wind_b, "Wind speed", wind_unit_b, rng) + lines_for(concrete_b, "Concrete grade", "MPa", rng) + lines_for(fee_b, "Fee estimate", "AUD", rng)
    if stamp:
        b_lines.append(case_wiggle("Status: draft - not for issue", rng))
    if engineer == "yes":
        b_lines.append(case_wiggle("This was checked by a second engineer.", rng))
    elif engineer == "wrapped":
        b_lines += ["This was checked by a", "second engineer."]
    wrapped_pair = [("This was checked by a", "second engineer.")] if engineer == "wrapped" else []
    pages_b = spread([l for l in b_lines if not any(l == w[0] or l == w[1] for w in wrapped_pair)], rng, unread_b)
    if wrapped_pair:                                              # a phrase broken over two lines sits together on one readable page
        readable_pages = [p for p in pages_b if p is not None]
        rng.choice(readable_pages).extend(wrapped_pair[0])
    a = read_pdf(make_pdf(spread(a_lines, rng, unread_a)), "a.pdf")
    b = read_pdf(make_pdf(pages_b), "b.pdf")
    assert (len(a.unreadable), len(b.unreadable)) == (unread_a, unread_b), "the fixture planted the pages it meant to"

    got = {f.id: f for f in compare(a, b, REQUIREMENTS, "0" * 64).findings}

    def match_expected(pa: tuple, pb: tuple, unit_b: str, ua: int, ub: int) -> str:
        for plan, unread in ((pa, ua), (pb, ub)):                 # document A is judged first, so its problem is the one reported
            problem = status_of(plan, bool(unread))
            if problem:
                return problem
        if unit_b != "m/s" and unit_b != "AUD":
            return "discrepancy"
        return "met" if pa[1] == pb[1] else "discrepancy"

    expected = {
        "W": match_expected(wind_a, wind_b, wind_unit_b, unread_a, unread_b),
        "F": match_expected(fee_a, fee_b, "AUD", unread_a, unread_b),
        "C": status_of(concrete_b, bool(unread_b)) or ("met" if concrete_b[1] >= 32 else "discrepancy"),
        "S": "discrepancy" if stamp else "unclear" if unread_b else "met",
        "P": "met" if engineer != "no" else "unclear" if unread_b else "missing",
    }
    return got, expected, a, b


@pytest.mark.parametrize("seed", range(80))
def test_every_rule_matches_the_plan_the_documents_were_built_from(seed):
    got, expected, a, b = build_world(seed)
    assert {k: f.status for k, f in got.items()} == expected, seed
    for f in got.values():
        if f.status == "discrepancy":
            assert f.evidence, (seed, f.id, "a discrepancy always points at a page")
        for e in f.evidence:
            page = (a if e.document == "a" else b).page(e.page)
            assert page is not None and e.quote.strip(".").strip().split("...")[0].strip() in " ".join(page.text.split()), (seed, f.id, "a quote is words from its page")


def test_the_80_worlds_reach_every_status_for_every_rule_that_can_give_it():
    reached: dict[str, set[str]] = {}
    for seed in range(80):
        _, expected, _, _ = build_world(seed)
        for rid, status in expected.items():
            reached.setdefault(rid, set()).add(status)
    everything = {"met", "discrepancy", "missing", "unclear"}
    assert reached["W"] == reached["F"] == reached["C"] == everything
    assert reached["S"] == {"met", "discrepancy", "unclear"} and reached["P"] == {"met", "missing", "unclear"}
