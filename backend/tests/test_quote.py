"""The quotation drafter: the fee schedule, the enquiry reader, the pricing rules, the model extractor's grounding, the approval step, the command line and 80 seeded worlds.

Every refusal has a positive control (the same call without the defect succeeds).
"""
from __future__ import annotations

import copy
import json
import random
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from app.compare.pages import CompareError
from app.quote.__main__ import main
from app.quote.quote import Requested, approve, draft, extract_with_model, money, parse_enquiry, parse_schedule, verify

BACKEND = Path(__file__).resolve().parents[1]
SAMPLE = BACKEND.parent / "docs" / "quote-sample"
sys.path.insert(0, str(SAMPLE))
import make_quote_samples as make_samples  # noqa: E402

FEES = {"currency": "AUD", "tax_rate": "0.10", "rules": {"discount_authority_percent": "10", "minimum_quote_total": "500"},
        "services": [{"code": "A1", "name": "Site inspection", "unit": "visit", "rate": "850.00", "aliases": ["site visit"], "min_qty": "1", "max_qty": "5"},
                     {"code": "B2", "name": "Drawing set", "unit": "sheet", "rate": "620.50", "aliases": ["drawings"], "min_qty": "1", "needs_review": True},
                     {"code": "C3", "name": "Letter", "unit": "letter", "rate": "380", "max_qty": "10"}]}
TEMPLATE = "Q {reference} for {client} in {currency}\n{lines}\nsub {subtotal} disc {discount} tax {tax} total {total}\n{flags}"


def schedule(**over):
    raw = copy.deepcopy(FEES)
    raw.update(over)
    return parse_schedule(raw)


def make(text, discount=0, **over):
    return draft("Marlow Council", "Q-1", parse_enquiry(text), schedule(**over), TEMPLATE, discount_percent=discount)


def flags(q):
    return {f.id: f.reason for f in q.flags}


# ---------------------------------------------------------------- the fee schedule

@pytest.mark.parametrize("edit, match", [
    (lambda r: r.pop("currency"), "ISO currency"),
    (lambda r: r.update(currency="aud"), "ISO currency"),
    (lambda r: r.update(tax_rate="10"), "fraction"),
    (lambda r: r.update(tax_rate="lots"), "not a number"),
    (lambda r: r.update(tax_rate=True), "must be a number"),
    (lambda r: r.update(tax_rate="-0.1"), "finite number"),
    (lambda r: r["rules"].update(discount_authority_percent="101"), "above 100"),
    (lambda r: r.update(services=[]), "at least one service"),
    (lambda r: r["services"][0].update(code="a b"), "usable code"),
    (lambda r: r["services"][1].update(code="A1"), "appears twice"),
    (lambda r: r["services"][0].update(name=""), "name and a unit"),
    (lambda r: r["services"][0].update(rate="free"), "not a number"),
    (lambda r: r["services"][0].update(rate="-5"), "finite number"),
    (lambda r: r["services"][0].update(max_qty="0"), "max_qty below min_qty"),
    (lambda r: r["services"][0].update(aliases="x"), "list of words"),
    (lambda r: r["services"][0].update(needs_review="yes"), "true or false"),
    (lambda r: r["services"][1].update(aliases=["site visit"]), "name both"),
    (lambda r: r["services"][1].update(name="site INSPECTION"), "name both"),
])
def test_a_bad_fee_schedule_is_refused_for_its_reason_and_the_good_one_is_not(edit, match):
    assert parse_schedule(copy.deepcopy(FEES))
    raw = copy.deepcopy(FEES)
    edit(raw)
    with pytest.raises(CompareError, match=match):
        parse_schedule(raw)


# ---------------------------------------------------------------- the enquiry

def test_both_line_shapes_are_read_and_anything_else_is_kept_for_a_person():
    got = parse_enquiry("2 x Site Inspection\nDrawings x 12\n  1.5 × letter \nsomething free text\n\n")
    assert [(r.text, r.quantity) for r in got] == [("Site Inspection", Decimal(2)), ("Drawings", Decimal(12)), ("letter", Decimal("1.5")), ("something free text", Decimal(0))]


@pytest.mark.parametrize("text, match", [("", "empty"), ("\n \n", "empty"), ("x" * 20001, "longer than"), ("\n".join("1 x a" for _ in range(201)), "more than")])
def test_an_unusable_enquiry_is_refused(text, match):
    assert parse_enquiry("1 x a")
    with pytest.raises(CompareError, match=match):
        parse_enquiry(text)


# ---------------------------------------------------------------- the pricing

def test_a_line_is_quantity_times_rate_rounded_half_up_and_the_totals_are_sums_of_rounded_lines():
    q = make("3 x drawings\n1 x site inspection")
    assert [str(l.amount) for l in q.lines] == ["1861.50", "850.00"] and str(q.subtotal) == "2711.50" and str(q.tax) == "271.15" and str(q.total) == "2982.65"
    half = make("3 x drawings", **{"services": [dict(FEES["services"][1], rate="0.335")]})
    assert str(half.lines[0].amount) == "1.01", "3 x 0.335 = 1.005 rounds half up"
    assert money(Decimal("2.675")) == Decimal("2.68") and money(Decimal("2.665")) == Decimal("2.67")


def test_a_service_not_in_the_schedule_has_no_price_and_is_flagged_not_guessed():
    q = make("1 x site inspection\n2 x geotechnical investigation")
    assert q.lines[1].amount is None and q.lines[1].code is None and "not in the fee schedule" in flags(q)["L2"]
    assert str(q.subtotal) == "850.00" and "NO PRICE" in q.body


def test_each_rule_that_needs_a_person_is_a_flag_with_its_reason():
    assert not make("1 x site inspection").flags, "positive control: a clean enquiry has no flag"
    assert "below the minimum" in flags(make("1 x site inspection", services=[dict(FEES["services"][0], min_qty="2")]))["L1"]
    assert "above the maximum" in flags(make("6 x site inspection"))["L1"] and not make("5 x site inspection").flags, "exactly the maximum is allowed"
    assert "needing a person's review" in flags(make("1 x drawings"))["L1"]
    assert "no quantity" in flags(make("site inspection x 0"))["L1"] and make("site inspection x 0").lines[0].amount is None
    assert "no quantity" in flags(make("site inspection"))["L1"] or "not in the fee schedule" in flags(make("site inspection"))["L1"]


def test_a_discount_is_capped_at_the_authority_and_the_excess_is_flagged():
    assert not make("1 x site inspection", discount=10).flags and str(make("1 x site inspection", discount=10).discount) == "85.00", "exactly the authority"
    q = make("1 x site inspection", discount=15)
    assert str(q.discount_percent) == "10" and str(q.discount) == "85.00" and "allows 10" in flags(q)["DISCOUNT"]
    assert str(make("1 x site inspection", discount="2.5").discount) == "21.25"


@pytest.mark.parametrize("discount", [-1, "lots", 101, True, 1.5])
def test_a_discount_that_is_not_a_sane_number_is_refused(discount):
    assert make("1 x site inspection", discount=5)
    with pytest.raises(CompareError):
        make("1 x site inspection", discount=discount)


def test_a_total_under_the_minimum_is_flagged_and_exactly_the_minimum_is_not():
    assert "below the minimum quote" in flags(make("1 x letter", services=[dict(FEES["services"][2], rate="100")]))["MINIMUM"]
    q = make("1 x site inspection", rules={"minimum_quote_total": "935"})
    assert str(q.total) == "935.00" and "MINIMUM" not in flags(q)


def test_the_template_is_filled_by_name_and_an_unknown_placeholder_is_an_error():
    q = make("1 x site inspection")
    assert q.body.startswith("Q Q-1 for Marlow Council in AUD") and "total 935.00" in q.body and q.body.endswith("none")
    with pytest.raises(CompareError, match="does not know: \\{discountt\\}"):
        draft("c", "r", parse_enquiry("1 x site inspection"), schedule(), "x {discountt} {total}")
    with pytest.raises(CompareError, match="a client and a reference"):
        draft("", "r", parse_enquiry("1 x site inspection"), schedule(), TEMPLATE)
    with pytest.raises(CompareError, match="nothing was requested"):
        draft("c", "r", [], schedule(), TEMPLATE)


# ---------------------------------------------------------------- the model extractor

class FakeExtractor:
    name = "fake"

    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error

    def extract(self, enquiry, sched):
        if self.error:
            raise self.error
        return self.answer


ENQUIRY = "Hi, could you send someone for a site visit at the creek, we think two visits, and a set of drawings, around twelve sheets."


def test_a_grounded_extraction_becomes_requested_lines_and_prices():
    got = extract_with_model(FakeExtractor([{"service": "site visit", "quantity": 2, "quote": "two visits"}, {"service": "drawings", "quantity": "12", "quote": "a set of drawings"}]), ENQUIRY, schedule())
    assert [(r.text, r.quantity) for r in got] == [("site visit", Decimal(2)), ("drawings", Decimal(12))]
    assert str(draft("c", "r", got, schedule(), TEMPLATE).subtotal) == "9146.00"


@pytest.mark.parametrize("answer, match", [
    ([{"service": "site visit", "quantity": 2, "quote": "an engineer will inspect the tower"}], "not in the enquiry"),
    ([{"service": "wind tunnel test", "quantity": 1, "quote": "two visits"}], "not in the fee schedule"),
    ([{"service": "site visit", "quantity": 0, "quote": "two visits"}], "above 0"),
    ([{"service": "site visit", "quantity": 2.5, "quote": "two visits"}], "plain number"),
    ([{"service": "site visit", "quantity": True, "quote": "two visits"}], "plain number"),
    ([{"service": "site visit", "quantity": "many", "quote": "two visits"}], "not a number"),
    ([{"service": "site visit", "quantity": 2, "quote": ""}], "not in the enquiry"),
    ([{"service": "site visit", "quantity": 2}], "no service and quote"),
    (["site visit"], "no service and quote"),
    ("site visit", "did not return a list"),
])
def test_an_extraction_that_is_not_grounded_in_the_enquiry_is_refused(answer, match):
    good = [{"service": "site visit", "quantity": 2, "quote": "two visits"}]
    assert extract_with_model(FakeExtractor(good), ENQUIRY, schedule())
    with pytest.raises(CompareError, match=match):
        extract_with_model(FakeExtractor(answer), ENQUIRY, schedule())


def test_an_extractor_that_fails_leaks_nothing_and_adds_no_service():
    with pytest.raises(CompareError, match="extractor failed \\(RuntimeError\\)") as caught:
        extract_with_model(FakeExtractor(error=RuntimeError("https://api.example.test/key-123")), ENQUIRY, schedule())
    assert "key-123" not in str(caught.value)


# ---------------------------------------------------------------- the approval step

ACKS = [{"flag": "L1", "reason": "Confirmed by the director by phone."}]


def flagged():
    return make("1 x drawings")


def test_every_flag_needs_a_named_reviewers_acknowledgement_before_approval():
    quote = flagged()
    assert approve(quote, "Dana Planner", ACKS)["reviewer"] == "Dana Planner", "positive control"
    with pytest.raises(CompareError, match="flags with no acknowledgement: L1"):
        approve(quote, "Dana Planner", [])
    assert approve(make("1 x site inspection"), "Dana Planner", [])["acknowledgements"] == [], "a clean draft needs none"


@pytest.mark.parametrize("acks, name, match", [
    (ACKS + [{"flag": "L9", "reason": "no such flag at all"}], "Dana Planner", "no such flag"),
    (ACKS + ACKS, "Dana Planner", "given twice"),
    ([{"flag": "L1", "reason": "too short"}], "Dana Planner", "needs a reason"),
    ([{"flag": "L1"}], "Dana Planner", "needs a reason"),
    ([{"reason": "no flag named here at all"}], "Dana Planner", "names no flag"),
    (ACKS, "", "reviewer"),
    (ACKS, None, "reviewer"),
])
def test_an_acknowledgement_that_breaks_a_rule_is_refused(acks, name, match):
    assert approve(flagged(), "Dana Planner", ACKS)
    with pytest.raises(CompareError, match=match):
        approve(flagged(), name, acks)


def test_an_intact_approval_verifies_and_every_kind_of_touching_is_found():
    quote = flagged()
    approved = approve(quote, "Dana Planner", ACKS, "2026-11-17T00:00:00Z")
    assert verify(json.loads(json.dumps(approved))) == quote.sha256

    def touched(edit):
        raw = copy.deepcopy(approved)
        edit(raw)
        return raw

    edits = [lambda r: r["quote"]["lines"][0].__setitem__("amount", "1.00"), lambda r: r["quote"].__setitem__("total", "1.00"),
             lambda r: r["quote"]["flags"].clear(), lambda r: r["quote"].__setitem__("body", "other"), lambda r: r.__setitem__("reviewer", "Someone Else"),
             lambda r: r["acknowledgements"][0].__setitem__("reason", "a different reason entirely"), lambda r: r["acknowledgements"].clear(),
             lambda r: r.__setitem__("approval_sha256", "0" * 64)]
    for edit in edits:
        with pytest.raises(CompareError):
            verify(touched(edit))
    with pytest.raises(CompareError, match="not an approved quote"):
        verify({"quote": {}})


def test_a_forged_approval_with_a_consistent_hash_and_a_missing_acknowledgement_still_fails():
    import hashlib

    quote = flagged()
    approved = approve(quote, "Dana Planner", ACKS)
    forged = copy.deepcopy(approved)
    forged["acknowledgements"] = []
    body = json.dumps({"quote_sha256": quote.sha256, "reviewer": "Dana Planner", "acknowledgements": []}, sort_keys=True, separators=(",", ":")).encode()
    forged["approval_sha256"] = hashlib.sha256(body).hexdigest()
    with pytest.raises(CompareError, match="no acknowledgement: L1"):
        verify(forged)


def test_the_same_inputs_give_the_same_hash_and_a_changed_input_gives_another():
    assert make("1 x drawings").sha256 == make("1 x drawings").sha256 != make("2 x drawings").sha256


# ---------------------------------------------------------------- the command line and the samples

def test_the_sample_files_on_disk_are_exactly_what_the_script_writes_and_say_what_the_doc_says():
    for name, data in make_samples.build().items():
        assert (SAMPLE / name).read_bytes() == data, name
    raw = json.loads((SAMPLE / "quote.json").read_text(encoding="utf-8"))
    assert [f["id"] for f in raw["flags"]] == ["L3", "L4", "L5", "DISCOUNT"] and raw["total"] == "10484.10" and raw["discount_percent"] == "10"
    assert verify(json.loads((SAMPLE / "approved.json").read_text(encoding="utf-8"))) == raw["quote_sha256"]


def test_the_command_line_drafts_approves_and_verifies_and_a_refusal_writes_nothing(tmp_path, capsys):
    out, approved = str(tmp_path / "q.json"), str(tmp_path / "a.json")
    base = ["draft", "--enquiry", str(SAMPLE / "enquiry.txt"), "--fees", str(SAMPLE / "fee_schedule.json"), "--template", str(SAMPLE / "template.txt"),
            "--client", "Marlow Council", "--reference", "Q-1001", "--discount", "15", "--out", out]
    assert main(base + ["--text", str(tmp_path / "q.txt")]) == 0 and "NO PRICE" in capsys.readouterr().out
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"acknowledgements": [{"flag": "L3", "reason": "confirmed by the director"}]}), encoding="utf-8")
    assert main(["approve", out, "--reviewer", "Dana Planner", "--acknowledgements", str(partial), "--out", approved]) == 2
    assert not Path(approved).exists() and "flags with no acknowledgement: L4, L5, DISCOUNT" in capsys.readouterr().err
    assert main(["approve", out, "--reviewer", "Dana Planner", "--acknowledgements", str(SAMPLE / "acknowledgements.json"), "--out", approved]) == 0
    assert main(["verify", approved]) == 0 and "approval intact" in capsys.readouterr().out
    raw = json.loads(Path(approved).read_text(encoding="utf-8"))
    raw["quote"]["total"] = "1.00"
    Path(approved).write_text(json.dumps(raw), encoding="utf-8")
    assert main(["verify", approved]) == 2 and "changed after it was drafted" in capsys.readouterr().err
    tampered = json.loads(Path(out).read_text(encoding="utf-8"))
    tampered["total"] = "1.00"
    Path(out).write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["approve", out, "--reviewer", "Dana Planner", "--acknowledgements", str(SAMPLE / "acknowledgements.json"), "--out", approved]) == 2


@pytest.mark.parametrize("args, match", [
    (["--enquiry", "{missing}"], "cannot read"),
    (["--fees", "{missing}"], "cannot open"),
    (["--template", "{missing}"], "cannot read"),
    (["--reference", " "], "a client and a reference"),
])
def test_an_input_that_cannot_be_used_is_exit_2_with_a_reason(tmp_path, capsys, args, match):
    base = {"--enquiry": str(SAMPLE / "enquiry.txt"), "--fees": str(SAMPLE / "fee_schedule.json"), "--template": str(SAMPLE / "template.txt"),
            "--client": "Marlow Council", "--reference": "Q-1", "--out": str(tmp_path / "o.json")}
    flat = [x for k, v in base.items() for x in (k, v)]
    assert main(["draft", *flat]) == 0, "positive control"
    base.update({args[0]: args[1].format(missing=str(tmp_path / "nope"))})
    assert main(["draft", *[x for k, v in base.items() for x in (k, v)]]) == 2
    assert match in capsys.readouterr().err


# ---------------------------------------------------------------- seeded worlds against an oracle on exact fractions

def build_world(seed):
    """One seeded world: the drafted quote and what an oracle on exact fractions says it must hold."""
    rng = random.Random(seed)
    services = []
    for i in range(rng.randint(1, 6)):
        e = {"code": "S%d" % i, "name": "Service %d" % i, "unit": "unit", "rate": "%d.%02d" % (rng.randint(0, 4000), rng.randint(0, 99)), "aliases": ["alias %d" % i],
             "min_qty": str(rng.randint(0, 2)), "needs_review": rng.random() < 0.2}
        if rng.random() < 0.6:
            e["max_qty"] = str(int(e["min_qty"]) + rng.randint(0, 8))
        services.append(e)
    authority, tax, minimum = rng.choice([0, 5, 10, 15]), rng.choice(["0", "0.10", "0.15"]), rng.choice([0, 500, 5000])
    sched = parse_schedule({"currency": "AUD", "tax_rate": tax, "rules": {"discount_authority_percent": str(authority), "minimum_quote_total": str(minimum)}, "services": services})
    lines, wanted = [], []
    for _ in range(rng.randint(1, 7)):
        if rng.random() < 0.2:
            lines.append("%d x unknown thing %d" % (rng.randint(1, 3), rng.randint(0, 99)))
            wanted.append(None)
            continue
        s = rng.choice(services)
        qty = rng.choice([0, 1, 2, 3, 7, 11])
        name = rng.choice([s["name"], s["aliases"][0], s["name"].upper()])
        lines.append("%d x %s" % (qty, name) if rng.random() < 0.5 else "%s x %d" % (name, qty))
        wanted.append((s, qty))
    asked = rng.choice([0, 3, 10, 12, 20])
    q = draft("Client", "R", parse_enquiry("\n".join(lines)), sched, TEMPLATE, discount_percent=asked)
    # oracle: exact fractions, rounding half up to the cent at each line, the discount and the tax
    def cents(x):
        return Fraction(int((Decimal(x.numerator) / Decimal(x.denominator) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP)), 100)

    subtotal, expected_flags = Fraction(0), set()
    for n, w in enumerate(wanted, start=1):
        if w is None:
            expected_flags.add("L%d" % n)
            continue
        s, qty = w
        if qty == 0:
            expected_flags.add("L%d" % n)
            continue
        subtotal += cents(Fraction(qty) * Fraction(s["rate"]))
        if qty < int(s["min_qty"]) or ("max_qty" in s and qty > int(s["max_qty"])) or s["needs_review"]:
            expected_flags.add("L%d" % n)
    applied = min(Fraction(asked), Fraction(authority))
    discount = cents(subtotal * applied / 100)
    tax_amount = cents((subtotal - discount) * Fraction(tax))
    total = subtotal - discount + tax_amount
    if asked > authority:
        expected_flags.add("DISCOUNT")
    if total < minimum:
        expected_flags.add("MINIMUM")
    return q, expected_flags, (subtotal, discount, tax_amount, total)


@pytest.mark.parametrize("seed", range(80))
def test_seeded_worlds_match_an_oracle_that_shares_no_code_with_the_drafter(seed):
    q, expected_flags, (subtotal, discount, tax_amount, total) = build_world(seed)
    assert Fraction(str(q.subtotal)) == subtotal and Fraction(str(q.discount)) == discount and Fraction(str(q.tax)) == tax_amount and Fraction(str(q.total)) == total, seed
    assert {f.id for f in q.flags} == expected_flags, seed
    assert approve(q, "Dana Planner", [{"flag": f, "reason": "checked by the reviewer"} for f in expected_flags])["quote"]["quote_sha256"] == q.sha256


def test_the_80_worlds_reach_every_kind_of_flag_and_a_clean_draft():
    reached = set()
    for seed in range(80):
        _, expected_flags, _ = build_world(seed)
        reached |= {f if f in ("DISCOUNT", "MINIMUM") else "LINE" for f in expected_flags}
        if not expected_flags:
            reached.add("CLEAN")
    assert reached == {"DISCOUNT", "MINIMUM", "LINE", "CLEAN"}
