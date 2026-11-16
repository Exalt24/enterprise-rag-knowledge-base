"""The report, the human approval checkpoint, the command line, the HTTP route and the committed sample documents."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.compare.__main__ import main
from app.compare.approval import approve, verify_approved
from app.compare.check import compare
from app.compare.pages import CompareError, read_pdf
from app.compare.report import canonical, render_text, report_from_dict
from app.compare.requirements import parse_requirements
from compare_pdfs import make_pdf

BACKEND = Path(__file__).resolve().parents[1]
SAMPLE = BACKEND.parent / "docs" / "compare-sample"
sys.path.insert(0, str(SAMPLE))
import make_samples  # noqa: E402


def build_report(blocking_ok=False):
    a = read_pdf(make_pdf([["Wind speed: 45 m/s"]]), "a.pdf")
    b = read_pdf(make_pdf([["Wind speed: %s m/s" % ("45" if blocking_ok else "40"), "Status: DRAFT"]]), "b.pdf")
    rs = parse_requirements({"requirements": [
        {"id": "R1", "title": "wind", "kind": "match", "severity": "blocking", "pattern": "wind speed:? (?P<value>[0-9.]+) (?P<unit>m/s)"},
        {"id": "R2", "title": "no draft", "kind": "forbidden", "severity": "major", "document": "b", "phrase": "DRAFT"},
        {"id": "R3", "title": "revision", "kind": "present", "severity": "blocking", "document": "a", "phrase": "Revision"},
    ]})
    return compare(a, b, rs, "1" * 64)


# ---------------------------------------------------------------- the report

def test_the_same_inputs_give_the_same_hash_and_a_changed_finding_gives_another():
    assert build_report().sha256 == build_report().sha256 and len(build_report().sha256) == 64
    assert build_report().sha256 != build_report(blocking_ok=True).sha256


def test_a_report_survives_json_and_a_touched_one_is_refused():
    report = build_report()
    again = report_from_dict(json.loads(json.dumps(report.to_dict())))
    assert again.sha256 == report.sha256 and again.findings == report.findings
    for edit in (lambda d: d["findings"][0].__setitem__("status", "met"), lambda d: d["findings"][0]["evidence"][0].__setitem__("page", 7),
                 lambda d: d["documents"]["a"].__setitem__("sha256", "0" * 64), lambda d: d.__setitem__("requirements_sha256", "2" * 64)):
        raw = report.to_dict()
        edit(raw)
        with pytest.raises(CompareError, match="hash does not match"):
            report_from_dict(raw)
    with pytest.raises(CompareError, match="not a report"):
        report_from_dict({"findings": []})


def test_the_text_report_names_every_finding_its_page_and_its_words():
    text = render_text(build_report())
    assert "R1  [blocking] DISCREPANCY" in text and 'A p.1: "Wind speed: 45 m/s"' in text and 'B p.1: "Wind speed: 40 m/s Status: DRAFT"' in text
    assert "open blocking: R1, R3" in text


# ---------------------------------------------------------------- the approval checkpoint

GOOD = [{"finding": "R1", "decision": "accept"}, {"finding": "R3", "decision": "waive", "reason": "the revision is on the cover sheet"}]


def test_every_open_blocking_finding_must_be_decided_before_approval():
    report = build_report()
    assert approve(report, "Dana Planner", GOOD)["reviewer"] == "Dana Planner", "positive control"
    with pytest.raises(CompareError, match="blocking findings with no decision: R1, R3"):
        approve(report, "Dana Planner", [])
    with pytest.raises(CompareError, match="no decision: R3"):
        approve(report, "Dana Planner", GOOD[:1])


def test_a_major_finding_needs_no_decision_and_a_decision_on_it_is_allowed():
    report = build_report()
    assert [f.id for f in report.findings if f.status != "met" and not f.open_blocking] == ["R2"]
    assert [d["finding"] for d in approve(report, "Dana Planner", GOOD)["decisions"]] == ["R1", "R3"], "R2 is major and open, and approval does not wait for it"
    extra = GOOD + [{"finding": "R2", "decision": "reject", "reason": "the draft stamp is the template footer"}]
    assert [d["finding"] for d in approve(report, "Dana Planner", extra)["decisions"]] == ["R1", "R2", "R3"]


@pytest.mark.parametrize("decisions, match", [
    (GOOD + [{"finding": "R9", "decision": "accept"}], "no such finding"),
    (GOOD + [{"finding": "R1", "decision": "accept"}], "decided twice"),
    ([{"finding": "R1", "decision": "ignore"}, GOOD[1]], "accept, reject or waive"),
    ([{"finding": "R1", "decision": "reject"}, GOOD[1]], "needs a reason"),
    ([{"finding": "R1", "decision": "reject", "reason": "too short"}, GOOD[1]], "needs a reason"),
    ([GOOD[0], {"finding": "R3", "decision": "waive", "reason": ""}], "needs a reason"),
    ([GOOD[0], {"finding": "R3"}], "accept, reject or waive"),
    ([{"decision": "accept"}] + GOOD, "names no finding"),
    (["accept R1"] + GOOD, "names no finding"),
])
def test_a_decision_that_breaks_a_rule_is_refused_and_the_good_set_is_not(decisions, match):
    report = build_report()
    assert approve(report, "Dana Planner", GOOD)
    with pytest.raises(CompareError, match=match):
        approve(report, "Dana Planner", decisions)


def test_a_decision_about_a_met_finding_is_refused():
    report = build_report(blocking_ok=True)
    ok = [{"finding": "R3", "decision": "accept"}]
    assert approve(report, "Dana Planner", ok)
    with pytest.raises(CompareError, match="nothing to decide"):
        approve(report, "Dana Planner", ok + [{"finding": "R1", "decision": "accept"}])


@pytest.mark.parametrize("name", ["", " ", "D", None, 7])
def test_a_reviewer_must_be_a_name(name):
    with pytest.raises(CompareError, match="reviewer"):
        approve(build_report(), name, GOOD)


def test_all_problems_are_named_at_once():
    with pytest.raises(CompareError) as caught:
        approve(build_report(), "", [{"finding": "R9", "decision": "accept"}])
    text = str(caught.value)
    assert "reviewer" in text and "no such finding" in text and "blocking findings with no decision" in text


def test_an_intact_approval_verifies_and_every_kind_of_touching_is_found():
    report = build_report()
    approved = approve(report, "Dana Planner", GOOD, "2026-11-17T00:00:00Z")
    assert verify_approved(json.loads(json.dumps(approved))).sha256 == report.sha256

    def touched(edit):
        raw = copy.deepcopy(approved)
        edit(raw)
        return raw

    edits = {
        "a finding flipped": lambda r: r["report"]["findings"][0].__setitem__("status", "met"),
        "a decision changed": lambda r: r["decisions"][0].__setitem__("decision", "waive"),
        "a decision removed": lambda r: r["decisions"].pop(),
        "the reviewer renamed": lambda r: r.__setitem__("reviewer", "Someone Else"),
        "the approval hash replaced": lambda r: r.__setitem__("approval_sha256", "0" * 64),
        "report and hash both rewritten": lambda r: (r["report"]["findings"][0].__setitem__("status", "met"),
                                                     r["report"].__setitem__("report_sha256", hashlib.sha256(canonical({k: v for k, v in r["report"].items() if k not in ("summary", "report_sha256")})).hexdigest())),
    }
    for name, edit in edits.items():
        with pytest.raises(CompareError):
            verify_approved(touched(edit))
        assert name


def test_a_forged_approval_with_a_consistent_hash_and_a_missing_decision_still_fails():
    from app.compare.approval import _approval_hash

    report = build_report()
    approved = approve(report, "Dana Planner", GOOD, "2026-11-17T00:00:00Z")
    assert verify_approved(copy.deepcopy(approved)), "positive control"
    forged = copy.deepcopy(approved)
    forged["decisions"] = GOOD[:1]
    forged["approval_sha256"] = _approval_hash(report.sha256, "Dana Planner", forged["decisions"])
    with pytest.raises(CompareError, match="no decision: R3"):
        verify_approved(forged)


def test_approved_at_is_outside_the_hash_so_it_may_differ_between_runs():
    report = build_report()
    one, two = approve(report, "Dana Planner", GOOD, "2026-11-17T00:00:00Z"), approve(report, "Dana Planner", GOOD, "2026-11-18T09:30:00Z")
    assert one["approval_sha256"] == two["approval_sha256"] and one["report"] == two["report"]
    assert approve(report, "Dana Planner", GOOD)["approved_at"].endswith("Z")


# ---------------------------------------------------------------- the command line

def write(tmp, name, data):
    path = tmp / name
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    return str(path)


def test_the_command_line_compares_reviews_and_verifies_end_to_end(tmp_path, capsys):
    files = make_samples.build()
    a, b = write(tmp_path, "a.pdf", files["design_basis.pdf"]), write(tmp_path, "b.pdf", files["calculation_report.pdf"])
    req, dec = write(tmp_path, "r.json", files["requirements.json"]), write(tmp_path, "d.json", files["decisions.json"])
    out, txt, approved = str(tmp_path / "report.json"), str(tmp_path / "report.txt"), str(tmp_path / "approved.json")
    assert main(["compare", a, b, "--requirements", req, "--out", out, "--text", txt]) == 0
    assert "open blocking: R1, R3, R4, R5" in capsys.readouterr().out and Path(txt).read_text(encoding="utf-8").startswith("QA REPORT")
    assert main(["compare", a, b, "--requirements", req, "--out", out, "--strict"]) == 1
    capsys.readouterr()
    assert main(["review", out, "--reviewer", "Dana Planner", "--decisions", dec, "--out", approved]) == 0
    assert main(["verify", approved]) == 0 and "approval intact" in capsys.readouterr().out
    raw = json.loads(Path(approved).read_text(encoding="utf-8"))
    raw["decisions"][0]["decision"] = "waive"
    Path(approved).write_text(json.dumps(raw), encoding="utf-8")
    assert main(["verify", approved]) == 2 and "changed after approval" in capsys.readouterr().err


def test_a_review_with_an_open_blocking_finding_writes_nothing_and_exits_2(tmp_path, capsys):
    files = make_samples.build()
    report = write(tmp_path, "report.json", files["report.json"])
    partial = write(tmp_path, "d.json", {"decisions": [{"finding": "R1", "decision": "accept"}]})
    target = tmp_path / "approved.json"
    assert main(["review", report, "--reviewer", "Dana Planner", "--decisions", partial, "--out", str(target)]) == 2
    assert not target.exists() and "blocking findings with no decision: R3, R4, R5" in capsys.readouterr().err


@pytest.mark.parametrize("args, match", [
    (["compare", "{missing}", "{b}", "--requirements", "{req}", "--out", "{out}"], "cannot open"),
    (["compare", "{req}", "{b}", "--requirements", "{req}", "--out", "{out}"], "not a PDF"),
    (["compare", "{a}", "{b}", "--requirements", "{missing}", "--out", "{out}"], "cannot open"),
    (["compare", "{a}", "{b}", "--requirements", "{notjson}", "--out", "{out}"], "not valid JSON"),
    (["review", "{notjson}", "--reviewer", "Dana Planner", "--decisions", "{notjson}", "--out", "{out}"], "not valid JSON"),
    (["verify", "{missing}"], "cannot open"),
])
def test_a_file_that_cannot_be_used_is_exit_2_with_a_reason(tmp_path, capsys, args, match):
    files = make_samples.build()
    sub = {"a": write(tmp_path, "a.pdf", files["design_basis.pdf"]), "b": write(tmp_path, "b.pdf", files["calculation_report.pdf"]),
           "req": write(tmp_path, "r.json", files["requirements.json"]), "notjson": write(tmp_path, "n.json", b"{not json"),
           "missing": str(tmp_path / "nope"), "out": str(tmp_path / "out.json")}
    assert main(["compare", sub["a"], sub["b"], "--requirements", sub["req"], "--out", sub["out"]]) == 0, "positive control"
    assert main([x.format(**sub) for x in args]) == 2
    assert match in capsys.readouterr().err


def test_the_module_runs_as_a_program(tmp_path):
    files = make_samples.build()
    a, b, req = (write(tmp_path, "a.pdf", files["design_basis.pdf"]), write(tmp_path, "b.pdf", files["calculation_report.pdf"]), write(tmp_path, "r.json", files["requirements.json"]))
    done = subprocess.run([sys.executable, "-m", "app.compare", "compare", a, b, "--requirements", req, "--out", str(tmp_path / "o.json")], cwd=BACKEND,
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0 and "met 3, discrepancy 3, missing 1, unclear 2" in done.stdout


# ---------------------------------------------------------------- the committed samples

def test_the_sample_files_on_disk_are_exactly_what_the_script_writes():
    for name, data in make_samples.build().items():
        assert (SAMPLE / name).read_bytes() == data, name


def test_the_sample_says_what_the_readme_says_and_its_approval_is_intact():
    report = report_from_dict(json.loads((SAMPLE / "report.json").read_text(encoding="utf-8")))
    assert {f.id: f.status for f in report.findings} == {"R1": "discrepancy", "R2": "met", "R3": "discrepancy", "R4": "discrepancy", "R5": "unclear",
                                                         "R6": "met", "R7": "met", "R8": "missing", "R9": "unclear"}
    assert report.documents["b"].unreadable_pages == (3,)
    assert verify_approved(json.loads((SAMPLE / "approved.json").read_text(encoding="utf-8"))).sha256 == report.sha256


# ---------------------------------------------------------------- the HTTP route

@pytest.fixture()
def http():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.compare.api import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def post(http, a=None, b=None, requirements=None):
    files = make_samples.build()
    return http.post("/api/compare", files={"document_a": ("a.pdf", a if a is not None else files["design_basis.pdf"], "application/pdf"),
                                             "document_b": ("b.pdf", b if b is not None else files["calculation_report.pdf"], "application/pdf")},
                     data={"requirements": requirements if requirements is not None else files["requirements.json"].decode()})


def test_the_route_answers_the_same_report_as_the_command_line(http):
    files = make_samples.build()
    answer = post(http)
    assert answer.status_code == 200
    body = answer.json()
    assert {f["id"]: f["status"] for f in body["findings"]} == {"R1": "discrepancy", "R2": "met", "R3": "discrepancy", "R4": "discrepancy", "R5": "unclear",
                                                                "R6": "met", "R7": "met", "R8": "missing", "R9": "unclear"}
    assert body["documents"]["a"]["name"] == "a.pdf" and body["requirements_sha256"] == hashlib.sha256(files["requirements.json"]).hexdigest()
    assert report_from_dict(body).sha256 == body["report_sha256"]


@pytest.mark.parametrize("kwargs, match", [
    ({"a": b"not a pdf"}, "not a PDF"),
    ({"b": b""}, "empty"),
    ({"requirements": "{not json"}, "not valid JSON"),
    ({"requirements": '{"requirements": []}'}, "at least one"),
])
def test_a_file_or_requirements_that_cannot_be_used_is_a_422_with_the_reason(http, kwargs, match):
    assert post(http).status_code == 200, "positive control"
    answer = post(http, **kwargs)
    assert answer.status_code == 422 and match in answer.json()["detail"]


def test_an_oversized_upload_or_requirements_text_is_a_413(http, monkeypatch):
    assert post(http).status_code == 200
    monkeypatch.setattr("app.compare.api.MAX_BYTES", 100)
    assert post(http).status_code == 413
    monkeypatch.undo()
    monkeypatch.setattr("app.compare.api.MAX_REQUIREMENTS_BYTES", 50)
    assert post(http).status_code == 413


def test_the_main_app_mounts_the_route():
    text = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
    assert "from app.compare.api import router as compare_router" in text and 'app.include_router(compare_router, prefix="/api"' in text
