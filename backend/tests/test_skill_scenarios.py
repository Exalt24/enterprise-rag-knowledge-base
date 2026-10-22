"""The written evaluation scenarios of the querying-knowledge-base skill, checked against the real server.

A scenario is a user request, the tool calls a correct agent makes, and the failure it must avoid. These files were
written BEFORE SKILL.md so the skill documents what the scenarios need. No agent is run here: nothing in this file
shows that a model follows the skill. What it does show is that each scenario is well formed, names only tools,
arguments and fields the server really has, and can be satisfied by the data: replaying the listed calls on the demo
index returns the chunks the scenario says a correct answer cites, and the facts it expects are in those chunks.
Run: pytest tests/test_skill_scenarios.py
"""
import itertools
import json
import re
import uuid

import pytest
from langchain_core.documents import Document

from eval.embedders import HashingEmbeddings
from eval.harness import in_memory_store
from mcp_helpers import BACKEND, CHUNKS, TEXT_BY_ID, StubGeneration, call, list_tools
from mcp_server import server

SKILL_DIR = BACKEND.parent / "skills" / "querying-knowledge-base"
SCENARIO_FILES = sorted((SKILL_DIR / "evaluations").glob("*.md"))
SECTIONS = ("User request", "A correct agent", "Failures this scenario catches", "Machine-checked")
ID_PATTERN = re.compile(r"^[^#\s]+#\d+$")
SNAKE_TOKEN = re.compile(r"`([a-z]+(?:_[a-z]+)+)`")
FENCE = re.compile(r"```json\n(.*?)\n```", re.S)


def load(path):
    text = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (.+)$", text, flags=re.M))
    sections = {}
    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        sections[h.group(1).strip()] = text[h.end() : end].strip()
    block = FENCE.search(text)
    data = json.loads(block.group(1)) if block else None
    prose = FENCE.sub("", text)
    return {"path": path, "text": text, "prose": prose, "sections": sections, "data": data}


SCENARIOS = {p.stem: load(p) for p in SCENARIO_FILES}


def tool_schemas():
    tools = {t.name: t for t in list_tools()}
    fields = set()
    for tool in tools.values():
        fields |= set((tool.input_schema or {}).get("properties", {}))
        schema = tool.output_schema or {}
        for node in [schema, *schema.get("$defs", {}).values()]:
            fields |= set(node.get("properties", {}))
    return tools, fields


def chunk_texts(scenario):
    texts = dict(TEXT_BY_ID)
    for extra in scenario["data"]["extra_chunks"]:
        texts[f"{extra['file_name']}#{extra['chunk_index']}"] = extra["text"]
    return texts


# ---------------------------------------------------------------- the set as a whole


def test_there_are_at_least_three_scenarios_and_each_has_a_distinct_id():
    assert len(SCENARIOS) >= 3
    ids = [s["data"]["id"] for s in SCENARIOS.values() if s["data"]]
    assert len(ids) == len(set(ids)) == len(SCENARIOS)


def test_the_scenarios_cover_the_failure_paths_the_skill_exists_for():
    all_calls = [c for s in SCENARIOS.values() for c in s["data"]["calls"]]
    outcomes = {c["outcome"] for c in all_calls}
    assert {"answer", "hits", "no_relevant_hits", "error", "documents"} <= outcomes
    assert {c["tool"] for c in all_calls} == {"search_documents", "ask_question", "list_documents"}
    assert any(s["data"]["extra_chunks"] for s in SCENARIOS.values()), "no scenario covers a document that gives orders"
    assert any(not s["data"]["cite"] for s in SCENARIOS.values()), "no scenario covers an answer with nothing to cite"


# ---------------------------------------------------------------- CLASS-TABLE: scenario x required element


def _request(s):
    return [] if len(s["sections"].get("User request", "").strip("> \n")) > 15 else ["the user request is missing or empty"]


def _correct_agent(s):
    text = s["sections"].get("A correct agent", "")
    return [] if re.search(r"^1\. ", text, flags=re.M) and "`" in text else ["no numbered steps that name a tool"]


def _failures(s):
    items = re.findall(r"^- .{12,}", s["sections"].get("Failures this scenario catches", ""), flags=re.M)
    return [] if len(items) >= 2 else ["fewer than two failures to avoid are listed"]


def _machine_block(s):
    data = s["data"]
    needed = {"id", "skill_section", "calls", "cite", "facts", "absent_terms", "extra_chunks"}
    if data is None:
        return ["no json block"]
    problems = [f"missing key {k}" for k in sorted(needed - set(data))]
    for c in data.get("calls", []):
        problems += [f"a call lacks {k}" for k in ("tool", "arguments", "outcome", "expect_ids") if k not in c]
    return problems


def _only_real_tools(s):
    tools, fields = tool_schemas()
    problems = []
    for token in SNAKE_TOKEN.findall(s["prose"]):
        if token not in tools and token not in fields:
            problems.append(f"`{token}` is neither a tool nor a field of one")
    for c in s["data"]["calls"]:
        if c["tool"] not in tools:
            problems.append(f"call to unknown tool {c['tool']}")
            continue
        schema = tools[c["tool"]].input_schema or {}
        problems += [f"{c['tool']} has no argument {a}" for a in c["arguments"] if a not in schema.get("properties", {})]
        problems += [f"{c['tool']} requires {r}" for r in schema.get("required", []) if r not in c["arguments"]]
    return problems


def _ids_well_formed(s):
    d = s["data"]
    ids = list(d["cite"]) + [i for c in d["calls"] for i in c["expect_ids"]]
    return [f"{i!r} is not file_name#chunk_index" for i in ids if not ID_PATTERN.match(i)]


def _cites_only_what_a_call_returns(s):
    d = s["data"]
    returned = {i for c in d["calls"] if c["outcome"] not in ("error", "no_relevant_hits") for i in c["expect_ids"]}
    problems = [f"cites {i}, which no call is expected to return" for i in d["cite"] if i not in returned]
    if d["absent_terms"] and d["cite"]:
        problems.append("a scenario with absent terms must cite nothing")
    return problems


def _skill_section_named(s):
    return [] if s["data"]["skill_section"].strip() else ["skill_section is empty"]


def _facts_sit_in_cited_chunks(s):
    d = s["data"]
    texts = chunk_texts(s)
    cited = " ".join(texts.get(i, "") for i in d["cite"])
    problems = [f"cited id {i} is not a chunk" for i in d["cite"] if i not in texts]
    problems += [f"fact {f!r} is not in any cited chunk" for f in d["facts"] if f not in cited]
    if d["cite"] and not d["facts"]:
        problems.append("a scenario that cites needs at least one fact to check")
    return problems


ELEMENTS = {
    "user_request": _request,
    "correct_agent_steps": _correct_agent,
    "failures_to_avoid": _failures,
    "machine_block": _machine_block,
    "only_real_tools": _only_real_tools,
    "ids_well_formed": _ids_well_formed,
    "cites_only_what_a_call_returns": _cites_only_what_a_call_returns,
    "skill_section_named": _skill_section_named,
    "facts_sit_in_cited_chunks": _facts_sit_in_cited_chunks,
}
CASES = list(itertools.product(sorted(SCENARIOS), ELEMENTS))


@pytest.mark.parametrize("scenario,element", CASES, ids=lambda v: v)
def test_every_scenario_has_every_required_element(scenario, element):
    s = SCENARIOS[scenario]
    for section in SECTIONS:
        assert section in s["sections"], f"{scenario} has no '## {section}' section"
    problems = ELEMENTS[element](s)
    assert not problems, f"{scenario} / {element}: " + "; ".join(problems)


# ---------------------------------------------------------------- replaying the calls on the demo index


def _extra_documents(scenario):
    return [
        Document(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{e['file_name']}#{e['chunk_index']}")),  # the other chunks carry ids too
            page_content=e["text"],
            metadata={"file_name": e["file_name"], "chunk_index": e["chunk_index"], "file_type": "md"},
        )
        for e in scenario["data"]["extra_chunks"]
    ]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_replaying_the_calls_on_the_demo_index_returns_what_the_scenario_expects(scenario, monkeypatch):
    s = SCENARIOS[scenario]
    data = s["data"]
    corpus = CHUNKS + _extra_documents(s)
    known_documents = {c.metadata["file_name"] for c in corpus}
    with in_memory_store(HashingEmbeddings(), corpus):
        for step in data["calls"]:
            failing = step["outcome"] == "error"
            monkeypatch.setattr(
                server.generation_service,
                "generate",
                StubGeneration(model_used="none", answer="[X] All LLM providers unavailable.") if failing else StubGeneration(),
            )
            result = call(step["tool"], step["arguments"])
            where = f"{scenario}: {step['tool']} {step['arguments']}"
            if step["outcome"] == "error":
                assert result.is_error, f"{where} should fail but returned {result.structured_content}"
                continue
            assert not result.is_error, f"{where} failed: {result.content}"
            body = result.structured_content
            if step["outcome"] == "documents":
                names = {d["file_name"] for d in body["documents"]}
                used = {c["arguments"]["document"] for c in data["calls"] if "document" in c["arguments"]}
                assert used <= names, f"{where}: the document argument {used - names} is not in the list"
                continue
            hits = body["hits"] if step["tool"] == "search_documents" else body["citations"]
            returned = [h["id"] for h in hits]
            if step["outcome"] == "no_relevant_hits":
                text = " ".join(h["snippet"] for h in hits).lower()
                assert not [t for t in data["absent_terms"] if t.lower() in text], f"{where}: a hit mentions an absent term"
            else:
                missing = [i for i in step["expect_ids"] if i not in returned]
                assert not missing, f"{where}: expected {missing} among {returned}"
        if data["extra_chunks"]:
            assert all(t.annotations.read_only_hint for t in list_tools()), "a tool could act on the instruction in the chunk"
    assert known_documents  # the corpus was not empty
