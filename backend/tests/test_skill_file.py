"""The querying-knowledge-base skill file against the Agent Skills rules and against the server it describes.

Checks the format (frontmatter fields, name and description limits, body length, references one level deep), that
every tool, argument and field the skill names exists on the server, that every id in its examples is a real chunk,
that errors.md matches the messages the server really returns, and, for each written scenario, that the skill has the
section the scenario points at and names the tools the scenario uses. It does not show that a model follows the skill.
Run: pytest tests/test_skill_file.py
"""
import itertools
import re

import pytest

from eval.embedders import HashingEmbeddings
from eval.harness import in_memory_store
from mcp_helpers import CHUNKS, TEXT_BY_ID, StubGeneration, call, error_text, list_tools
from mcp_server import server
from test_skill_scenarios import SCENARIOS, SKILL_DIR, SNAKE_TOKEN, tool_schemas

SKILL = SKILL_DIR / "SKILL.md"
ERRORS = SKILL_DIR / "errors.md"
ALLOWED_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
RESERVED = ("anthropic", "claude")
PERSON = re.compile(r"\b(i|me|my|we|our|you|your)\b", re.I)


def split_skill(text):
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert match, "SKILL.md must open with a --- frontmatter block"
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        assert sep and key == key.strip(), f"frontmatter line is not 'key: value': {line!r}"
        fields[key.strip()] = value.strip()
    return fields, match.group(2)


FIELDS, BODY = split_skill(SKILL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the format


def test_frontmatter_has_only_the_allowed_fields_and_both_required_ones():
    assert set(FIELDS) <= ALLOWED_FIELDS, f"fields the spec does not allow: {sorted(set(FIELDS) - ALLOWED_FIELDS)}"
    assert FIELDS.get("name") and FIELDS.get("description")


def test_the_name_follows_the_naming_rules_and_the_folder():
    name = FIELDS["name"]
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name) and len(name) <= 64
    assert not any(word in name for word in RESERVED)
    assert name == SKILL_DIR.name, "the skill name must equal its folder name"


def test_the_description_says_what_and_when_in_the_third_person_within_the_limit():
    text = FIELDS["description"]
    assert 0 < len(text) <= 1024
    assert not re.search(r"<[^>]+>", text), "no XML tags in the description"
    assert not PERSON.search(text), f"the description must be third person, found {PERSON.search(text).group(0)!r}"
    assert re.search(r"\buse when\b", text, re.I), "the description must say when to use the skill"
    assert all(tool in text for tool in ("search_documents", "ask_question", "list_documents"))


def test_the_body_is_under_500_lines():
    assert len(BODY.splitlines()) < 500


def test_references_resolve_and_stay_one_level_deep():
    links = re.findall(r"\]\(([^)#]+)\)", BODY)
    assert links, "the skill is expected to link its reference file"
    for link in links:
        target = SKILL_DIR / link
        assert target.is_file(), f"SKILL.md links {link}, which does not exist"
        assert target.parent == SKILL_DIR, f"{link} is not directly beside SKILL.md"
        assert not re.findall(r"\]\((?!http)[^)]+\.md\)", target.read_text(encoding="utf-8")), f"{link} links to a further file"


# ---------------------------------------------------------------- the skill against the server


def test_the_skill_names_every_tool_and_only_real_tools_arguments_and_fields():
    tools, fields = tool_schemas()
    text = SKILL.read_text(encoding="utf-8") + ERRORS.read_text(encoding="utf-8")
    for name in tools:
        assert f"`{name}`" in BODY, f"the skill never names `{name}`"
    unknown = {t for t in SNAKE_TOKEN.findall(text) if t not in tools and t not in fields}
    assert not unknown, f"the skill names things the server does not have: {sorted(unknown)}"


def test_every_chunk_id_in_the_skill_is_a_real_chunk():
    ids = re.findall(r"[\w.-]+\.md#\d+", SKILL.read_text(encoding="utf-8") + ERRORS.read_text(encoding="utf-8"))
    assert ids
    assert [i for i in ids if i not in TEXT_BY_ID] == []


def test_the_example_in_the_skill_is_what_the_server_returns():
    example_query = re.search(r'query "([^"]+)", limit (\d+)', BODY)
    cited = re.search(r"`([\w.-]+\.md#\d+)`", BODY.split("## Example")[1]).group(1)
    with in_memory_store(HashingEmbeddings(), CHUNKS):
        hits = call("search_documents", {"query": example_query.group(1), "limit": int(example_query.group(2))}).structured_content["hits"]
    assert hits[0]["id"] == cited, "the example says the top hit is a chunk the server does not return first"
    assert "15 or more" in hits[0]["snippet"] and "Programme Board" in hits[0]["snippet"]


def test_the_citation_format_is_the_one_the_server_returns():
    assert "file_name#chunk_index" in BODY
    with in_memory_store(HashingEmbeddings(), CHUNKS):
        hit = call("search_documents", {"query": "risk", "limit": 1}).structured_content["hits"][0]
    assert hit["id"] == f"{hit['document']}#{hit['chunk_index']}"


# ---------------------------------------------------------------- errors.md against the real messages


def _real_messages(monkeypatch):
    messages = {}
    with in_memory_store(HashingEmbeddings(), CHUNKS):
        messages["empty query"] = error_text(call("search_documents", {"query": " "}))
        messages["empty question"] = error_text(call("ask_question", {"question": ""}))
        messages["too long"] = error_text(call("search_documents", {"query": "x" * 501}))
        messages["bad limit"] = error_text(call("search_documents", {"query": "risk", "limit": 99}))
        messages["unknown document"] = error_text(call("search_documents", {"query": "risk", "document": "nope.md"}))
        monkeypatch.setattr(server.generation_service, "generate", StubGeneration(model_used="none", answer="[X] down"))
        messages["generation down"] = error_text(call("ask_question", {"question": "risk escalation"}))
        monkeypatch.setattr(server.advanced_retrieval, "hybrid_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        messages["other failure"] = error_text(call("search_documents", {"query": "risk"}))
    monkeypatch.setattr(server.vector_store, "get_stats", lambda: (_ for _ in ()).throw(server.VectorStoreUnavailable("down")))
    messages["unreachable"] = error_text(call("search_documents", {"query": "risk"}))
    with in_memory_store(HashingEmbeddings(), []):
        messages["empty index"] = call("list_documents").structured_content["message"]
    return messages


def _rows():
    rows = []
    for line in ERRORS.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `"):
            first = line.split("|")[1]
            keys = [k.strip("` ") for k in first.split(" or ")]
            rows.append([[frag for frag in re.split(r" N | N\b|'x'", key) if frag.strip()] for key in keys])
    return rows


def test_every_real_error_message_has_a_row_in_errors_md(monkeypatch):
    messages = _real_messages(monkeypatch)
    rows = _rows()
    for label, message in messages.items():
        assert any(all(frag.strip() in message for frag in key) for row in rows for key in row), f"errors.md has no row for the {label} message: {message[:100]!r}"


def test_every_row_of_errors_md_matches_a_real_message(monkeypatch):
    messages = " || ".join(_real_messages(monkeypatch).values())
    for row in _rows():
        assert any(all(frag.strip() in messages for frag in key) for key in row), f"errors.md describes a message the server never returns: {row}"


def test_every_real_error_carries_the_tool_prefix_errors_md_describes(monkeypatch):
    messages = _real_messages(monkeypatch)
    errors = {label: text for label, text in messages.items() if label != "empty index"}  # the empty index is a normal result
    for label, text in errors.items():
        assert re.match(r"Error executing tool \w+: ", text), f"the {label} message no longer starts with the tool prefix: {text[:80]!r}"
    example = re.findall(r"`(Error executing tool \w+: [^`]+)`", ERRORS.read_text(encoding="utf-8"))
    assert example, "errors.md no longer shows a full error text with its prefix"
    joined = " || ".join(errors.values())
    for text in example:
        assert text in joined, f"errors.md quotes an error text the server never returns: {text!r}"


# CLASS-TABLE: scenario x what the skill must hold for it
REQUIREMENTS = ("section_exists", "scenario_tools_named_in_skill", "section_gives_at_least_two_instructions")
CASES = list(itertools.product(sorted(SCENARIOS), REQUIREMENTS))


@pytest.mark.parametrize("scenario,requirement", CASES, ids=lambda v: v)
def test_the_skill_holds_what_each_scenario_needs(scenario, requirement):
    data = SCENARIOS[scenario]["data"]
    if requirement == "section_exists":
        assert re.search(rf"^## {re.escape(data['skill_section'])}$", BODY, re.M), f"SKILL.md has no '## {data['skill_section']}' section for {scenario}"
    elif requirement == "scenario_tools_named_in_skill":
        for tool in {c["tool"] for c in data["calls"]}:
            assert f"`{tool}`" in BODY, f"{scenario} uses {tool}, which the skill never names"
    else:
        section = re.search(rf"^## {re.escape(data['skill_section'])}$(.*?)(?=^## |\Z)", BODY, re.M | re.S)
        assert section, f"no '## {data['skill_section']}' section"
        assert len(re.findall(r"^- ", section.group(1), re.M)) >= 2, f"the '{data['skill_section']}' section gives fewer than two instructions"
