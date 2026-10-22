"""Every tool against every kind of failure, generated.

The server has three tools and four ways a call can go wrong. Writing a test per pair by hand would miss the pair nobody
thought of, so the cases below are the product of the two lists and each pair has an expected outcome: a readable
ToolError, a normal empty result, a normal result (the failure does not touch this tool), or not applicable (the tool
has no such input, which the test proves from its schema instead of skipping the pair).
Run: pytest tests/test_mcp_server_classes.py
"""
import itertools

import pytest

from eval.embedders import HashingEmbeddings
from eval.harness import in_memory_store
from mcp_helpers import CHUNKS, StubGeneration, call, error_text, list_tools
from mcp_server import server

# CLASS-TABLE: tool x failure kind; the expected outcome of each pair is written once in EXPECT
TOOLS = ("search_documents", "ask_question", "list_documents")
KINDS = ("empty_query", "unknown_document", "generation_unavailable", "index_empty")
CASES = list(itertools.product(TOOLS, KINDS))

ERROR, EMPTY, NORMAL, NOT_APPLICABLE = "tool_error", "empty_result", "normal_result", "not_applicable"
EXPECT = {
    ("search_documents", "empty_query"): (ERROR, "query is empty"),
    ("ask_question", "empty_query"): (ERROR, "question is empty"),
    ("list_documents", "empty_query"): (NOT_APPLICABLE, "query"),
    ("search_documents", "unknown_document"): (ERROR, "Unknown document"),
    ("ask_question", "unknown_document"): (NOT_APPLICABLE, "document"),
    ("list_documents", "unknown_document"): (NOT_APPLICABLE, "document"),
    ("search_documents", "generation_unavailable"): (NORMAL, None),
    ("ask_question", "generation_unavailable"): (ERROR, "unavailable"),
    ("list_documents", "generation_unavailable"): (NORMAL, None),
    ("search_documents", "index_empty"): (EMPTY, "no documents"),
    ("ask_question", "index_empty"): (EMPTY, "no documents"),
    ("list_documents", "index_empty"): (EMPTY, "no documents"),
}

ARGUMENTS = {
    ("search_documents", "empty_query"): {"query": "   "},
    ("ask_question", "empty_query"): {"question": ""},
    ("search_documents", "unknown_document"): {"query": "risk", "document": "missing.md"},
    ("search_documents", "generation_unavailable"): {"query": "risk"},
    ("ask_question", "generation_unavailable"): {"question": "who approves a change request"},
    ("list_documents", "generation_unavailable"): {},
    ("search_documents", "index_empty"): {"query": "risk"},
    ("ask_question", "index_empty"): {"question": "who approves a change request"},
    ("list_documents", "index_empty"): {},
}


def test_the_table_has_an_expectation_for_every_pair():
    assert len(CASES) == len(TOOLS) * len(KINDS)
    assert set(EXPECT) == set(CASES)
    assert {outcome for outcome, _ in EXPECT.values()} <= {ERROR, EMPTY, NORMAL, NOT_APPLICABLE}
    # every pair that can be run has arguments, and the pairs without arguments are exactly the not applicable ones
    assert {c for c in CASES if c not in ARGUMENTS} == {c for c, (o, _) in EXPECT.items() if o == NOT_APPLICABLE}


def _is_empty_result(tool, data):
    if tool == "search_documents":
        return data["hits"] == []
    if tool == "ask_question":
        return data["citations"] == [] and data["model_used"] is None
    return data["documents"] == [] and data["total_documents"] == 0


@pytest.mark.parametrize("tool,kind", CASES, ids=lambda v: v)
def test_each_tool_ends_each_failure_the_way_the_table_says(tool, kind, monkeypatch):
    outcome, expected_text = EXPECT[(tool, kind)]
    generate = StubGeneration()  # a working model, unless the kind says it is down
    if kind == "generation_unavailable":
        generate = StubGeneration(model_used="none", answer="[X] All LLM providers unavailable. Please check your configuration.")
    monkeypatch.setattr(server.generation_service, "generate", generate)
    # the index is full except for the "index_empty" kind
    with in_memory_store(HashingEmbeddings(), [] if kind == "index_empty" else CHUNKS):
        _check(tool, kind, outcome, expected_text, generate)


def _check(tool, kind, outcome, expected_text, generate):
    if outcome == NOT_APPLICABLE:
        schema = next(t for t in list_tools() if t.name == tool).input_schema
        assert expected_text not in schema.get("properties", {}), f"{tool} takes a {expected_text!r} input, so the pair applies"
        return

    result = call(tool, ARGUMENTS[(tool, kind)])
    if outcome == ERROR:
        message = error_text(result)
        assert expected_text in message
        assert 10 < len(message) < 500 and "Traceback" not in message and "[X]" not in message
        assert result.structured_content is None
    else:
        assert not result.is_error, f"{tool} failed on {kind}: {result.content}"
        data = result.structured_content
        if outcome == EMPTY:
            assert _is_empty_result(tool, data)
            assert expected_text in data.get("message", data.get("answer", "")).lower()
            assert not generate.calls, "no model should be called when there is nothing to answer from"
        else:
            assert not _is_empty_result(tool, data)
