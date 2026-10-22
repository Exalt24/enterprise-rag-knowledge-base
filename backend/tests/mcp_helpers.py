"""Shared setup for the MCP server tests: an in-memory index of the synthetic corpus, a stub for answer generation, and
a way to call a tool through the SDK's in-process Client. Not collected as a test file (no test_ prefix).

No Qdrant, no language model key and no model download: the index is QdrantClient(":memory:") with the hashing embedder
(the same wiring the evaluation uses), and generation is replaced where a test needs an answer.
"""
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("RENDER", "true")
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pytest  # noqa: E402

# A bare `pytest` on a machine without the SDK skips these files instead of erroring; CI installs it first and would
# fail at the install step if it could not, so a skip there cannot hide a missing SDK.
pytest.importorskip("mcp", reason="the MCP SDK is not installed: pip install -r requirements-mcp.txt")

from eval.embedders import HashingEmbeddings  # noqa: E402
from eval.harness import chunk_corpus, in_memory_store, load_corpus  # noqa: E402
from mcp import Client  # noqa: E402

from app.services.generation import GenerationResponse  # noqa: E402
from mcp_server import server  # noqa: E402

CHUNKS = chunk_corpus(load_corpus())
# The ground truth for citation ids, built from the chunker and never from the server: id -> chunk text.
TEXT_BY_ID = {f"{c.metadata['file_name']}#{c.metadata['chunk_index']}": c.page_content for c in CHUNKS}
DOCUMENTS = sorted({c.metadata["file_name"] for c in CHUNKS})


@pytest.fixture(scope="module")
def demo_index():
    """The whole synthetic corpus in the repo's real VectorStoreService on an in-memory Qdrant."""
    with in_memory_store(HashingEmbeddings(), CHUNKS):
        yield


@pytest.fixture()
def empty_index():
    """The same wiring with no chunk in it."""
    with in_memory_store(HashingEmbeddings(), []):
        yield


class StubGeneration:
    """Stands in for generation_service.generate: records what it was given, returns a fixed model name and answer."""

    def __init__(self, model_used="stub/test", answer="Stub answer."):
        self.model_used, self.answer, self.calls = model_used, answer, []

    def __call__(self, query, context):
        self.calls.append((query, context))
        return GenerationResponse(answer=self.answer, query=query, model_used=self.model_used, context_length=len(context))


@pytest.fixture()
def stub_generation(monkeypatch):
    stub = StubGeneration()
    monkeypatch.setattr(server.generation_service, "generate", stub)
    return stub


@pytest.fixture()
def failing_generation(monkeypatch):
    """What generation_service.generate really returns when every provider fails: a normal response, model_used "none"."""
    stub = StubGeneration(model_used="none", answer="[X] All LLM providers unavailable. Please check your configuration.")
    monkeypatch.setattr(server.generation_service, "generate", stub)
    return stub


async def _call(name, arguments):
    async with Client(server.mcp) as client:
        return await client.call_tool(name, arguments)


def call(name, arguments=None):
    """Call a tool through the SDK's in-process Client and return its CallToolResult."""
    return asyncio.run(_call(name, arguments or {}))


async def _tools():
    async with Client(server.mcp) as client:
        return (await client.list_tools()).tools


def list_tools():
    return asyncio.run(_tools())


def error_text(result):
    """The message a model would read from an error result."""
    assert result.is_error, f"expected an error result, got {result.structured_content}"
    return " ".join(getattr(part, "text", "") for part in result.content)
