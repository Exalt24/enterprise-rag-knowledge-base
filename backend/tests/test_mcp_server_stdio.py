"""The real server process over real stdio.

The MCP stdio transport reserves stdout for protocol messages, and this repo's services print progress lines to stdout
on import and on every call. These tests start `python -m mcp_server.server --demo-corpus` as a subprocess (in-memory
index of the synthetic corpus, no Qdrant, no model download), talk to it, and check that every byte on stdout is a
JSON-RPC message while the progress lines are on stderr.
Run: pytest tests/test_mcp_server_stdio.py
"""
import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from mcp_helpers import BACKEND, TEXT_BY_ID

DEADLINE_SECONDS = 120  # a cold start imports langchain and builds an index; the answer arrives well inside this
COMMAND = [sys.executable, "-m", "mcp_server.server", "--demo-corpus"]


def _env():
    return dict(os.environ, RENDER="true", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")


def _lines(stream, sink):
    for line in iter(stream.readline, b""):
        sink.put(line)


def test_stdout_holds_protocol_messages_and_nothing_else():
    proc = subprocess.Popen(COMMAND, cwd=BACKEND, env=_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = queue.Queue(), queue.Queue()
    for stream, sink in ((proc.stdout, out), (proc.stderr, err)):
        threading.Thread(target=_lines, args=(stream, sink), daemon=True).start()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "stdio-test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "search_documents", "arguments": {"query": "how is a risk escalated"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_documents", "arguments": {}}},
    ]
    stdout_lines, answered = [], set()
    try:
        proc.stdin.write(("\n".join(json.dumps(r) for r in requests) + "\n").encode())
        proc.stdin.flush()
        deadline = time.time() + DEADLINE_SECONDS
        while answered != {1, 2, 3, 4} and time.time() < deadline:
            try:
                line = out.get(timeout=0.25)
            except queue.Empty:
                continue
            stdout_lines.append(line)
            message = json.loads(line)  # a non-JSON line on stdout raises here, naming the line in the traceback
            if "id" in message:
                answered.add(message["id"])
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    while not out.empty():
        stdout_lines.append(out.get())
    stderr_text = b"".join(list(err.queue)).decode("utf-8", "replace")

    assert answered == {1, 2, 3, 4}, f"no answer to request(s) {sorted({1, 2, 3, 4} - answered)}; stderr said: {stderr_text[-400:]}"
    for line in stdout_lines:
        message = json.loads(line)
        assert message.get("jsonrpc") == "2.0", f"stdout holds something that is not a protocol message: {line[:120]!r}"
    # not vacuous: the services really did print, and it went to stderr
    assert "[i]" in stderr_text, "the services print progress lines; none reached stderr, so this test proved nothing"
    search = next(json.loads(l) for l in stdout_lines if json.loads(l).get("id") == 3)
    assert not search["result"].get("isError"), search
    hits = search["result"]["structuredContent"]["hits"]
    assert hits and all(h["id"] in TEXT_BY_ID for h in hits)


async def _through_the_sdk_client():
    params = StdioServerParameters(command=sys.executable, args=COMMAND[1:], cwd=str(BACKEND), env={"RENDER": "true", "PYTHONUTF8": "1"})
    async with Client(params) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        search = await client.call_tool("search_documents", {"query": "who signs off a change", "limit": 3})
        ask = await client.call_tool("ask_question", {"question": "who signs off a change"})
        return tools, search, ask


def test_the_sdk_client_can_use_the_server_over_stdio():
    tools, search, ask = asyncio.run(asyncio.wait_for(_through_the_sdk_client(), DEADLINE_SECONDS))
    assert tools == {"search_documents", "ask_question", "list_documents"}
    assert not search.is_error and search.structured_content["hits"]
    assert all(h["id"] in TEXT_BY_ID for h in search.structured_content["hits"])
    # with no language model configured the demo server must fail the way the contract says, not return error text
    from app.services.generation import generation_service

    if generation_service.primary_llm is None:
        assert ask.is_error and "unavailable" in " ".join(getattr(p, "text", "") for p in ask.content)
