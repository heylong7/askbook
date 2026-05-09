"""Integration tests for the askbook MCP stdio server.

Tests use subprocess to launch `askbook serve` and communicate via newline-delimited
JSON-RPC 2.0 messages (matching the mcp.server.stdio transport format).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

TIMEOUT = 20  # seconds — generous for startup on Windows
CWD = str(Path(__file__).parents[2])  # repo root: tests/integration/../../


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(tmp_path: Path) -> dict[str, str]:
    """Build subprocess environment with stub providers."""
    return {
        **os.environ,
        "ASKBOOK_LLM__PROVIDER": "stub",
        "ASKBOOK_EMBEDDING__PROVIDER": "stub",
        "ASKBOOK_VECTORSTORE__PATH": str(tmp_path / "chroma"),
        "ASKBOOK_DATA_DIR": str(tmp_path),
        "PYTHONIOENCODING": "utf-8",
    }


def _launch_server(tmp_path: Path) -> subprocess.Popen[bytes]:
    """Launch `askbook serve` subprocess with stub providers."""
    return subprocess.Popen(
        ["uv", "run", "askbook", "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_make_env(tmp_path),
        cwd=CWD,
    )


def _send(proc: subprocess.Popen[bytes], obj: dict[str, Any]) -> None:
    """Write one JSON-RPC message line to the server stdin."""
    line = json.dumps(obj) + "\n"
    assert proc.stdin is not None
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def _recv(proc: subprocess.Popen[bytes], timeout: float = TIMEOUT) -> dict[str, Any]:
    """Read one JSON-RPC message line from the server stdout."""
    assert proc.stdout is not None
    # Set a deadline using poll + readline approach
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check if process has died
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited unexpectedly with code {proc.returncode}"
            )
        line = proc.stdout.readline()
        if line:
            return json.loads(line.decode("utf-8"))
        time.sleep(0.05)
    raise TimeoutError(f"No response from server within {timeout}s")


def _do_initialize_handshake(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Perform the MCP initialize handshake and return the initialize result."""
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1"},
            },
        },
    )
    result = _recv(proc)

    # Send notifications/initialized (no response expected)
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mcp_stdio_initialize_and_list_tools(tmp_path: Path) -> None:
    """Server handshake completes and tools/list returns exactly 4 tools."""
    proc = _launch_server(tmp_path)
    try:
        # 1. Initialize handshake
        init_result = _do_initialize_handshake(proc)
        assert init_result.get("jsonrpc") == "2.0"
        assert init_result.get("id") == 1
        assert "result" in init_result, f"Expected result, got: {init_result}"

        # 2. Request tool list
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools_resp = _recv(proc)

        assert tools_resp.get("jsonrpc") == "2.0"
        assert tools_resp.get("id") == 2
        result = tools_resp.get("result", {})
        tools = result.get("tools", [])

        # Harness: exactly 7 tools (4 core + 3 diagnostic)
        tool_name_list = [t.get("name") for t in tools]
        assert len(tools) == 7, f"Expected 7 tools, got {len(tools)}: {tool_name_list}"

        tool_names = {t["name"] for t in tools}
        assert tool_names == {
            "search",
            "ask",
            "list_collections",
            "get_document_summary",
            "trace_lookup",
            "collection_stats",
        }, f"Unexpected tool names: {tool_names}"

        # Each tool must have name, description, inputSchema
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_stdio_call_list_collections_returns_tool_response(tmp_path: Path) -> None:
    """list_collections returns a valid ToolResponse JSON in content[0].text."""
    proc = _launch_server(tmp_path)
    try:
        _do_initialize_handshake(proc)

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_collections", "arguments": {}},
            },
        )
        resp = _recv(proc)

        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == 3
        result = resp.get("result", {})
        content = result.get("content", [])

        assert len(content) >= 1, f"Expected content, got: {content}"
        first = content[0]
        assert first.get("type") == "text", f"Expected text content, got: {first}"

        # Must be valid JSON matching ToolResponse shape
        tool_resp = json.loads(first["text"])
        assert "status" in tool_resp, f"Missing 'status' in: {tool_resp}"
        assert tool_resp["status"] in ("success", "warning", "error")
        assert "summary" in tool_resp
        assert "source_ids" in tool_resp

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_stdio_ask_empty_collection_returns_warning(tmp_path: Path) -> None:
    """ask with no ingested data returns warning (Harness 30.1.2 end-to-end)."""
    proc = _launch_server(tmp_path)
    try:
        _do_initialize_handshake(proc)

        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "ask",
                    "arguments": {"question": "test question", "collection": "demo"},
                },
            },
        )
        resp = _recv(proc)

        assert resp.get("jsonrpc") == "2.0"
        assert resp.get("id") == 4
        result = resp.get("result", {})
        content = result.get("content", [])
        assert len(content) >= 1

        tool_resp = json.loads(content[0]["text"])
        assert tool_resp["status"] == "warning", (
            f"Expected warning for empty collection, got: {tool_resp['status']}"
        )
        assert tool_resp["source_ids"] == [], (
            f"Expected empty source_ids, got: {tool_resp['source_ids']}"
        )

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mcp_stdio_logs_go_to_stderr_not_stdout(tmp_path: Path) -> None:
    """All stdout lines are valid JSON-RPC; logs must not leak to stdout."""
    proc = _launch_server(tmp_path)
    stdout_lines: list[str] = []
    try:
        # Do full sequence: initialize + notifications/initialized + tools/list
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.1"},
                },
            },
        )
        init_resp = _recv(proc)
        stdout_lines.append(json.dumps(init_resp))

        _send(
            proc,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools_resp = _recv(proc)
        stdout_lines.append(json.dumps(tools_resp))

        # Close stdin to signal EOF; give server a moment to flush
        assert proc.stdin is not None
        proc.stdin.close()
        time.sleep(0.5)

        # Read any remaining stdout output
        assert proc.stdout is not None
        remaining = proc.stdout.read()
        for raw_line in remaining.decode("utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if line:
                stdout_lines.append(line)

    finally:
        proc.terminate()
        proc.wait(timeout=5)

    # Every line captured from stdout must be valid JSON
    for line in stdout_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Non-JSON output on stdout (logs leaking?): {stripped!r}"
            ) from exc
        # Must be a JSON object (dict), not a bare string/int
        assert isinstance(parsed, dict), (
            f"Expected JSON object on stdout, got: {type(parsed)}"
        )
