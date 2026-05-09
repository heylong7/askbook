"""MCP stdio server for askbook (Layer 3 of 3).

Layer structure:
  contracts.py  — Pydantic I/O models
  tools.py      — Pure function handlers
  deps.py       — Startup dependency wiring
  server.py     — MCP SDK wrapping (this file)

stdout is JSON-RPC ONLY.  All logging goes to stderr.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from askbook.mcp_server.contracts import ToolResponse
from askbook.mcp_server.deps import build_server_deps
from askbook.mcp_server.tools import TOOL_REGISTRY, ServerDeps

_DESCRIPTIONS: dict[str, str] = {
    "search": (
        "Keyword+semantic search over an askbook collection."
        " Returns ranked snippets with chunk_ids."
    ),
    "ask": (
        "Ask a question against an askbook collection."
        " Returns a grounded answer with citations."
    ),
    "list_collections": ("List all ingested askbook collections with chunk counts."),
    "get_document_summary": (
        "Get chunk count, source path, and first snippets for a given doc_id."
    ),
    "get_chunk_content": (
        "Retrieve the full content of a single chunk by its chunk_id."
    ),
    "trace_lookup": (
        "Look up trace events by trace_id or list recent events."
    ),
    "collection_stats": (
        "Get detailed statistics for a collection (chunk count, doc count, disk size)."
    ),
}


def _describe(name: str) -> str:
    return _DESCRIPTIONS[name]


def _configure_stderr_logging() -> None:
    """Force all logging to stderr so stdout remains JSON-RPC only."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def build_server(deps: ServerDeps) -> Server:
    """Construct and configure the MCP Server instance with all tool handlers."""
    server: Server = Server("askbook")

    @server.list_tools()  # type: ignore
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=name,
                description=_describe(name),
                inputSchema=input_model.model_json_schema(),
            )
            for name, (input_model, _handler) in TOOL_REGISTRY.items()
        ]

    @server.call_tool()  # type: ignore
    async def _call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[mcp_types.TextContent]:
        if name not in TOOL_REGISTRY:
            err = ToolResponse(
                status="error",
                summary=f"Unknown tool {name!r}.",
                data={},
                source_ids=[],
            )
            return [mcp_types.TextContent(type="text", text=err.model_dump_json())]

        input_model, handler = TOOL_REGISTRY[name]
        try:
            inp = input_model.model_validate(arguments or {})
        except Exception as exc:
            err = ToolResponse(
                status="error",
                summary=f"Invalid arguments: {exc}",
                data={},
                source_ids=[],
            )
            return [mcp_types.TextContent(type="text", text=err.model_dump_json())]

        try:
            resp = await asyncio.to_thread(handler, inp, deps)
        except Exception as exc:
            logging.exception("Tool %r raised an unexpected error", name)
            resp = ToolResponse(
                status="error",
                summary=f"{type(exc).__name__}: {exc}",
                data={},
                source_ids=[],
            )

        return [mcp_types.TextContent(type="text", text=resp.model_dump_json())]

    return server


async def _amain(config_path: Path | None, collection_hint: str) -> None:
    _configure_stderr_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting askbook MCP server (collection_hint=%r)", collection_hint)

    deps = build_server_deps(config_path=config_path, collection_hint=collection_hint)
    server = build_server(deps)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("askbook MCP server shut down cleanly")


def run_server(
    *,
    config_path: Path | None = None,
    collection_hint: str = "default",
) -> None:
    """Synchronous entry point: start the async MCP stdio server."""
    asyncio.run(_amain(config_path=config_path, collection_hint=collection_hint))


__all__ = ["build_server", "run_server"]
