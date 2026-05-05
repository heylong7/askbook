"""Integration tests for diagnostic MCP tools."""

from __future__ import annotations

from askbook.mcp_server.tools import TOOL_REGISTRY


def test_mcp_tool_count_is_exactly_6() -> None:
    """TOOL_REGISTRY has exactly 6 tools (4 core + 2 diagnostic)."""
    assert len(TOOL_REGISTRY) == 6, f"Expected 6, got {len(TOOL_REGISTRY)}"


def test_trace_lookup_input_model() -> None:
    """TraceLookupInput accepts valid parameters."""
    from askbook.mcp_server.contracts import TraceLookupInput

    inp = TraceLookupInput(trace_id="abc123", limit=10, days=3)
    assert inp.trace_id == "abc123"
    assert inp.limit == 10
    assert inp.days == 3


def test_trace_lookup_input_defaults() -> None:
    """TraceLookupInput uses sensible defaults."""
    from askbook.mcp_server.contracts import TraceLookupInput

    inp = TraceLookupInput()
    assert inp.trace_id is None
    assert inp.limit == 20
    assert inp.days == 7


def test_collection_stats_input_model() -> None:
    """CollectionStatsInput accepts collection name."""
    from askbook.mcp_server.contracts import CollectionStatsInput

    inp = CollectionStatsInput(collection="demo")
    assert inp.collection == "demo"


def test_collection_stats_input_default() -> None:
    """CollectionStatsInput defaults to 'default'."""
    from askbook.mcp_server.contracts import CollectionStatsInput

    inp = CollectionStatsInput()
    assert inp.collection == "default"


def test_tool_registry_has_diagnostic_tools() -> None:
    """TOOL_REGISTRY includes trace_lookup and collection_stats."""
    assert "trace_lookup" in TOOL_REGISTRY
    assert "collection_stats" in TOOL_REGISTRY
