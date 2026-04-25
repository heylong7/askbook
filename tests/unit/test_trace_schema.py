"""Unit tests for askbook.observability.schema — trace schema validation."""

import datetime

import pytest
from pydantic import ValidationError

from askbook.observability.schema import (
    IngestionSpan,
    MCPToolSpan,
    QuerySpan,
    TraceEvent,
)


def test_trace_event_required_fields() -> None:
    """TraceEvent requires trace_id, span_id, event_type, node_name, timestamp_utc."""
    now = datetime.datetime.now(tz=datetime.UTC)

    # Valid construction succeeds
    event = TraceEvent(
        trace_id="t1",
        span_id="s1",
        event_type="span_start",
        node_name="test_node",
        timestamp_utc=now,
    )
    assert event.trace_id == "t1"
    assert event.span_id == "s1"
    assert event.event_type == "span_start"
    assert event.node_name == "test_node"

    # Missing trace_id → ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent(  # type: ignore[call-arg]
            span_id="s1",
            event_type="span_start",
            node_name="test_node",
            timestamp_utc=now,
        )
    assert "trace_id" in str(exc_info.value)

    # Missing span_id → ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent(  # type: ignore[call-arg]
            trace_id="t1",
            event_type="span_start",
            node_name="test_node",
            timestamp_utc=now,
        )
    assert "span_id" in str(exc_info.value)

    # Missing event_type → ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent(  # type: ignore[call-arg]
            trace_id="t1",
            span_id="s1",
            node_name="test_node",
            timestamp_utc=now,
        )
    assert "event_type" in str(exc_info.value)

    # Missing node_name → ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent(  # type: ignore[call-arg]
            trace_id="t1",
            span_id="s1",
            event_type="span_start",
            timestamp_utc=now,
        )
    assert "node_name" in str(exc_info.value)

    # Missing timestamp_utc → ValidationError
    with pytest.raises(ValidationError) as exc_info:
        TraceEvent(  # type: ignore[call-arg]
            trace_id="t1",
            span_id="s1",
            event_type="span_start",
            node_name="test_node",
        )
    assert "timestamp_utc" in str(exc_info.value)


def test_query_span_requires_original_query_field() -> None:
    """Harness 30.1.3: original_query is required; rewritten_query defaults to None."""
    # Valid construction with only required fields
    span = QuerySpan(trace_id="t", original_query="q")
    assert span.original_query == "q"
    assert span.rewritten_query is None

    # Missing original_query → ValidationError pointing to original_query
    with pytest.raises(ValidationError) as exc_info:
        QuerySpan(trace_id="t")  # type: ignore[call-arg]
    assert "original_query" in str(exc_info.value)

    # With rewritten_query provided
    span2 = QuerySpan(trace_id="t", original_query="q", rewritten_query="q rewritten")
    assert span2.rewritten_query == "q rewritten"


def test_query_span_rejects_raw_text_in_tags() -> None:
    """Harness 30.1.1: TraceEvent tags with forbidden keys raise ValidationError."""
    now = datetime.datetime.now(tz=datetime.UTC)

    base_kwargs = {
        "trace_id": "t1",
        "span_id": "s1",
        "event_type": "span_start",
        "node_name": "test_node",
        "timestamp_utc": now,
    }

    # "raw_text" key → ValidationError
    with pytest.raises(ValidationError):
        TraceEvent(**base_kwargs, tags={"raw_text": "some content"})  # type: ignore[arg-type]

    # "full_content" key → ValidationError
    with pytest.raises(ValidationError):
        TraceEvent(**base_kwargs, tags={"full_content": "some content"})  # type: ignore[arg-type]

    # "page_content" key → ValidationError
    with pytest.raises(ValidationError):
        TraceEvent(**base_kwargs, tags={"page_content": "some content"})  # type: ignore[arg-type]

    # Safe keys are accepted
    event = TraceEvent(**base_kwargs, tags={"chunk_id": "abc", "score": 0.9})  # type: ignore[arg-type]
    assert event.tags["chunk_id"] == "abc"


def test_ingestion_span_aggregates_counts() -> None:
    """IngestionSpan fields have correct types and integer defaults of 0."""
    span = IngestionSpan(trace_id="t", source_path="/docs/book.pdf")
    assert span.trace_id == "t"
    assert span.source_path == "/docs/book.pdf"
    assert span.doc_count == 0
    assert span.chunk_count == 0
    assert span.embedding_tokens == 0

    # All counts can be set
    span2 = IngestionSpan(
        trace_id="t",
        source_path="/docs/book.pdf",
        doc_count=3,
        chunk_count=42,
        embedding_tokens=8000,
    )
    assert span2.doc_count == 3
    assert span2.chunk_count == 42
    assert span2.embedding_tokens == 8000


def test_mcp_tool_span_required_fields() -> None:
    """MCPToolSpan requires trace_id, tool_name from allowed set, status allowed."""
    # Valid construction
    span = MCPToolSpan(trace_id="t", tool_name="search", status="success")
    assert span.tool_name == "search"
    assert span.status == "success"

    # All valid tool_name values
    for tool in ("search", "ask", "list_collections", "get_document_summary"):
        s = MCPToolSpan(trace_id="t", tool_name=tool, status="success")  # type: ignore[arg-type]
        assert s.tool_name == tool

    # All valid status values
    for status in ("success", "warning", "error"):
        s = MCPToolSpan(trace_id="t", tool_name="ask", status=status)  # type: ignore[arg-type]
        assert s.status == status

    # Invalid tool_name → ValidationError
    with pytest.raises(ValidationError):
        MCPToolSpan(trace_id="t", tool_name="unknown_tool", status="success")  # type: ignore[arg-type]

    # Invalid status → ValidationError
    with pytest.raises(ValidationError):
        MCPToolSpan(trace_id="t", tool_name="search", status="pending")  # type: ignore[arg-type]

    # Missing trace_id → ValidationError
    with pytest.raises(ValidationError):
        MCPToolSpan(tool_name="search", status="success")  # type: ignore[call-arg]


def test_trace_event_json_roundtrip() -> None:
    """TraceEvent serialises to JSON and deserialises back with identical values."""
    now = datetime.datetime.now(tz=datetime.UTC)
    original = TraceEvent(
        trace_id="trace-abc",
        span_id="span-xyz",
        parent_span_id="span-parent",
        event_type="span_end",
        node_name="query_node",
        timestamp_utc=now,
        duration_ms=123.45,
        tags={"collection": "books"},
        prompt_template_hash="deadbeef",
    )

    json_str = original.model_dump_json()
    restored = TraceEvent.model_validate_json(json_str)

    assert restored.trace_id == original.trace_id
    assert restored.span_id == original.span_id
    assert restored.parent_span_id == original.parent_span_id
    assert restored.event_type == original.event_type
    assert restored.node_name == original.node_name
    assert restored.duration_ms == original.duration_ms
    assert restored.tags == original.tags
    assert restored.prompt_template_hash == original.prompt_template_hash
    assert restored.timestamp_utc == original.timestamp_utc
