"""Trace schema for askbook observability.

Defines TraceEvent and three span types used in JSONL tracing.
Harness requirements:
- 30.1.3: QuerySpan.original_query must be a required field (no default).
- 30.1.1: TraceEvent.tags must reject raw_text, full_content, page_content keys.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TraceEvent(BaseModel):
    """A single observability event anchored to a pipeline span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    event_type: Literal["span_start", "span_end", "error"]
    node_name: str
    timestamp_utc: datetime
    duration_ms: float | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    prompt_template_hash: str | None = None
    error: str | None = None

    @field_validator("tags")
    @classmethod
    def _no_raw_text(cls, v: dict[str, Any]) -> dict[str, Any]:
        forbidden = {"raw_text", "full_content", "page_content"}
        found = forbidden & v.keys()
        if found:
            raise ValueError(f"tags must not contain sensitive keys: {sorted(found)}")
        return v


class QuerySpan(BaseModel):
    """Span capturing query-pipeline specific data.

    Harness 30.1.3: original_query has no default — it MUST be supplied.
    """

    trace_id: str
    original_query: str  # required — no default (Harness 30.1.3)
    rewritten_query: str | None = None
    collection: str | None = None
    retrieval_count: int = 0
    answer_source_ids: list[str] = Field(default_factory=list)


class IngestionSpan(BaseModel):
    """Span capturing document ingestion aggregate counts."""

    trace_id: str
    source_path: str
    doc_count: int = 0
    chunk_count: int = 0
    embedding_tokens: int = 0


class MCPToolSpan(BaseModel):
    """Span capturing an MCP tool invocation outcome."""

    trace_id: str
    tool_name: Literal["search", "ask", "list_collections", "get_document_summary"]
    status: Literal["success", "warning", "error"]
    source_ids: list[str] = Field(default_factory=list)
