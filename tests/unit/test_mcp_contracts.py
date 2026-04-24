"""Unit tests for MCP tool I/O contracts (DEV_SPEC Ch 30.1.2)."""

import pytest
from pydantic import ValidationError

from askbook.mcp_server.contracts import (
    AskInput,
    GetDocumentSummaryInput,
    ListCollectionsInput,
    SearchHit,
    SearchInput,
    ToolResponse,
)

# ---------------------------------------------------------------------------
# ToolResponse
# ---------------------------------------------------------------------------


def test_tool_response_success_requires_non_empty_source_ids() -> None:
    with pytest.raises(ValueError, match="non-empty source_ids"):
        ToolResponse(status="success", summary="ok", data={}, source_ids=[])


def test_tool_response_warning_allows_empty_source_ids() -> None:
    resp = ToolResponse(status="warning", summary="no results", source_ids=[])
    assert resp.status == "warning"
    assert resp.source_ids == []


def test_tool_response_error_allows_empty_source_ids() -> None:
    resp = ToolResponse(status="error", summary="something went wrong", source_ids=[])
    assert resp.status == "error"
    assert resp.source_ids == []


def test_tool_response_json_schema_is_exportable() -> None:
    schema = ToolResponse.model_json_schema()
    assert isinstance(schema, dict)
    props = schema.get("properties", {})
    assert "status" in props
    assert "source_ids" in props


def test_tool_response_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ToolResponse(  # type: ignore[call-arg]
            status="warning",
            summary="ok",
            source_ids=[],
            unknown_field="bad",
        )


# ---------------------------------------------------------------------------
# SearchInput
# ---------------------------------------------------------------------------


def test_search_input_schema_bounds() -> None:
    with pytest.raises(ValidationError):
        SearchInput(query="q", top_k=0)

    with pytest.raises(ValidationError):
        SearchInput(query="q", top_k=21)

    valid = SearchInput(query="hello", top_k=10)
    assert valid.top_k == 10
    assert valid.collection == "default"


# ---------------------------------------------------------------------------
# AskInput
# ---------------------------------------------------------------------------


def test_ask_input_schema_bounds_top_k_to_10() -> None:
    with pytest.raises(ValidationError):
        AskInput(question="what?", top_k=11)

    valid = AskInput(question="what?", top_k=10)
    assert valid.top_k == 10


# ---------------------------------------------------------------------------
# ListCollectionsInput
# ---------------------------------------------------------------------------


def test_list_collections_input_accepts_no_args() -> None:
    inp = ListCollectionsInput()
    assert inp is not None


# ---------------------------------------------------------------------------
# GetDocumentSummaryInput
# ---------------------------------------------------------------------------


def test_get_document_summary_input_requires_doc_id() -> None:
    with pytest.raises(ValidationError):
        GetDocumentSummaryInput()  # type: ignore[call-arg]

    valid = GetDocumentSummaryInput(doc_id="abc123")
    assert valid.doc_id == "abc123"
    assert valid.collection == "default"


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------


def test_search_hit_snippet_max_length_200() -> None:
    long_snippet = "x" * 201
    with pytest.raises(ValidationError):
        SearchHit(
            chunk_id="c1",
            score=0.9,
            snippet=long_snippet,
            retrieval_method="bm25",
        )

    valid = SearchHit(
        chunk_id="c1",
        score=0.9,
        snippet="x" * 200,
        retrieval_method="dense",
    )
    assert len(valid.snippet) == 200
