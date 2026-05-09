"""Pydantic models for MCP tool I/O contracts (DEV_SPEC Ch 30.1.2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from askbook.core.models import Citation, TokenUsage


class ToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "warning", "error"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_empty_sources_on_success(self) -> ToolResponse:
        if self.status == "success" and not self.source_ids:
            raise ValueError(
                "status=success requires non-empty source_ids; "
                "use status=warning when retrieval yields no results"
            )
        return self


class SearchInput(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    snippet: str = Field(max_length=200)
    retrieval_method: str


class AskInput(BaseModel):
    question: str
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=10)


class AskData(BaseModel):
    answer: str
    citations: list[Citation]
    usage: TokenUsage


class ListCollectionsInput(BaseModel):
    pass


class CollectionDescriptor(BaseModel):
    namespace: str
    full_name: str
    chunk_count: int
    embed_model: str


class GetDocumentSummaryInput(BaseModel):
    doc_id: str
    collection: str = "default"


class DocumentSummaryData(BaseModel):
    doc_id: str
    source_path: str
    chunk_count: int
    first_snippets: list[str]


class TraceLookupInput(BaseModel):
    trace_id: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    days: int = Field(default=7, ge=1, le=30)


class TraceLookupData(BaseModel):
    events: list[dict[str, Any]]
    total_count: int


class CollectionStatsInput(BaseModel):
    collection: str = "default"


class CollectionStatsData(BaseModel):
    namespace: str
    full_name: str
    chunk_count: int
    embed_model: str
    document_count: int
    disk_size_bytes: int


class GetChunkContentInput(BaseModel):
    chunk_id: str
    collection: str = "default"


class GetChunkContentData(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    source_path: str


__all__ = [
    "ToolResponse",
    "SearchInput",
    "SearchHit",
    "AskInput",
    "AskData",
    "ListCollectionsInput",
    "CollectionDescriptor",
    "GetDocumentSummaryInput",
    "DocumentSummaryData",
    "TraceLookupInput",
    "TraceLookupData",
    "CollectionStatsInput",
    "CollectionStatsData",
    "GetChunkContentInput",
    "GetChunkContentData",
]
