"""Core pydantic data models (DEV_SPEC Ch 19 + Ch 30.1.1 hardening)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None = None


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage
    model: str
    provider: str
    latency_ms: float


class Document(BaseModel):
    doc_id: str  # SHA256(filepath + mtime)
    source_path: str
    content: str  # MarkItDown 转换后的 Markdown
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str  # SHA256(content)
    doc_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Harness 30.1.1: fixed whitelist, snippet capped at 200 chars.
    Full text fetched by chunk_id from VectorStore — no raw_text here."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float
    snippet: str = Field(max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_method: str  # "bm25" | "dense" | "rrf" | "reranked"


class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    usage: TokenUsage
    pipeline_trace_id: str


class CollectionInfo(BaseModel):
    name: str
    size: int
    embed_model: str


class CollectionStats(BaseModel):
    collection: str
    chunk_count: int
    doc_count: int
    last_updated: str


class QAPair(BaseModel):
    question: str
    answer: str
    source_ids: list[str] = Field(default_factory=list)


__all__ = [
    "TokenUsage",
    "LLMResponse",
    "Document",
    "Chunk",
    "RetrievalResult",
    "Citation",
    "Answer",
    "CollectionInfo",
    "CollectionStats",
    "QAPair",
]
