"""Protocol + ABC definitions for the pluggable abstraction layer
(DEV_SPEC Ch 19). Single source of truth — no concrete implementations live here."""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

from askbook.core.models import (
    Answer,
    Chunk,
    CollectionInfo,
    CollectionStats,
    Document,
    IngestionResult,
    LLMResponse,
    QAPair,
    RetrievalResult,
)


# ====================================================================
# LLM Provider (Protocol — 对外暴露)
# ====================================================================
@runtime_checkable
class LLMProviderProtocol(Protocol):
    @property
    def provider_name(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    async def astream(
        self,
        prompt: str,
        **kwargs: object,
    ) -> AsyncIterator[str]: ...


# ====================================================================
# Embedder (Protocol)
# ====================================================================
@runtime_checkable
class EmbedderProtocol(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passage(self, text: str) -> list[float]: ...

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]: ...


# ====================================================================
# Reranker (Protocol)
# ====================================================================
@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]: ...

    async def arerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]: ...


# ====================================================================
# Splitter (Protocol)
# ====================================================================
@runtime_checkable
class SplitterProtocol(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...


# ====================================================================
# VectorStore (ABC — 含 collection 命名规范)
# ====================================================================
class VectorStoreABC(ABC):
    COLLECTION_NAME_PATTERN = "{namespace}__{embed_model}__{version}"

    def make_collection_name(
        self,
        namespace: str,
        embed_model: str,
        version: str = "v1",
    ) -> str:
        return self.COLLECTION_NAME_PATTERN.format(
            namespace=namespace,
            embed_model=embed_model.replace("/", "-"),
            version=version,
        )

    @abstractmethod
    def upsert(self, chunks: list[Chunk], collection: str) -> int: ...

    @abstractmethod
    def delete(self, doc_ids: list[str], collection: str) -> int: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]: ...

    @abstractmethod
    def list_collections(self) -> list[CollectionInfo]: ...

    @abstractmethod
    def get_collection_stats(self, collection: str) -> CollectionStats: ...

    @abstractmethod
    def get_document_chunks(
        self,
        doc_id: str,
        collection: str,
    ) -> list[Chunk]:
        """Return all chunks belonging to doc_id in collection (may be empty)."""
        ...


# ====================================================================
# Pipeline primitives
# ====================================================================
@dataclass
class TraceSpan:
    """Runtime span object passed to pipeline nodes.

    Concrete TraceWriter implementations (Phase 4) return richer objects;
    the dataclass here keeps Phase 0 testable without observability deps."""

    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: object) -> None:
        """Store a tag on this span (no-op base implementation)."""
        self.attributes[key] = value


class PipelineContext(TypedDict, total=False):
    """Shared context threaded through pipeline nodes."""

    query: str
    collection: str
    source_path: str
    documents: list[Document]
    chunks: list[Chunk]
    new_chunks: list[Chunk]
    stale_chunk_ids: list[str]
    existing_chunk_ids: set[str]
    retrieval_results: list[RetrievalResult]
    answer: Answer
    ingestion_result: IngestionResult
    bm25_results: list[RetrievalResult]
    dense_results: list[RetrievalResult]
    rewritten_query: str
    original_query: str
    pipeline_trace_id: str


# ====================================================================
# Pipeline Node (ABC — 含 Trace 自动上报)
# ====================================================================
class BasePipelineNode(ABC):
    def __init__(self, name: str, trace_writer: TraceWriterProtocol) -> None:
        self.name = name
        self._trace = trace_writer

    def __call__(self, context: PipelineContext) -> PipelineContext:
        with self._trace.span(self.name) as span:
            self.before_run(context)
            result = self.run(context)
            self.after_run(result, span)
        return result

    def before_run(self, context: PipelineContext) -> None:
        return None

    def after_run(self, context: PipelineContext, span: TraceSpan) -> None:
        return None

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext: ...


# ====================================================================
# Evaluator (ABC)
# ====================================================================
@dataclass
class EvalResult:
    metrics: dict[str, float]
    notes: str = ""


class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def metric_names(self) -> list[str]: ...

    @abstractmethod
    def evaluate(
        self,
        queries: list[str],
        retrieved: list[list[RetrievalResult]],
        answers: list[str],
        ground_truths: list[QAPair],
    ) -> EvalResult: ...

    def to_dict(self, result: EvalResult) -> dict[str, float]:
        return {k: float(v) for k, v in result.metrics.items()}


# ====================================================================
# TraceWriter (Protocol — 由 Phase 4 的 observability 子包实现)
# ====================================================================
@runtime_checkable
class TraceWriterProtocol(Protocol):
    def span(self, name: str) -> contextlib.AbstractContextManager[TraceSpan]: ...

    def flush(self) -> None: ...


__all__ = [
    "LLMProviderProtocol",
    "EmbedderProtocol",
    "RerankerProtocol",
    "SplitterProtocol",
    "VectorStoreABC",
    "BasePipelineNode",
    "BaseEvaluator",
    "EvalResult",
    "TraceWriterProtocol",
    "TraceSpan",
    "PipelineContext",
]
