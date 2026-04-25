"""Protocol conformance + ABC abstract-method set assertions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import nullcontext

import pytest

from askbook.core import interfaces
from askbook.core.models import (
    Chunk,
    CollectionStats,
    Document,
    LLMResponse,
    RetrievalResult,
    TokenUsage,
)


# ---- LLM Provider ----------------------------------------------------
class FakeLLM:
    provider_name = "fake"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="ok",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="fake",
            provider="fake",
            latency_ms=0.0,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self.complete(prompt)

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        yield "ok"


def test_llm_provider_protocol_runtime_check() -> None:
    obj = FakeLLM()
    assert isinstance(obj, interfaces.LLMProviderProtocol)


# ---- Embedder --------------------------------------------------------
class FakeEmbedder:
    model_name = "fake-embed"
    dimension = 3

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_passage(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_embedder_protocol() -> None:
    assert isinstance(FakeEmbedder(), interfaces.EmbedderProtocol)


# ---- Reranker --------------------------------------------------------
class FakeReranker:
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return results[:top_k]

    async def arerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        return self.rerank(query, results, top_k)


def test_reranker_protocol() -> None:
    assert isinstance(FakeReranker(), interfaces.RerankerProtocol)


# ---- Splitter --------------------------------------------------------
class FakeSplitter:
    def split(self, document: Document) -> list[Chunk]:
        return [Chunk(chunk_id="c0", doc_id=document.doc_id, content=document.content)]


def test_splitter_protocol() -> None:
    assert isinstance(FakeSplitter(), interfaces.SplitterProtocol)


# ---- VectorStore ABC -------------------------------------------------
def test_vectorstore_is_abstract() -> None:
    with pytest.raises(TypeError):
        interfaces.VectorStoreABC()  # type: ignore[abstract]


def test_vectorstore_abstract_methods() -> None:
    abstract = interfaces.VectorStoreABC.__abstractmethods__
    assert abstract == {
        "upsert",
        "delete",
        "search",
        "list_collections",
        "get_collection_stats",
        "get_document_chunks",
    }


def test_vectorstore_collection_name_helper() -> None:
    class Dummy(interfaces.VectorStoreABC):
        def upsert(self, chunks, collection):  # type: ignore[no-untyped-def]
            return 0

        def delete(self, doc_ids, collection):  # type: ignore[no-untyped-def]
            return 0

        def search(self, query_embedding, collection, top_k, filters=None):  # type: ignore[no-untyped-def]
            return []

        def list_collections(self):  # type: ignore[no-untyped-def]
            return []

        def get_collection_stats(self, collection):  # type: ignore[no-untyped-def]
            return CollectionStats(
                collection=collection,
                chunk_count=0,
                doc_count=0,
                last_updated="",
            )

        def get_document_chunks(self, doc_id, collection):  # type: ignore[no-untyped-def]
            return []

    d = Dummy()
    name = d.make_collection_name("notes", "BAAI/bge-m3")
    assert name == "notes__BAAI-bge-m3__v1"


# ---- Pipeline Node ABC -----------------------------------------------
def test_pipeline_node_abstract() -> None:
    assert "run" in interfaces.BasePipelineNode.__abstractmethods__


# ---- Evaluator ABC ---------------------------------------------------
def test_evaluator_abstract() -> None:
    assert "evaluate" in interfaces.BaseEvaluator.__abstractmethods__
    assert "metric_names" in interfaces.BaseEvaluator.__abstractmethods__


# ---- TraceWriter Protocol --------------------------------------------
class FakeTrace:
    def span(self, name: str):  # type: ignore[no-untyped-def]
        return nullcontext(interfaces.TraceSpan(name=name))

    def flush(self) -> None:
        return None


def test_trace_writer_protocol() -> None:
    assert isinstance(FakeTrace(), interfaces.TraceWriterProtocol)
