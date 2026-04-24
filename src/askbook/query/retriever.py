"""HybridRetriever: BM25 + Dense search in parallel, wrapped as a PipelineNode."""

from __future__ import annotations

import asyncio

from askbook.core.interfaces import (
    BasePipelineNode,
    EmbedderProtocol,
    PipelineContext,
    TraceWriterProtocol,
    VectorStoreABC,
)
from askbook.core.models import RetrievalResult
from askbook.vectorstores.bm25_index import BM25PersistentIndex

_CANDIDATE_MULTIPLIER: int = 2


class HybridRetriever:
    """Parallel BM25 + dense retriever."""

    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        store: VectorStoreABC,
        bm25_index: BM25PersistentIndex,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25_index

    async def retrieve(
        self,
        query: str,
        collection: str,
        candidate_k: int,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        async def _bm25() -> list[RetrievalResult]:
            return await asyncio.to_thread(
                self._bm25.search_as_results, query, candidate_k
            )

        async def _dense() -> list[RetrievalResult]:
            vec = await asyncio.to_thread(self._embedder.embed_query, query)
            return await asyncio.to_thread(
                self._store.search, vec, collection, candidate_k
            )

        bm25_res, dense_res = await asyncio.gather(_bm25(), _dense())
        return bm25_res, dense_res


class HybridRetrieverNode(BasePipelineNode):
    """Pipeline node: parallel BM25 + dense retrieval."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        top_k: int,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="hybrid_retriever", trace_writer=trace_writer)
        self._retriever = retriever
        self._top_k = top_k

    def run(self, context: PipelineContext) -> PipelineContext:
        query = context.get("rewritten_query") or context["query"]
        collection = context["collection"]
        candidate_k = self._top_k * _CANDIDATE_MULTIPLIER
        bm25_res, dense_res = asyncio.run(
            self._retriever.retrieve(query, collection, candidate_k)
        )
        return {**context, "bm25_results": bm25_res, "dense_results": dense_res}


__all__ = ["HybridRetriever", "HybridRetrieverNode"]
