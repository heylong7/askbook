"""Ingestion pipeline nodes (DEV_SPEC Ch 20)."""

from __future__ import annotations

import threading
from pathlib import Path

from askbook.core.interfaces import (
    BasePipelineNode,
    EmbedderProtocol,
    PipelineContext,
    TraceSpan,
    TraceWriterProtocol,
    VectorStoreABC,
)
from askbook.ingestion.dedup import SHA256Deduplicator
from askbook.ingestion.enrichment import (
    LLMEnrichmentNode as EnrichmentNode,  # noqa: F401
)
from askbook.ingestion.loaders import MarkItDownLoader
from askbook.splitters.recursive import RecursiveTextSplitter
from askbook.vectorstores.bm25_index import BM25PersistentIndex


class DocumentLoaderNode(BasePipelineNode):
    def __init__(self, trace_writer: TraceWriterProtocol) -> None:
        super().__init__("load", trace_writer)
        self._loader = MarkItDownLoader()

    def run(self, context: PipelineContext) -> PipelineContext:
        source = Path(context["source_path"])
        docs = [self._loader.load(p) for p in MarkItDownLoader.iter_files(source)]
        return {**context, "documents": docs}

    def after_run(self, context: PipelineContext, span: TraceSpan) -> None:
        span.attributes["load_count"] = len(context.get("documents", []))


class SplitterNode(BasePipelineNode):
    def __init__(
        self,
        trace_writer: TraceWriterProtocol,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
    ) -> None:
        super().__init__("split", trace_writer)
        self._splitter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def run(self, context: PipelineContext) -> PipelineContext:
        docs = context.get("documents", [])
        chunks = [c for d in docs for c in self._splitter.split(d)]
        return {**context, "chunks": chunks}

    def after_run(self, context: PipelineContext, span: TraceSpan) -> None:
        span.attributes["chunk_count"] = len(context.get("chunks", []))


class DedupNode(BasePipelineNode):
    def __init__(self, trace_writer: TraceWriterProtocol) -> None:
        super().__init__("dedup", trace_writer)
        self._dedup = SHA256Deduplicator()

    def run(self, context: PipelineContext) -> PipelineContext:
        new, stale = self._dedup.filter_new_chunks(
            new_chunks=list(context.get("chunks", [])),
            existing_chunk_ids=set(context.get("existing_chunk_ids", set())),
        )
        return {**context, "new_chunks": new, "stale_chunk_ids": stale}


class EmbeddingNode(BasePipelineNode):
    def __init__(
        self,
        embedder: EmbedderProtocol,
        trace_writer: TraceWriterProtocol,
        concurrency: int = 4,
    ) -> None:
        super().__init__("embed", trace_writer)
        self._embedder = embedder
        self._sem = threading.BoundedSemaphore(max(1, concurrency))

    def run(self, context: PipelineContext) -> PipelineContext:
        new_chunks = list(context.get("new_chunks", []))
        if not new_chunks:
            return {**context, "new_chunks": []}
        with self._sem:
            vectors = self._embedder.embed_batch(
                [c.content for c in new_chunks], is_query=False
            )
        embedded = [
            c.model_copy(update={"embedding": v})
            for c, v in zip(new_chunks, vectors, strict=True)
        ]
        return {**context, "new_chunks": embedded}

    def after_run(self, context: PipelineContext, span: TraceSpan) -> None:
        span.attributes["embed_count"] = len(context.get("new_chunks", []))


class VectorStoreWriteNode(BasePipelineNode):
    def __init__(
        self,
        store: VectorStoreABC,
        trace_writer: TraceWriterProtocol,
        concurrency: int = 4,
    ) -> None:
        super().__init__("write", trace_writer)
        self._store = store
        self._sem = threading.BoundedSemaphore(max(1, concurrency))

    def run(self, context: PipelineContext) -> PipelineContext:
        collection = context["collection"]
        new_chunks = list(context.get("new_chunks", []))
        stale_ids = list(context.get("stale_chunk_ids", []))
        with self._sem:
            if stale_ids:
                self._store_delete_ids(stale_ids, collection)
            if new_chunks:
                self._store.upsert(new_chunks, collection=collection)
        return context

    def after_run(self, context: PipelineContext, span: TraceSpan) -> None:
        span.attributes["written_count"] = len(context.get("new_chunks", []))

    def _store_delete_ids(self, ids: list[str], collection: str) -> None:
        from askbook.vectorstores.chroma_store import ChromaVectorStore

        if isinstance(self._store, ChromaVectorStore):
            col = self._store._get_collection(collection)  # noqa: SLF001
            col.delete(ids=ids)
        else:
            raise NotImplementedError(
                "VectorStoreWriteNode chunk-level delete requires ChromaVectorStore"
            )


class BM25IndexUpdateNode(BasePipelineNode):
    def __init__(
        self,
        index: BM25PersistentIndex,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__("bm25", trace_writer)
        self._index = index

    def run(self, context: PipelineContext) -> PipelineContext:
        stale = list(context.get("stale_chunk_ids", []))
        if stale:
            self._index.remove(stale)
        new_chunks = list(context.get("new_chunks", []))
        if new_chunks:
            self._index.add(
                [(c.chunk_id, c.content, c.metadata) for c in new_chunks]
            )
        self._index.save()
        return context


__all__ = [
    "DocumentLoaderNode",
    "SplitterNode",
    "EnrichmentNode",
    "DedupNode",
    "EmbeddingNode",
    "VectorStoreWriteNode",
    "BM25IndexUpdateNode",
]
