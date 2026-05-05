"""Ingestion pipeline orchestrator."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from askbook.core.interfaces import (
    EmbedderProtocol,
    PipelineContext,
    TraceWriterProtocol,
    VectorStoreABC,
)
from askbook.core.models import IngestionResult
from askbook.ingestion.nodes import (
    BM25IndexUpdateNode,
    DedupNode,
    DocumentLoaderNode,
    EmbeddingNode,
    EnrichmentNode,
    SplitterNode,
    VectorStoreWriteNode,
)
from askbook.observability.null_trace import NullTraceWriter
from askbook.observability.trace import use_trace_id
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        store: VectorStoreABC,
        bm25_index: BM25PersistentIndex,
        llm: Any = None,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        embed_concurrency: int = 4,
        chroma_concurrency: int = 4,
        vision_concurrency: int = 2,
        trace_writer: TraceWriterProtocol | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25_index
        self._embed_concurrency = embed_concurrency
        self._chroma_concurrency = chroma_concurrency
        self._trace: TraceWriterProtocol = trace_writer or NullTraceWriter()

        # Build the 7-node pipeline
        self._load_node = DocumentLoaderNode(self._trace)
        self._split_node = SplitterNode(self._trace, chunk_size, chunk_overlap)
        if llm is not None:
            self._enrich_node = EnrichmentNode(
                llm=llm,
                trace_writer=self._trace,
                vision_concurrency=vision_concurrency,
            )
        else:
            self._enrich_node = EnrichmentNode(trace_writer=self._trace)
        self._dedup_node = DedupNode(self._trace)
        self._embed_node = EmbeddingNode(self._embedder, self._trace, embed_concurrency)
        self._write_node = VectorStoreWriteNode(
            self._store, self._trace, chroma_concurrency
        )
        self._bm25_node = BM25IndexUpdateNode(self._bm25, self._trace)

    def _collection_name(self, namespace: str) -> str:
        return self._store.make_collection_name(
            namespace=namespace, embed_model=self._embedder.model_name
        )

    def _fetch_existing_chunk_ids(
        self,
        ctx: PipelineContext,
        collection: str,
        force_reindex: bool,
    ) -> set[str]:
        """Return existing chunk IDs from the store for all loaded documents."""
        if force_reindex:
            return set()
        if not isinstance(self._store, ChromaVectorStore):
            return set()

        existing: set[str] = set()
        for doc in ctx.get("documents", []):
            source_path = doc.source_path
            if source_path:
                by_path = set(
                    self._store.list_chunk_ids_by_source_path(
                        source_path, collection=collection
                    )
                )
                if by_path:
                    existing |= by_path
                    continue
            # Fallback: lookup by doc_id
            existing |= set(
                self._store.list_chunk_ids_by_doc(doc.doc_id, collection=collection)
            )
        return existing

    def run(
        self,
        *,
        source: Path,
        collection: str,
        dry_run: bool = False,
        force_reindex: bool = False,
    ) -> IngestionResult:
        started = time.perf_counter()
        collection_full = self._collection_name(collection)
        trace_id = uuid.uuid4().hex

        ctx: PipelineContext = {
            "source_path": str(source),
            "collection": collection_full,
            "pipeline_trace_id": trace_id,
        }

        errors: list[str] = []

        with use_trace_id(trace_id):
            try:
                ctx = self._load_node(ctx)
                ctx = self._split_node(ctx)
                ctx = self._enrich_node(ctx)

                if not dry_run:
                    # Populate existing_chunk_ids before dedup
                    existing = self._fetch_existing_chunk_ids(
                        ctx, collection_full, force_reindex
                    )
                    ctx = {**ctx, "existing_chunk_ids": existing}

                    ctx = self._dedup_node(ctx)
                    ctx = self._embed_node(ctx)
                    ctx = self._write_node(ctx)
                    ctx = self._bm25_node(ctx)
            except Exception as exc:
                errors.append(str(exc))

        chunks = ctx.get("chunks", [])
        new_chunks = ctx.get("new_chunks", [])
        stale_chunk_ids = ctx.get("stale_chunk_ids", [])
        docs = ctx.get("documents", [])

        chunks_added = 0 if dry_run else len(new_chunks)
        # Reused = total chunks minus new ones (after dedup)
        chunks_reused = 0 if dry_run else max(0, len(chunks) - len(new_chunks))
        chunks_deleted = 0 if dry_run else len(stale_chunk_ids)

        return IngestionResult(
            docs_processed=len(docs),
            chunks_added=chunks_added,
            chunks_reused=chunks_reused,
            chunks_deleted=chunks_deleted,
            collection=collection_full,
            duration_seconds=time.perf_counter() - started,
            errors=errors,
        )


__all__ = ["IngestionPipeline"]
