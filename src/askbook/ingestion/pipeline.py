"""Ingestion pipeline orchestrator."""

from __future__ import annotations

import time
from pathlib import Path

from askbook.core.interfaces import EmbedderProtocol, VectorStoreABC
from askbook.core.models import Chunk, IngestionResult
from askbook.ingestion.dedup import SHA256Deduplicator
from askbook.ingestion.loaders import MarkItDownLoader
from askbook.observability.null_trace import NullTraceWriter
from askbook.splitters.recursive import RecursiveTextSplitter
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore


class IngestionPipeline:
    def __init__(
        self,
        *,
        embedder: EmbedderProtocol,
        store: VectorStoreABC,
        bm25_index: BM25PersistentIndex,
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        embed_concurrency: int = 4,
        chroma_concurrency: int = 4,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25_index
        self._loader = MarkItDownLoader()
        self._splitter = RecursiveTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._dedup = SHA256Deduplicator()
        self._trace = NullTraceWriter()
        self._embed_concurrency = embed_concurrency
        self._chroma_concurrency = chroma_concurrency

    def _collection_name(self, namespace: str) -> str:
        return self._store.make_collection_name(
            namespace=namespace, embed_model=self._embedder.model_name
        )

    def _existing_chunk_ids_for_doc(
        self, doc_id: str, source_path: str, collection: str
    ) -> set[str]:
        if isinstance(self._store, ChromaVectorStore):
            # Query by source_path so we catch chunks from previous doc_id versions
            # (doc_id changes when mtime changes; source_path stays stable)
            by_path = set(
                self._store.list_chunk_ids_by_source_path(
                    source_path, collection=collection
                )
            )
            if by_path:
                return by_path
            # Fallback: original doc_id lookup (handles first-run edge cases)
            return set(self._store.list_chunk_ids_by_doc(doc_id, collection=collection))
        return set()

    def _process_one_doc(
        self,
        *,
        doc_chunks: list[Chunk],
        collection: str,
        force_reindex: bool,
        dry_run: bool,
    ) -> tuple[int, int, int]:
        """Return (added, reused, deleted) for this doc."""
        if not doc_chunks:
            return 0, 0, 0
        doc_id = doc_chunks[0].doc_id
        source_path = str(doc_chunks[0].metadata.get("source_path", ""))

        if force_reindex:
            existing: set[str] = set()
            if not dry_run:
                self._store.delete([doc_id], collection=collection)
        else:
            existing = self._existing_chunk_ids_for_doc(doc_id, source_path, collection)

        new, stale = self._dedup.filter_new_chunks(
            new_chunks=doc_chunks, existing_chunk_ids=existing
        )
        reused = len(doc_chunks) - len(new)

        if dry_run:
            return len(new), reused, len(stale)

        if new:
            vectors = self._embedder.embed_batch(
                [c.content for c in new], is_query=False
            )
            embedded = [
                c.model_copy(update={"embedding": v})
                for c, v in zip(new, vectors, strict=True)
            ]
            self._store.upsert(embedded, collection=collection)
            self._bm25.add([(c.chunk_id, c.content) for c in embedded])
        if stale:
            if isinstance(self._store, ChromaVectorStore):
                col = self._store._get_collection(collection)  # noqa: SLF001
                col.delete(ids=stale)
            self._bm25.remove(stale)

        return len(new), reused, len(stale)

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

        docs_processed = 0
        total_added = 0
        total_reused = 0
        total_deleted = 0
        errors: list[str] = []

        for path in MarkItDownLoader.iter_files(Path(source)):
            try:
                doc = self._loader.load(path)
                chunks = self._splitter.split(doc)
                added, reused, deleted = self._process_one_doc(
                    doc_chunks=chunks,
                    collection=collection_full,
                    force_reindex=force_reindex,
                    dry_run=dry_run,
                )
                docs_processed += 1
                total_added += added
                total_reused += reused
                total_deleted += deleted
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if not dry_run:
            self._bm25.save()

        return IngestionResult(
            docs_processed=docs_processed,
            chunks_added=0 if dry_run else total_added,
            chunks_reused=total_reused,
            chunks_deleted=0 if dry_run else total_deleted,
            collection=collection_full,
            duration_seconds=time.perf_counter() - started,
            errors=errors,
        )


__all__ = ["IngestionPipeline"]
