"""Ingestion CLI helper — called by askbook.cli.ingest."""

from __future__ import annotations

from pathlib import Path

import typer

from askbook.config.settings import Settings, load_settings
from askbook.core.registry import ServiceRegistry
from askbook.ingestion.pipeline import IngestionPipeline
from askbook.vectorstores.bm25_index import BM25PersistentIndex


def run_ingest(
    source: Path,
    collection: str,
    *,
    settings: Settings | None = None,
    config_path: Path | None = None,
    dry_run: bool = False,
    force_reindex: bool = False,
) -> None:
    """Build pipeline from settings and run ingestion."""
    cfg = settings or load_settings(config_path)
    reg = ServiceRegistry()

    embedder = reg.build_embedder(cfg.embedding)
    store = reg.build_vectorstore(cfg.vectorstore)

    bm25_dir = cfg.data_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = bm25_dir / f"{collection}.pkl"
    bm25 = BM25PersistentIndex(path=bm25_path)

    pipeline = IngestionPipeline(
        embedder=embedder,
        store=store,
        bm25_index=bm25,
        chunk_size=cfg.ingestion.chunk_size,
        chunk_overlap=cfg.ingestion.chunk_overlap,
    )

    result = pipeline.run(
        source=source,
        collection=collection,
        dry_run=dry_run,
        force_reindex=force_reindex,
    )

    prefix = "[DRY-RUN] " if dry_run else ""
    typer.echo(
        f"{prefix}Ingestion complete — "
        f"docs={result.docs_processed} "
        f"added={result.chunks_added} "
        f"reused={result.chunks_reused} "
        f"deleted={result.chunks_deleted} "
        f"collection={result.collection} "
        f"({result.duration_seconds:.2f}s)"
    )
    if result.errors:
        for err in result.errors:
            typer.echo(f"  ERROR: {err}", err=True)
        raise typer.Exit(code=1)
