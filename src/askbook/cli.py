"""askbook CLI entry point (Typer)."""

from __future__ import annotations

from typing import Annotated

import typer

from askbook import __version__

app = typer.Typer(
    name="askbook",
    help="Local RAG + MCP Server for private knowledge bases.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"askbook {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """askbook root command."""


@app.command()
def ingest(
    source: Annotated[
        str,
        typer.Argument(help="Path to a file or directory to ingest."),
    ] = ".",
    collection: Annotated[
        str, typer.Option(help="Chroma collection name.")
    ] = "default",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    force_reindex: Annotated[bool, typer.Option("--force-reindex")] = False,
    config: Annotated[
        str, typer.Option("--config", help="Path to YAML config file.")
    ] = "",
) -> None:
    """Ingest documents into the vector store."""
    from pathlib import Path as _Path

    from askbook.ingestion.cli import run_ingest

    run_ingest(
        source=_Path(source),
        collection=collection,
        config_path=_Path(config) if config else None,
        dry_run=dry_run,
        force_reindex=force_reindex,
    )


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Natural language question.")],
    collection: Annotated[
        str, typer.Option(help="Collection namespace (used to find BM25 index).")
    ] = "default",
    config: Annotated[
        str, typer.Option("--config", help="Path to YAML config file.")
    ] = "",
) -> None:
    """Ask a question against an ingested collection."""
    from pathlib import Path as _Path

    from askbook.query.cli import run_query

    run_query(
        question=question,
        collection=collection,
        config_path=_Path(config) if config else None,
    )


@app.command(name="eval")
def eval_(
    dataset: Annotated[
        str, typer.Option("--dataset", help="Path to QA dataset YAML.")
    ] = "datasets/seed_manual.yaml",
    collection: Annotated[str, typer.Option(help="Collection namespace.")] = "demo",
    k: Annotated[int, typer.Option(help="Top-k cutoff for metrics.")] = 5,
    output: Annotated[
        str, typer.Option("--output", help="Where to write the JSON report.")
    ] = "",
    update_baseline: Annotated[
        str,
        typer.Option("--update-baseline", help="Path to baseline JSON to overwrite."),
    ] = "",
    config: Annotated[
        str, typer.Option("--config", help="Path to YAML config file.")
    ] = "",
) -> None:
    """Run the retrieval-evaluation suite (Phase 5)."""
    from pathlib import Path as _Path

    from askbook.evaluation.cli import run_eval

    raise typer.Exit(
        run_eval(
            dataset=_Path(dataset),
            collection=collection,
            k=k,
            output=_Path(output) if output else None,
            update_baseline=_Path(update_baseline) if update_baseline else None,
            config_path=_Path(config) if config else None,
        )
    )


@app.command()
def serve(
    collection: Annotated[
        str,
        typer.Option(help="Default collection for BM25 index binding."),
    ] = "default",
    config: Annotated[
        str,
        typer.Option("--config", help="Path to YAML config file."),
    ] = "",
) -> None:
    """Launch the MCP stdio server."""
    from pathlib import Path as _Path

    from askbook.mcp_server import run_server

    run_server(
        config_path=_Path(config) if config else None,
        collection_hint=collection,
    )


@app.command()
def migrate(
    from_: Annotated[str, typer.Option("--from", help="Source directory.")] = "",
    to: Annotated[str, typer.Option("--to", help="Target directory.")] = "",
) -> None:
    """Migrate data directory across askbook versions (R3)."""
    raise typer.Exit(code=0)


@app.command()
def dashboard(
    config: Annotated[
        str, typer.Option("--config", help="Path to YAML config file.")
    ] = "",
    port: Annotated[
        int, typer.Option("--port", help="Streamlit server port (0 = default 8501).")
    ] = 0,
) -> None:
    """Launch the Streamlit dashboard."""
    import os
    import subprocess
    import sys
    from importlib.resources import files

    app_path = str(files("askbook.dashboard").joinpath("app.py"))
    env = os.environ.copy()
    if config:
        env["ASKBOOK_CONFIG"] = config
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    if port:
        cmd += ["--server.port", str(port)]
    subprocess.run(cmd, env=env, check=False)


@app.command()
def gc(
    data_dir: Annotated[
        str,
        typer.Option("--data-dir", help="askbook data directory."),
    ] = "",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    trace_retention_days: Annotated[
        int,
        typer.Option("--trace-retention", help="Days to retain trace files."),
    ] = 30,
) -> None:
    """Garbage collect: orphan chunks, interface check, trace archive."""
    from datetime import UTC, datetime, timedelta
    from pathlib import Path
    from typing import Any

    from askbook.config.settings import Settings, load_settings

    # Determine data directory and derive sub-paths
    settings: Settings
    if data_dir:
        d = Path(data_dir).expanduser().resolve()
        settings = load_settings()
        settings.data_dir = d
        settings.vectorstore.path = str(d / "chroma")
        settings.observability.trace_dir = str(d / "traces")
    else:
        settings = load_settings()

    chromadb_path = Path(settings.vectorstore.path).expanduser()
    trace_path = Path(settings.observability.trace_dir).expanduser()

    stats: dict[str, Any] = {
        "orphan_chunks_deleted": 0,
        "orphan_collections_scanned": 0,
        "interface_methods_checked": 0,
        "interface_errors": [],
        "trace_files_archived": 0,
    }

    # ..................................................................
    # Helper to simplify stats increment
    def _inc(key: str, n: int = 1) -> None:
        stats[key] = stats[key] + n

    # --- 1. Orphan chunk scan ---
    typer.echo("--- Orphan chunk scan ---")
    try:
        from askbook.vectorstores.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(str(chromadb_path))
        collections = store.list_collections()
        _inc("orphan_collections_scanned", len(collections))

        for col in collections:
            docs = store.list_documents(col.name)
            for doc in docs:
                source = Path(str(doc.get("source_path", "")))
                doc_id = str(doc.get("doc_id", ""))
                if not source.exists() and doc_id:
                    if dry_run:
                        typer.echo(
                            "  [DRY-RUN] would delete orphan doc"
                            f" {doc_id} from {col.name}"
                        )
                    else:
                        deleted = store.delete([doc_id], col.name)
                        _inc("orphan_chunks_deleted", deleted)
                        typer.echo(
                            f"  Deleted {deleted} orphan chunk(s)"
                            f" for {doc_id} in {col.name}"
                        )
    except Exception as exc:
        typer.echo(f"  Orphan scan error: {exc}")

    # --- 2. Interface signature check ---
    typer.echo("--- Interface signature check ---")
    try:
        from inspect import isfunction, ismethoddescriptor

        from askbook.core.interfaces import (
            EmbedderProtocol,
            LLMProviderProtocol,
            VectorStoreABC,
        )

        # Check VectorStoreABC abstract methods
        abc_methods = list(VectorStoreABC.__abstractmethods__)
        for name in abc_methods:
            method = getattr(VectorStoreABC, name, None)
            if method is None or not (isfunction(method) or ismethoddescriptor(method)):
                stats["interface_errors"].append(
                    f"VectorStoreABC.{name} missing or non-callable"
                )
            _inc("interface_methods_checked")

        # Check LLMProviderProtocol methods
        for name in ("complete", "acomplete", "astream"):
            if not hasattr(LLMProviderProtocol, name):
                stats["interface_errors"].append(f"LLMProviderProtocol.{name} missing")
            _inc("interface_methods_checked")

        # Check EmbedderProtocol methods
        for name in (
            "embed_query",
            "embed_passage",
            "embed_batch",
        ):
            if not hasattr(EmbedderProtocol, name):
                stats["interface_errors"].append(f"EmbedderProtocol.{name} missing")
            _inc("interface_methods_checked")

        if not stats["interface_errors"]:
            typer.echo(f"  All {stats['interface_methods_checked']} methods OK")
        else:
            for err in stats["interface_errors"]:
                typer.echo(f"  ERROR: {err}")
    except Exception as exc:
        typer.echo(f"  Interface check error: {exc}")

    # --- 3. Trace archive ---
    typer.echo("--- Trace archive ---")
    try:
        cutoff = datetime.now(UTC) - timedelta(days=trace_retention_days)
        if trace_path.is_dir():
            for p in sorted(trace_path.glob("*.jsonl")):
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)
                if mtime < cutoff:
                    if dry_run:
                        typer.echo(f"  [DRY-RUN] would archive {p.name}")
                    else:
                        import gzip
                        import shutil

                        gz_path = p.with_suffix(".jsonl.gz")
                        with p.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        p.unlink()
                        _inc("trace_files_archived")
                        typer.echo(f"  Archived {p.name}")
        else:
            typer.echo(f"  Trace directory not found: {trace_path}")
    except Exception as exc:
        typer.echo(f"  Trace archive error: {exc}")

    # --- Summary ---
    typer.echo("--- Summary ---")
    if dry_run:
        typer.echo("  DRY-RUN: no files were modified")
    for key, val in stats.items():
        if key == "interface_errors":
            continue
        typer.echo(f"  {key}: {val}")
    if stats["interface_errors"]:
        typer.echo(f"  interface_errors: {len(stats['interface_errors'])}")


if __name__ == "__main__":
    app()
