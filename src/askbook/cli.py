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
    dataset: Annotated[str, typer.Option(help="Path to QA dataset YAML.")] = "",
) -> None:
    """Run the evaluation suite (Phase 5)."""
    raise typer.Exit(code=0)


@app.command()
def serve() -> None:
    """Launch the MCP stdio server (Phase 3)."""
    raise typer.Exit(code=0)


@app.command()
def migrate(
    from_: Annotated[str, typer.Option("--from", help="Source directory.")] = "",
    to: Annotated[str, typer.Option("--to", help="Target directory.")] = "",
) -> None:
    """Migrate data directory across askbook versions (R3)."""
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
