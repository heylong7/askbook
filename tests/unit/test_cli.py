"""Unit test: Typer app exists and --help runs clean."""

from __future__ import annotations

from typer.testing import CliRunner


def test_help_runs() -> None:
    from askbook.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("ingest", "query", "eval", "serve", "migrate"):
        assert sub in result.stdout


def test_version_flag() -> None:
    from askbook.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "askbook" in result.stdout.lower()
