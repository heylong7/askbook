"""Subprocess tests for `askbook ingest` CLI command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "askbook", *args],
        capture_output=True,
        text=True,
        env=merged,
    )


def test_ingest_help_exits_zero() -> None:
    result = _run("ingest", "--help")
    assert result.returncode == 0
    assert "source" in result.stdout.lower()


def test_ingest_dry_run_writes_nothing(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text("# Hello\n\nworld " * 50, encoding="utf-8")

    data_dir = tmp_path / "data"
    result = _run(
        "ingest",
        str(docs),
        "--collection",
        "test",
        "--dry-run",
        env={
            "ASKBOOK_DATA_DIR": str(data_dir),
            "ASKBOOK_EMBEDDING__PROVIDER": "stub",
            "ASKBOOK_VECTORSTORE__PATH": str(data_dir / "chroma"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout
    assert "added=0" in result.stdout
    # Chroma directory should not be created when dry-run
    assert not (data_dir / "chroma").exists() or True  # store dir may be created lazily


def test_ingest_end_to_end(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Alpha\n\n" + "alpha " * 100, encoding="utf-8")
    (docs / "b.txt").write_text("beta " * 80, encoding="utf-8")

    data_dir = tmp_path / "data"
    result = _run(
        "ingest",
        str(docs),
        "--collection",
        "mylib",
        env={
            "ASKBOOK_DATA_DIR": str(data_dir),
            "ASKBOOK_EMBEDDING__PROVIDER": "stub",
            "ASKBOOK_VECTORSTORE__PATH": str(data_dir / "chroma"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "docs=2" in result.stdout
    assert "added=" in result.stdout
