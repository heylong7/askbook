"""Integration tests for the askbook query CLI command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "askbook", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_query_help_exits_zero() -> None:
    result = _run("query", "--help")
    assert result.returncode == 0
    assert "question" in result.stdout.lower() or "help" in result.stdout.lower()


def test_query_end_to_end_stub(tmp_path: Path) -> None:
    """Run ingest then query using stub LLM/embedder."""
    sample_doc = tmp_path / "doc.md"
    sample_doc.write_text("# Alpha\nalpha beta gamma content for testing.")
    data_dir = tmp_path / "data"

    stub_env = {
        "ASKBOOK_LLM__PROVIDER": "stub",
        "ASKBOOK_EMBEDDING__PROVIDER": "stub",
        "ASKBOOK_VECTORSTORE__PATH": str(data_dir / "chroma"),
        "ASKBOOK_DATA_DIR": str(data_dir),
    }

    # Ingest first
    ingest_result = _run(
        "ingest", str(sample_doc), "--collection", "mylib", env=stub_env
    )
    assert ingest_result.returncode == 0, f"Ingest failed: {ingest_result.stderr}"

    # Query
    query_result = _run("query", "alpha", "--collection", "mylib", env=stub_env)
    assert query_result.returncode == 0, f"Query failed: {query_result.stderr}"
    assert "Sources:" in query_result.stdout
