"""Integration tests for `askbook gc` command."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from askbook.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Return a fresh temp directory used as the askbook data dir."""
    return tmp_path


@pytest.fixture
def chroma_dir(data_dir: Path) -> Path:
    """Chroma sub-directory inside data_dir."""
    p = data_dir / "chroma"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def trace_dir(data_dir: Path) -> Path:
    """Trace sub-directory inside data_dir."""
    p = data_dir / "traces"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(*args: str) -> str:
    """Run `askbook gc` with given args and return stdout."""
    result = runner.invoke(app, ["gc", *args])
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    return result.output


def _make_trace_file(trace_dir: Path, days_ago: int) -> Path:
    """Create a trace .jsonl file with a fake mtime."""
    p = trace_dir / f"{days_ago}_days_ago.jsonl"
    p.write_text(json.dumps({"event": "test"}) + "\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(days=days_ago)
    old_ts = old_time.timestamp()
    # set mtime/atime on the file
    os_mod_time(p, old_ts)
    return p


def os_mod_time(path: Path, ts: float) -> None:
    """Set mtime and atime on *path*."""
    import os

    stat = path.stat()
    os.utime(path, (stat.st_atime, ts))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gc_orphan_chunk_detected(
    data_dir: Path, chroma_dir: Path, trace_dir: Path
) -> None:
    """GC should detect (and skip) orphan chunk deletions when no
    Chroma data exists — the orphan scan runs without error."""
    output = _invoke(
        "--data-dir",
        str(data_dir),
        "--trace-retention",
        "30",
    )
    # Should mention orphan scan phase
    assert "Orphan chunk scan" in output
    assert "collections_scanned: 0" in output


def test_gc_interface_check_passes(data_dir: Path, trace_dir: Path) -> None:
    """GC interface check should pass on the current codebase."""
    output = _invoke(
        "--data-dir",
        str(data_dir),
        "--trace-retention",
        "30",
    )
    assert "Interface signature check" in output
    assert "methods OK" in output or "interface_errors: 0" in output


def test_gc_trace_archive_old_files(
    data_dir: Path, chroma_dir: Path, trace_dir: Path
) -> None:
    """Old trace files should be archived (gzipped) and the originals removed."""
    # Create old trace files (40 days old, over 30-day retention)
    old_file = _make_trace_file(trace_dir, 40)
    # Create a recent trace file (should NOT be archived)
    recent_file = _make_trace_file(trace_dir, 5)

    output = _invoke(
        "--data-dir",
        str(data_dir),
        "--trace-retention",
        "30",
    )

    # Old file should be replaced with .gz version
    gz_path = old_file.with_suffix(".jsonl.gz")
    assert gz_path.exists(), f"Expected gz file: {gz_path}"
    assert not old_file.exists(), f"Original should be deleted: {old_file}"

    # Verify gzip content
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        content = f.read()
    assert "test" in content

    # Recent file should remain unchanged
    assert recent_file.exists(), f"Recent file should remain: {recent_file}"
    # No .gz for the recent file
    assert not recent_file.with_suffix(".jsonl.gz").exists()

    assert "trace_files_archived: 1" in output


def test_gc_dry_run_no_deletion(
    data_dir: Path, chroma_dir: Path, trace_dir: Path
) -> None:
    """Dry-run mode should report what would happen but not delete anything."""
    old_file = _make_trace_file(trace_dir, 40)

    output = _invoke(
        "--data-dir",
        str(data_dir),
        "--trace-retention",
        "30",
        "--dry-run",
    )

    assert "DRY-RUN" in output
    # Original file should still exist (not archived)
    assert old_file.exists(), "Original file must NOT be deleted in dry-run"
    # No gz file should exist
    assert not old_file.with_suffix(".jsonl.gz").exists()
