"""Integration smoke: run CLI via subprocess as an end user would."""

from __future__ import annotations

import subprocess
import sys


def test_python_module_entry() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "askbook", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout


def test_ingest_placeholder_message() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "askbook", "ingest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
