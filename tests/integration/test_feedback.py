"""Integration tests for user feedback (_save_feedback in dashboard Page 5)."""

from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_page_module(monkeypatch: pytest.MonkeyPatch) -> object:
    """Import the dashboard page 5 module with streamlit mocked out."""
    module_name = "askbook.dashboard.pages.5_evaluation"
    sys.modules.pop(module_name, None)

    # Mock settings/events that run at module level
    mock_settings = MagicMock()
    mock_settings.observability.trace_dir = "/nonexistent/traces"
    monkeypatch.setattr("askbook.config.settings.load_settings", lambda: mock_settings)
    monkeypatch.setattr(
        "askbook.dashboard.loader.load_events",
        lambda trace_dir, days: [],
    )

    # Mock streamlit functions that run at module level
    monkeypatch.setattr("streamlit.title", MagicMock())
    monkeypatch.setattr("streamlit.header", MagicMock())
    monkeypatch.setattr(
        "streamlit.columns",
        lambda n, **kw: [
            MagicMock() for _ in range(len(n) if isinstance(n, (list, tuple)) else n)
        ],
    )
    monkeypatch.setattr("streamlit.metric", MagicMock())
    monkeypatch.setattr("streamlit.expander", lambda title: MagicMock())
    monkeypatch.setattr("streamlit.markdown", MagicMock())
    monkeypatch.setattr("streamlit.info", MagicMock())
    monkeypatch.setattr("streamlit.text_input", MagicMock(return_value=""))
    monkeypatch.setattr("streamlit.button", MagicMock(return_value=False))
    monkeypatch.setattr("streamlit.success", MagicMock())
    # Path.exists/glob/read_text are NOT mocked here — the tests set
    # Path.cwd() to tmp_path, a clean temp dir, so these naturally
    # return empty/false results during import.

    return importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feedback_jsonl_created_on_first_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """First feedback write creates the JSONL file with correct content."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    mod = _import_page_module(monkeypatch)
    mod._save_feedback("test question", "up")

    fb_path = tmp_path / "eval_runs" / "feedback.jsonl"
    assert fb_path.exists()
    lines = fb_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = __import__("json").loads(lines[0])
    assert entry["question"] == "test question"
    assert entry["feedback"] == "up"
    assert "timestamp" in entry


def test_feedback_entry_has_required_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each feedback entry contains question, feedback, and timestamp."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    mod = _import_page_module(monkeypatch)
    mod._save_feedback("what is askbook?", "down")

    fb_path = tmp_path / "eval_runs" / "feedback.jsonl"
    lines = fb_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = __import__("json").loads(lines[0])
    assert entry["question"] == "what is askbook?"
    assert entry["feedback"] == "down"
    assert "timestamp" in entry
    datetime.fromisoformat(entry["timestamp"])


def test_feedback_multiple_entries_appended(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Multiple feedback calls append entries without overwriting."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    mod = _import_page_module(monkeypatch)
    mod._save_feedback("q1", "up")
    mod._save_feedback("q2", "down")
    mod._save_feedback("q3", "up")

    fb_path = tmp_path / "eval_runs" / "feedback.jsonl"
    lines = fb_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    entries = [__import__("json").loads(line) for line in lines]
    assert entries[0]["question"] == "q1"
    assert entries[0]["feedback"] == "up"
    assert entries[1]["question"] == "q2"
    assert entries[1]["feedback"] == "down"
    assert entries[2]["question"] == "q3"
    assert entries[2]["feedback"] == "up"
