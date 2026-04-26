"""Smoke tests for dashboard pages 1-3 via streamlit AppTest."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from askbook.config.schema import ObservabilityConfig, VectorStoreConfig
from askbook.config.settings import Settings
from askbook.observability.schema import TraceEvent


def _make_settings(tmp_path: Path) -> Settings:
    """Return a Settings instance pointing observability and vectorstore at tmp_path."""
    return Settings(
        observability=ObservabilityConfig(
            trace_dir=str(tmp_path / "traces"),
            enabled=True,
            pii_redaction=False,
        ),
        vectorstore=VectorStoreConfig(
            provider="chroma",
            path=str(tmp_path / "chroma"),
        ),
    )


def _write_trace_event(trace_dir: Path, event: TraceEvent) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    jsonl_file = trace_dir / f"{today}.jsonl"
    with jsonl_file.open("a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")


def test_overview_page_renders_without_exception(tmp_path: Path) -> None:
    """Page 1 renders at least one metric and raises no exception."""
    trace_dir = tmp_path / "traces"
    event = TraceEvent(
        trace_id="test-trace-001",
        span_id="span-abc",
        event_type="span_end",
        node_name="rewriter",
        timestamp_utc=datetime.now(UTC),
        duration_ms=10.0,
    )
    _write_trace_event(trace_dir, event)

    cfg = _make_settings(tmp_path)

    at = AppTest.from_file("src/askbook/dashboard/pages/1_overview.py")
    at.session_state["cfg"] = cfg
    at.run(timeout=20)

    assert not at.exception
    assert len(at.metric) >= 1


def test_data_browser_page_handles_empty_chroma(tmp_path: Path) -> None:
    """Page 2 shows an info message when no Chroma collections exist."""
    cfg = _make_settings(tmp_path)

    at = AppTest.from_file("src/askbook/dashboard/pages/2_data_browser.py")
    at.session_state["cfg"] = cfg
    at.run(timeout=20)

    assert not at.exception
    assert len(at.info) >= 1


def test_ingestion_page_renders_when_no_traces(tmp_path: Path) -> None:
    """Page 3 shows an info message when the trace directory is empty."""
    # Create an empty traces dir so the loader doesn't choke on a missing path
    (tmp_path / "traces").mkdir(parents=True)

    cfg = _make_settings(tmp_path)

    at = AppTest.from_file("src/askbook/dashboard/pages/3_ingestion.py")
    at.session_state["cfg"] = cfg
    at.run(timeout=20)

    assert not at.exception
    assert len(at.info) >= 1
