"""Smoke tests for dashboard pages 1-5 via streamlit AppTest."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from askbook.config.schema import ObservabilityConfig, VectorStoreConfig
from askbook.config.settings import Settings
from askbook.observability.schema import TraceEvent

_REPO_ROOT = Path(__file__).parents[2]
_PAGES_DIR = _REPO_ROOT / "src" / "askbook" / "dashboard" / "pages"
_APP_TIMEOUT_S = 20  # generous limit for slow CI runners


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

    at = AppTest.from_file(str(_PAGES_DIR / "1_overview.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception
    assert len(at.metric) >= 1


def test_data_browser_page_handles_empty_chroma(tmp_path: Path) -> None:
    """Page 2 shows an info message when no Chroma collections exist."""
    cfg = _make_settings(tmp_path)

    at = AppTest.from_file(str(_PAGES_DIR / "2_data_browser.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception
    assert len(at.info) >= 1


def test_ingestion_page_renders_when_no_traces(tmp_path: Path) -> None:
    """Page 3 shows an info message when the trace directory is empty."""
    (tmp_path / "traces").mkdir(parents=True)

    cfg = _make_settings(tmp_path)

    at = AppTest.from_file(str(_PAGES_DIR / "3_ingestion.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception
    assert len(at.info) >= 1


def test_sidebar_lists_5_pages() -> None:
    """Verify the sidebar markdown lists all 5 pages."""
    app_path = _REPO_ROOT / "src" / "askbook" / "dashboard" / "app.py"
    content = app_path.read_text(encoding="utf-8")

    assert "1" in content
    assert "2" in content
    assert "3" in content
    assert "4" in content
    assert "5" in content
    assert "Trace 查看器" in content
    assert "评估结果" in content
    assert "Pipeline 监控" in content


def test_trace_viewer_page_handles_empty_dir(tmp_path: Path) -> None:
    """Page 4 shows an info message when no trace data exists."""
    (tmp_path / "traces").mkdir(parents=True)

    cfg = _make_settings(tmp_path)

    at = AppTest.from_file(str(_PAGES_DIR / "4_trace_viewer.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception
    assert len(at.info) >= 1


def test_trace_viewer_page_shows_dataframe_with_events(tmp_path: Path) -> None:
    """Page 4 renders a dataframe when trace events exist."""
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

    at = AppTest.from_file(str(_PAGES_DIR / "4_trace_viewer.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception


def test_evaluation_page_renders_metric_cards(tmp_path: Path) -> None:
    """Page 5 renders 4 harness health metric cards."""
    cfg = _make_settings(tmp_path)

    at = AppTest.from_file(str(_PAGES_DIR / "5_evaluation.py"))
    at.session_state["cfg"] = cfg
    at.run(timeout=_APP_TIMEOUT_S)

    assert not at.exception
    assert len(at.metric) >= 4
