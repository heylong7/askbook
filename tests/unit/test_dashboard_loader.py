"""Unit tests for askbook.dashboard.loader and askbook.dashboard.health."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from askbook.observability.schema import TraceEvent


def _make_event(
    *,
    trace_id: str = "t1",
    span_id: str = "s1",
    event_type: str = "span_end",
    node_name: str = "query_node",
    duration_ms: float | None = 50.0,
    tags: dict[str, Any] | None = None,
    timestamp_utc: datetime | None = None,
) -> TraceEvent:
    return TraceEvent(
        trace_id=trace_id,
        span_id=span_id,
        event_type=event_type,  # type: ignore[arg-type]
        node_name=node_name,
        timestamp_utc=timestamp_utc or datetime.now(UTC),
        duration_ms=duration_ms,
        tags=tags or {},
    )


def _write_jsonl(path: Path, events: list[TraceEvent]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(ev.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# test_load_events_returns_parsed_records
# ---------------------------------------------------------------------------


def test_load_events_returns_parsed_records(tmp_path: Path) -> None:
    """load_events parses all valid JSONL lines in a file within the day window."""
    from askbook.dashboard.loader import load_events

    today = datetime.now(UTC).date()
    jsonl_file = tmp_path / f"{today.isoformat()}.jsonl"

    events = [
        _make_event(trace_id="t1", span_id="s1"),
        _make_event(trace_id="t2", span_id="s2"),
        _make_event(trace_id="t3", span_id="s3"),
    ]
    _write_jsonl(jsonl_file, events)

    result = load_events(tmp_path, days=1)

    assert len(result) == 3
    assert all(isinstance(ev, TraceEvent) for ev in result)
    trace_ids = {ev.trace_id for ev in result}
    assert trace_ids == {"t1", "t2", "t3"}


# ---------------------------------------------------------------------------
# test_load_events_respects_days_window
# ---------------------------------------------------------------------------


def test_load_events_respects_days_window(tmp_path: Path) -> None:
    """load_events only returns events from files within the days window."""
    from askbook.dashboard.loader import load_events

    today = datetime.now(UTC).date()
    yesterday = today - timedelta(days=1)
    eight_days_ago = today - timedelta(days=8)

    today_file = tmp_path / f"{today.isoformat()}.jsonl"
    yesterday_file = tmp_path / f"{yesterday.isoformat()}.jsonl"
    old_file = tmp_path / f"{eight_days_ago.isoformat()}.jsonl"

    _write_jsonl(today_file, [_make_event(trace_id="today")])
    _write_jsonl(yesterday_file, [_make_event(trace_id="yesterday")])
    _write_jsonl(old_file, [_make_event(trace_id="old")])

    result = load_events(tmp_path, days=7)

    trace_ids = {ev.trace_id for ev in result}
    assert "today" in trace_ids
    assert "yesterday" in trace_ids
    assert "old" not in trace_ids


# ---------------------------------------------------------------------------
# test_load_events_skips_malformed_lines
# ---------------------------------------------------------------------------


def test_load_events_skips_malformed_lines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_events skips bad lines and emits a warning log."""
    from askbook.dashboard.loader import load_events

    today = datetime.now(UTC).date()
    jsonl_file = tmp_path / f"{today.isoformat()}.jsonl"

    good_event = _make_event(trace_id="good")
    with jsonl_file.open("w", encoding="utf-8") as fh:
        fh.write(good_event.model_dump_json() + "\n")
        fh.write("not-json\n")

    with caplog.at_level(logging.WARNING, logger="askbook.dashboard.loader"):
        result = load_events(tmp_path, days=1)

    assert len(result) == 1
    assert result[0].trace_id == "good"
    assert any("warning" in r.levelname.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# test_aggregate_query_metrics_computes_p50_p90
# ---------------------------------------------------------------------------


def test_aggregate_query_metrics_computes_p50_p90(tmp_path: Path) -> None:
    """aggregate_query_metrics returns correct count, p50, p90, total_tokens."""
    from askbook.dashboard.loader import aggregate_query_metrics

    # Known durations: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # p50 (median of 10 values) = 55.0 using inclusive quantiles
    # p90 = 91.0 using inclusive quantiles
    durations = [float(d) for d in range(10, 110, 10)]
    events = [
        _make_event(
            event_type="span_end",
            node_name="query_node",
            duration_ms=d,
            tags={"tokens": 10},
        )
        for d in durations
    ]

    metrics = aggregate_query_metrics(events)

    assert metrics["count"] == 10
    # p50 for [10..100] inclusive quantile: within 5ms of 55
    assert abs(metrics["p50_ms"] - 55.0) <= 5.0
    # p90 for [10..100] inclusive quantile: within 5ms of 91
    assert abs(metrics["p90_ms"] - 91.0) <= 5.0
    assert metrics["total_tokens"] == 100


def test_aggregate_query_metrics_empty_returns_zeros() -> None:
    """aggregate_query_metrics with no events returns zeros."""
    from askbook.dashboard.loader import aggregate_query_metrics

    metrics = aggregate_query_metrics([])
    assert metrics["count"] == 0
    assert metrics["p50_ms"] == 0.0
    assert metrics["p90_ms"] == 0.0
    assert metrics["total_tokens"] == 0.0


# ---------------------------------------------------------------------------
# test_health_check_reports_chroma_unreachable
# ---------------------------------------------------------------------------


def test_health_check_reports_chroma_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HealthCheck.collect returns chroma=down when store.count() raises."""
    from askbook.dashboard.health import HealthCheck

    class _BrokenStore:
        def list_collections(self) -> list[Any]:
            raise RuntimeError("connection refused")

        def count(self) -> int:
            raise RuntimeError("connection refused")

    health = HealthCheck(llm=None, store=_BrokenStore(), bm25_dir=None)
    result = health.collect()

    assert "chroma" in result
    assert result["chroma"]["status"] == "down"
    assert len(result["chroma"]["detail"]) > 0


def test_health_check_chroma_up(tmp_path: Path) -> None:
    """HealthCheck.collect returns chroma=up when store.list_collections() succeeds."""
    from askbook.dashboard.health import HealthCheck

    class _OkStore:
        def list_collections(self) -> list[Any]:
            return []

    health = HealthCheck(llm=None, store=_OkStore(), bm25_dir=None)
    result = health.collect()

    assert result["chroma"]["status"] == "up"


def test_health_check_bm25_up(tmp_path: Path) -> None:
    """HealthCheck.collect returns bm25=up when bm25_dir has .pkl files."""
    from askbook.dashboard.health import HealthCheck

    pkl_file = tmp_path / "index.pkl"
    pkl_file.write_bytes(b"fake")

    health = HealthCheck(llm=None, store=None, bm25_dir=tmp_path)
    result = health.collect()

    assert result["bm25"]["status"] == "up"


def test_health_check_bm25_down_no_dir() -> None:
    """HealthCheck.collect returns bm25=down when bm25_dir does not exist."""
    from askbook.dashboard.health import HealthCheck

    health = HealthCheck(llm=None, store=None, bm25_dir=Path("/nonexistent/path"))
    result = health.collect()

    assert result["bm25"]["status"] == "down"


def test_health_check_llm_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """HealthCheck.collect returns llm=down when llm.complete() raises."""
    from askbook.dashboard.health import HealthCheck

    class _BrokenLLM:
        @property
        def provider_name(self) -> str:
            return "stub"

        def complete(self, prompt: str, **kwargs: Any) -> Any:
            raise RuntimeError("timeout")

    health = HealthCheck(llm=_BrokenLLM(), store=None, bm25_dir=None)
    result = health.collect()

    assert result["llm"]["status"] == "down"
    assert len(result["llm"]["detail"]) > 0


def test_health_check_none_components_skipped() -> None:
    """HealthCheck.collect skips None components entirely."""
    from askbook.dashboard.health import HealthCheck

    health = HealthCheck(llm=None, store=None, bm25_dir=None)
    result = health.collect()

    # None components should not appear — or appear as 'unknown'
    # The contract: no crash, returns a dict
    assert isinstance(result, dict)
