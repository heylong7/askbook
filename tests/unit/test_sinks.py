from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from askbook.observability.schema import TraceEvent


def _make_event() -> TraceEvent:
    return TraceEvent(
        trace_id="t1",
        span_id="s1",
        event_type="span_start",
        node_name="test_node",
        timestamp_utc=datetime(2026, 4, 25, 0, 0, 0, tzinfo=UTC),
    )


def test_null_sink_accepts_events_without_io(tmp_path: Path) -> None:
    from askbook.observability.sinks import NullSink

    sink = NullSink()
    event = _make_event()
    sink.write([event, event])  # must not raise
    files = list(tmp_path.iterdir())
    assert files == []  # no files created


def test_file_sink_appends_jsonl_to_today(tmp_path: Path) -> None:
    from askbook.observability.sinks import FileSink

    event = _make_event()
    sink = FileSink(base_dir=tmp_path)
    sink.write([event])

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    expected = tmp_path / f"{today}.jsonl"
    assert expected.exists()
    last_line = expected.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert last_line == event.model_dump_json()


def test_file_sink_creates_dir_if_missing(tmp_path: Path) -> None:
    from askbook.observability.sinks import FileSink

    nested = tmp_path / "deeply" / "nested" / "logs"
    assert not nested.exists()

    sink = FileSink(base_dir=nested)
    sink.write([_make_event()])

    assert nested.exists()


def test_file_sink_purges_files_older_than_retention(tmp_path: Path) -> None:
    from askbook.observability.sinks import FileSink

    today = datetime.now(UTC).date()

    def make_file(days_ago: int) -> Path:
        date = today - timedelta(days=days_ago)
        p = tmp_path / f"{date.strftime('%Y-%m-%d')}.jsonl"
        p.write_text("{}\n", encoding="utf-8")
        return p

    old_file = make_file(8)  # 8 days ago — beyond 7-day retention
    recent_file = make_file(1)  # 1 day ago — within retention
    today_file = make_file(0)  # today

    sink = FileSink(base_dir=tmp_path, retention_days=7)
    deleted = sink.purge_expired()

    assert deleted == 1
    assert not old_file.exists()
    assert recent_file.exists()
    assert today_file.exists()


def test_multi_sink_dispatches_to_all(tmp_path: Path) -> None:
    from askbook.observability.sinks import FileSink, MultiSink, NullSink

    file_sink = FileSink(base_dir=tmp_path)
    multi = MultiSink(sinks=[NullSink(), file_sink])
    multi.write([_make_event()])

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    assert (tmp_path / f"{today}.jsonl").exists()
