"""Integration tests for AsyncTraceWriter."""

from __future__ import annotations

import json
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from askbook.observability.redact import Redactor
from askbook.observability.sinks import FileSink
from askbook.observability.trace import AsyncTraceWriter


def _read_events(tmp_path: Path) -> list[dict]:
    events: list[dict] = []
    for p in sorted(tmp_path.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def test_writer_emits_span_start_and_end_events(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(FileSink(tmp_path), redactor=Redactor(enabled=False))
    with writer.span("nodeA"):
        pass
    writer.flush()

    events = _read_events(tmp_path)
    assert len(events) >= 2

    types = [e["event_type"] for e in events]
    assert "span_start" in types
    assert "span_end" in types

    for e in events:
        assert e["node_name"] == "nodeA"


def test_writer_records_error_event_on_exception(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(FileSink(tmp_path), redactor=Redactor(enabled=False))
    with pytest.raises(RuntimeError, match="boom"), writer.span("nodeA"):
        raise RuntimeError("boom")
    writer.flush()

    events = _read_events(tmp_path)
    error_events = [e for e in events if e["event_type"] == "error"]
    assert len(error_events) >= 1
    assert "boom" in error_events[0]["error"]


def test_nested_spans_set_parent_span_id(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(FileSink(tmp_path), redactor=Redactor(enabled=False))
    with writer.span("outer") as outer, writer.span("inner"):
        pass
    writer.flush()

    events = _read_events(tmp_path)
    inner_events = [e for e in events if e["node_name"] == "inner"]
    assert len(inner_events) >= 1
    for e in inner_events:
        assert e["parent_span_id"] == outer.span_id


def test_concurrent_100_spans_no_loss(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(
        FileSink(tmp_path), redactor=Redactor(enabled=False), buffer_size=200
    )

    def _do_span() -> None:
        with writer.span("node"):
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_do_span) for _ in range(100)]
        for f in futs:
            f.result()

    writer.flush()

    events = _read_events(tmp_path)
    end_events = [e for e in events if e["event_type"] == "span_end"]
    assert len(end_events) == 100


def test_atexit_flush_persists_buffered_events(tmp_path: Path) -> None:
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{Path("E:/ClaudeCode/askbook/src").as_posix()}")
        from pathlib import Path
        from askbook.observability.sinks import FileSink
        from askbook.observability.trace import AsyncTraceWriter
        from askbook.observability.redact import Redactor

        writer = AsyncTraceWriter(
            FileSink(Path(r"{tmp_path.as_posix()}")),
            redactor=Redactor(enabled=False),
        )
        for _ in range(5):
            with writer.span("node"):
                pass
        sys.exit(0)
    """)

    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    events = _read_events(tmp_path)
    end_events = [e for e in events if e["event_type"] == "span_end"]
    assert len(end_events) == 5


def test_writer_respects_disabled_flag(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(
        FileSink(tmp_path), redactor=Redactor(enabled=False), enabled=False
    )
    with writer.span("nodeA"):
        pass

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) == 0


def test_writer_redacts_text_attributes(tmp_path: Path) -> None:
    writer = AsyncTraceWriter(FileSink(tmp_path), redactor=Redactor(enabled=True))
    with writer.span("ask") as s:
        s.set_attribute("query", "alice@x.com")
    writer.flush()

    events = _read_events(tmp_path)
    end_events = [e for e in events if e["event_type"] == "span_end"]
    assert len(end_events) >= 1
    assert end_events[0]["tags"]["query"] == "<REDACTED:EMAIL>"
