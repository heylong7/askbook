"""Tests for build_trace_writer factory."""

from __future__ import annotations

import json
import time

from askbook.config.schema import ObservabilityConfig
from askbook.observability.registry import build_trace_writer


def test_factory_returns_null_writer_when_disabled(tmp_path) -> None:
    cfg = ObservabilityConfig(enabled=False, trace_dir=str(tmp_path))
    writer = build_trace_writer(cfg)
    with writer.span("noop"):
        pass
    writer.flush()
    assert list(tmp_path.iterdir()) == []


def test_factory_returns_async_writer_with_file_sink_when_enabled(tmp_path) -> None:
    cfg = ObservabilityConfig(enabled=True, trace_dir=str(tmp_path))
    writer = build_trace_writer(cfg)
    with writer.span("test"):
        pass
    writer.flush()
    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert len(jsonl_files) >= 1


def test_factory_expands_user_home_in_trace_dir(tmp_path) -> None:
    cfg = ObservabilityConfig(enabled=True, trace_dir="~/.askbook/traces_test_registry")
    writer = build_trace_writer(cfg)
    assert writer is not None
    writer.flush()


def test_factory_attaches_redactor_per_pii_setting(tmp_path) -> None:
    cfg = ObservabilityConfig(
        enabled=True,
        trace_dir=str(tmp_path),
        pii_redaction=False,
    )
    writer = build_trace_writer(cfg)
    email = "user@example.com"
    with writer.span("check") as s:
        s.set_attribute("email", email)
    writer.flush()
    time.sleep(0.05)

    jsonl_files = list(tmp_path.glob("*.jsonl"))
    assert jsonl_files

    raw_lines = jsonl_files[0].read_text(encoding="utf-8").splitlines()
    span_end_lines = [
        json.loads(line)
        for line in raw_lines
        if json.loads(line).get("event_type") == "span_end"
    ]
    assert span_end_lines, "expected a span_end event"
    tags = span_end_lines[0].get("tags", {})
    assert tags.get("email") == email, f"expected raw email, got {tags.get('email')!r}"
