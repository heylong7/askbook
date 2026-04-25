"""Factory for building configured AsyncTraceWriter instances."""

from __future__ import annotations

from pathlib import Path

from askbook.config.schema import ObservabilityConfig
from askbook.observability.redact import Redactor
from askbook.observability.sinks import FileSink, NullSink
from askbook.observability.trace import AsyncTraceWriter


def build_trace_writer(cfg: ObservabilityConfig) -> AsyncTraceWriter:
    if not cfg.enabled:
        return AsyncTraceWriter(NullSink(), enabled=False)
    sink = FileSink(
        base_dir=Path(cfg.trace_dir).expanduser(),
        retention_days=cfg.retention_days,
    )
    sink.purge_expired()
    return AsyncTraceWriter(
        sink,
        redactor=Redactor(enabled=cfg.pii_redaction),
        flush_interval_seconds=cfg.flush_interval_seconds,
    )
