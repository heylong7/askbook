"""Null trace writer — Phase 1 placeholder until Phase 4's async writer lands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from askbook.core.interfaces import TraceSpan


class NullTraceWriter:
    """Implements TraceWriterProtocol with no side effects."""

    @contextmanager
    def span(self, name: str) -> Iterator[TraceSpan]:
        yield TraceSpan(name=name)

    def flush(self) -> None:
        return None


__all__ = ["NullTraceWriter"]
