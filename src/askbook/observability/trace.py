"""Async trace writer with span context management."""

from __future__ import annotations

import atexit
import contextlib
import queue
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime

from askbook.observability.redact import Redactor
from askbook.observability.schema import TraceEvent
from askbook.observability.sinks import TraceSink

_current_trace_id: ContextVar[str | None] = ContextVar(
    "_current_trace_id", default=None
)
_current_span_stack: ContextVar[tuple[str, ...]] = ContextVar(
    "_current_span_stack", default=()
)


@dataclass
class _RichSpan:
    node_name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _attributes: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def set_attribute(self, key: str, value: object) -> None:
        self._attributes[key] = value


class AsyncTraceWriter:
    def __init__(
        self,
        sink: TraceSink,
        *,
        redactor: Redactor | None = None,
        enabled: bool = True,
        buffer_size: int = 1000,
        flush_interval_seconds: float = 1.0,
    ) -> None:
        self._sink = sink
        self._redactor = redactor or Redactor(enabled=False)
        self._enabled = enabled
        self._flush_interval = flush_interval_seconds
        self._queue: queue.Queue[TraceEvent] = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        if enabled:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            atexit.register(self.flush)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self._flush_interval)
            self._drain()

    def _drain(self) -> None:
        batch: list[TraceEvent] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._sink.write(batch)

    @contextmanager
    def span(self, node_name: str) -> Iterator[_RichSpan]:
        if not self._enabled:
            yield _RichSpan(node_name=node_name)
            return

        trace_id = _current_trace_id.get() or uuid.uuid4().hex
        stack = _current_span_stack.get()
        parent_span_id = stack[-1] if stack else None

        s = _RichSpan(node_name=node_name, trace_id=trace_id)

        trace_token = _current_trace_id.set(trace_id)
        stack_token = _current_span_stack.set(stack + (s.span_id,))

        self._enqueue(
            TraceEvent(
                trace_id=trace_id,
                span_id=s.span_id,
                parent_span_id=parent_span_id,
                event_type="span_start",
                node_name=node_name,
                timestamp_utc=datetime.now(UTC),
            )
        )

        exc_raised: BaseException | None = None
        try:
            yield s
        except BaseException as exc:
            exc_raised = exc
            self._enqueue(
                TraceEvent(
                    trace_id=trace_id,
                    span_id=s.span_id,
                    parent_span_id=parent_span_id,
                    event_type="error",
                    node_name=node_name,
                    timestamp_utc=datetime.now(UTC),
                    error=str(exc),
                )
            )
            raise
        finally:
            _current_trace_id.reset(trace_token)
            _current_span_stack.reset(stack_token)
            if exc_raised is None:
                now = datetime.now(UTC)
                duration_ms = (now - s.started_at).total_seconds() * 1000
                tags = self._redactor.apply_to_mapping(s._attributes)
                self._enqueue(
                    TraceEvent(
                        trace_id=trace_id,
                        span_id=s.span_id,
                        parent_span_id=parent_span_id,
                        event_type="span_end",
                        node_name=node_name,
                        timestamp_utc=now,
                        duration_ms=duration_ms,
                        tags=tags,
                    )
                )

    def _enqueue(self, event: TraceEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to make room, then retry — prefer not losing the new event.
            with contextlib.suppress(queue.Empty):
                self._queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(event)

    def flush(self) -> None:
        self._stop.set()
        self._drain()

    def close(self) -> None:
        self.flush()
        self._sink.close()


@contextmanager
def use_trace_id(trace_id: str) -> Iterator[None]:
    token = _current_trace_id.set(trace_id)
    try:
        yield
    finally:
        _current_trace_id.reset(token)
