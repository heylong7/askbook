from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from askbook.observability.schema import TraceEvent


@runtime_checkable
class TraceSink(Protocol):
    def write(self, events: Sequence[TraceEvent]) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def write(self, events: Sequence[TraceEvent]) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class FileSink:
    base_dir: Path
    retention_days: int = 7

    def _today_path(self) -> Path:
        name = datetime.now(UTC).strftime("%Y-%m-%d") + ".jsonl"
        return self.base_dir / name

    def write(self, events: Sequence[TraceEvent]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self._today_path()
        with path.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.model_dump_json() + "\n")

    def purge_expired(self) -> int:
        today = datetime.now(UTC).date()
        deleted = 0
        for p in self.base_dir.glob("*.jsonl"):
            try:
                file_date = datetime.strptime(p.stem, "%Y-%m-%d").date()
            except ValueError:
                # filename doesn't match date pattern — leave it alone
                continue
            if (today - file_date).days > self.retention_days:
                p.unlink()
                deleted += 1
        return deleted

    def close(self) -> None:
        return None


@dataclass
class MultiSink:
    sinks: list[TraceSink]

    def write(self, events: Sequence[TraceEvent]) -> None:
        for sink in self.sinks:
            sink.write(events)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
