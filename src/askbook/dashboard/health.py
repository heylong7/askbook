"""Health probe module for askbook dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HealthCheck:
    """Ping each infrastructure component and report status.

    Parameters
    ----------
    llm:
        Any object with a ``complete(prompt, **kwargs)`` method, or ``None``
        to skip the LLM probe.
    store:
        Any object with a ``list_collections()`` method (satisfies
        ``VectorStoreABC``), or ``None`` to skip the Chroma probe.
    bm25_dir:
        Directory expected to contain ``*.pkl`` BM25 index files, or ``None``
        to skip the BM25 probe.
    """

    llm: Any | None
    store: Any | None
    bm25_dir: Path | None

    def collect(self) -> dict[str, dict[str, str]]:
        """Ping each component.

        Returns a mapping of component name to status dict::

            {
                "chroma": {"status": "up" | "down", "detail": "..."},
                "llm":    {"status": "up" | "down", "detail": "..."},
                "bm25":   {"status": "up" | "down", "detail": "..."},
            }

        Components whose constructor argument is ``None`` are omitted.
        """
        result: dict[str, dict[str, str]] = {}

        if self.store is not None:
            result["chroma"] = self._probe_chroma()

        if self.llm is not None:
            result["llm"] = self._probe_llm()

        if self.bm25_dir is not None:
            result["bm25"] = self._probe_bm25()

        return result

    # ------------------------------------------------------------------
    # Private probes
    # ------------------------------------------------------------------

    def _probe_chroma(self) -> dict[str, str]:
        store: Any = self.store
        try:
            store.list_collections()
            return {"status": "up", "detail": ""}
        except Exception as exc:  # noqa: BLE001
            return {"status": "down", "detail": str(exc)}

    def _probe_llm(self) -> dict[str, str]:
        llm: Any = self.llm
        try:
            llm.complete("ping", max_tokens=1)
            return {"status": "up", "detail": ""}
        except Exception as exc:  # noqa: BLE001
            return {"status": "down", "detail": str(exc)}

    def _probe_bm25(self) -> dict[str, str]:
        bm25_dir = self.bm25_dir
        if bm25_dir is None:
            raise RuntimeError("_probe_bm25 called with bm25_dir=None")

        if not bm25_dir.exists():
            return {"status": "down", "detail": "directory does not exist"}

        pkl_files = list(bm25_dir.glob("*.pkl"))
        if not pkl_files:
            return {"status": "down", "detail": "no .pkl index files found"}

        return {"status": "up", "detail": f"{len(pkl_files)} index file(s) found"}
