"""ServiceRegistry — central place to obtain built dependencies.

Phase 0 ships the registry shell; concrete build_* methods raise
NotImplementedError pointing at the Phase that wires them up."""

from __future__ import annotations

from typing import Any


class ServiceRegistry:
    def __init__(self) -> None:
        self._bindings: dict[str, Any] = {}

    # Manual binding (useful for tests / DI) --------------------------
    def bind(self, key: str, value: Any) -> None:
        self._bindings[key] = value

    def get(self, key: str) -> Any:
        if key not in self._bindings:
            raise LookupError(f"No binding for {key!r}")
        return self._bindings[key]

    # Typed accessors --------------------------------------------------
    def get_llm(self) -> Any:
        return self.get("llm")

    def get_embedder(self) -> Any:
        return self.get("embedder")

    def get_vectorstore(self) -> Any:
        return self.get("vectorstore")

    # Factory entry points (implemented in later phases) --------------
    def build_llm(self, config: Any) -> Any:
        raise NotImplementedError(
            "ServiceRegistry.build_llm is wired up in Phase 2 (Query MVP)."
        )

    def build_embedder(self, config: Any) -> Any:
        raise NotImplementedError(
            "ServiceRegistry.build_embedder is wired up in Phase 1 (Ingestion MVP)."
        )

    def build_vectorstore(self, config: Any) -> Any:
        raise NotImplementedError(
            "ServiceRegistry.build_vectorstore is wired up in Phase 1 (Ingestion MVP)."
        )


__all__ = ["ServiceRegistry"]
