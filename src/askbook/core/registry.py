"""ServiceRegistry — central place to obtain built dependencies."""

from __future__ import annotations

from typing import Any

from askbook.config.schema import EmbeddingConfig, IngestionConfig, VectorStoreConfig
from askbook.core.interfaces import EmbedderProtocol, VectorStoreABC
from askbook.splitters.recursive import RecursiveCharacterTextSplitter


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

    # Factory entry points --------------------------------------------
    def build_llm(self, config: Any) -> Any:
        raise NotImplementedError(
            "ServiceRegistry.build_llm is wired up in Phase 2 (Query MVP)."
        )

    def build_embedder(self, config: EmbeddingConfig) -> EmbedderProtocol:
        provider = config.provider.lower()
        if provider == "stub":
            from askbook.embeddings.stub import StubEmbedder

            return StubEmbedder()
        if provider in ("bge-m3", "bge_m3"):
            from askbook.embeddings.bge_m3 import BGEM3Embedder

            return BGEM3Embedder(
                model=config.model,
                normalize=config.normalize,
                device=config.device,
                batch_size=config.batch_size,
            )
        raise ValueError(
            f"Unknown embedder provider {config.provider!r}. "
            "Supported: 'stub', 'bge-m3'."
        )

    def build_vectorstore(self, config: VectorStoreConfig) -> VectorStoreABC:
        provider = config.provider.lower()
        if provider == "chroma":
            from askbook.vectorstores.chroma_store import ChromaVectorStore

            return ChromaVectorStore(path=config.path)
        raise ValueError(
            f"Unknown vectorstore provider {config.provider!r}. Supported: 'chroma'."
        )

    def build_splitter(self, config: IngestionConfig) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )


__all__ = ["ServiceRegistry"]
