"""ServiceRegistry — factory and binding tests."""

from __future__ import annotations

import pytest

from askbook.core.registry import ServiceRegistry


def test_registry_singletons_not_yet_bound() -> None:
    reg = ServiceRegistry()
    with pytest.raises(LookupError):
        reg.get_llm()


def test_registry_supports_manual_binding() -> None:
    reg = ServiceRegistry()
    sentinel = object()
    reg.bind("llm", sentinel)
    assert reg.get("llm") is sentinel


def test_build_llm_stub() -> None:
    from askbook.config.schema import LLMConfig
    from askbook.providers.stub import StubLLMProvider

    reg = ServiceRegistry()
    llm = reg.build_llm(LLMConfig(provider="stub"))
    assert isinstance(llm, StubLLMProvider)


def test_build_llm_unknown_raises() -> None:
    from askbook.config.schema import LLMConfig

    reg = ServiceRegistry()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        reg.build_llm(LLMConfig(provider="bogus"))


def test_build_embedder_stub() -> None:
    from askbook.config.schema import EmbeddingConfig
    from askbook.embeddings.stub import StubEmbedder

    reg = ServiceRegistry()
    embedder = reg.build_embedder(EmbeddingConfig(provider="stub"))
    assert isinstance(embedder, StubEmbedder)


def test_build_embedder_unknown_provider_raises() -> None:
    from askbook.config.schema import EmbeddingConfig

    reg = ServiceRegistry()
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        reg.build_embedder(EmbeddingConfig(provider="nonexistent"))


def test_build_vectorstore_chroma(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from askbook.config.schema import VectorStoreConfig
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    reg = ServiceRegistry()
    store = reg.build_vectorstore(
        VectorStoreConfig(provider="chroma", path=str(tmp_path / "chroma"))
    )
    assert isinstance(store, ChromaVectorStore)


def test_build_vectorstore_unknown_provider_raises() -> None:
    from askbook.config.schema import VectorStoreConfig

    reg = ServiceRegistry()
    with pytest.raises(ValueError, match="Unknown vectorstore provider"):
        reg.build_vectorstore(VectorStoreConfig(provider="pinecone"))


def test_build_splitter_returns_correct_config() -> None:
    from askbook.config.schema import IngestionConfig
    from askbook.splitters.recursive import RecursiveCharacterTextSplitter

    reg = ServiceRegistry()
    splitter = reg.build_splitter(IngestionConfig(chunk_size=400, chunk_overlap=50))
    assert isinstance(splitter, RecursiveCharacterTextSplitter)
    assert splitter.chunk_size == 400
    assert splitter.chunk_overlap == 50
