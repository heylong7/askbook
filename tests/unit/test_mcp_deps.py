"""Unit smoke test for build_server_deps (DEV_SPEC Ch 30.1.4 — Task 4).

Uses stub providers via monkeypatched env vars so no real models are loaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from askbook.query.pipeline import QueryPipeline
from askbook.vectorstores.chroma_store import ChromaVectorStore


def _clear_askbook_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any real ASKBOOK_* env vars to prevent leakage from user env."""
    for key in list(os.environ):
        if key.startswith("ASKBOOK_"):
            monkeypatch.delenv(key, raising=False)


def test_build_server_deps_from_stub_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_server_deps() assembles a fully wired ServerDeps using stub providers."""
    _clear_askbook_env(monkeypatch)

    # Override relevant settings via environment variables
    monkeypatch.setenv("ASKBOOK_LLM__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_EMBEDDING__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PROVIDER", "chroma")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PATH", str(tmp_path))

    from askbook.mcp_server.deps import build_server_deps

    deps = build_server_deps()

    assert deps.pipeline is not None
    assert isinstance(deps.pipeline, QueryPipeline)
    assert isinstance(deps.store, ChromaVectorStore)
    assert deps.fallback_text  # non-empty string
    assert deps.embedder is not None


def test_build_server_deps_fallback_text_matches_synthesizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fallback_text in ServerDeps matches AnswerSynthesizerNode.FALLBACK_TEXT."""
    _clear_askbook_env(monkeypatch)

    monkeypatch.setenv("ASKBOOK_LLM__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_EMBEDDING__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PROVIDER", "chroma")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PATH", str(tmp_path))

    from askbook.mcp_server.deps import build_server_deps
    from askbook.query.synthesizer import AnswerSynthesizerNode

    deps = build_server_deps()

    assert deps.fallback_text == AnswerSynthesizerNode.FALLBACK_TEXT


def test_build_server_deps_embedder_is_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Embedder is a StubEmbedder when ASKBOOK_EMBEDDING__PROVIDER=stub."""
    _clear_askbook_env(monkeypatch)

    monkeypatch.setenv("ASKBOOK_LLM__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_EMBEDDING__PROVIDER", "stub")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PROVIDER", "chroma")
    monkeypatch.setenv("ASKBOOK_VECTORSTORE__PATH", str(tmp_path))

    from askbook.embeddings.stub import StubEmbedder
    from askbook.mcp_server.deps import build_server_deps

    deps = build_server_deps()

    assert isinstance(deps.embedder, StubEmbedder)


def test_build_server_deps_not_called_at_import() -> None:
    """Importing deps module must not trigger build_server_deps() side-effects."""
    # If this import raises, real settings / models would be loaded at import time
    import importlib

    import askbook.mcp_server.deps as deps_mod

    importlib.reload(deps_mod)
    # No assertion needed — absence of error is the assertion
