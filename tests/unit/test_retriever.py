"""Unit tests for HybridRetriever (uses fakes, no real Chroma/BM25 files)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.embeddings.stub import StubEmbedder
from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
from askbook.vectorstores.bm25_index import BM25PersistentIndex


def _make_trace() -> Any:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _make_retriever(tmp_path: Path) -> HybridRetriever:
    from askbook.vectorstores.chroma_store import ChromaVectorStore

    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    return HybridRetriever(embedder=StubEmbedder(), store=store, bm25_index=bm25)


def test_hybrid_retriever_returns_lists(tmp_path: Path) -> None:
    retriever = _make_retriever(tmp_path)
    bm25_res, dense_res = asyncio.run(
        retriever.retrieve("test query", "col__stub-model__v1", candidate_k=5)
    )
    assert isinstance(bm25_res, list)
    assert isinstance(dense_res, list)


def test_hybrid_retriever_empty_index_returns_empty(tmp_path: Path) -> None:
    retriever = _make_retriever(tmp_path)
    bm25_res, dense_res = asyncio.run(
        retriever.retrieve("hello", "col__stub__v1", candidate_k=5)
    )
    assert bm25_res == []
    # dense may be empty if collection doesn't exist yet
    assert isinstance(dense_res, list)


def test_hybrid_retriever_does_not_populate_raw_text(tmp_path: Path) -> None:
    retriever = _make_retriever(tmp_path)
    bm25_res, dense_res = asyncio.run(
        retriever.retrieve("q", "col__stub__v1", candidate_k=5)
    )
    for r in bm25_res + dense_res:
        dumped = r.model_dump()
        assert "raw_text" not in dumped
        assert "full_content" not in dumped
        assert "page_content" not in dumped


def test_hybrid_retriever_node_populates_context(tmp_path: Path) -> None:
    retriever = _make_retriever(tmp_path)
    node = HybridRetrieverNode(retriever=retriever, top_k=5, trace_writer=_make_trace())
    ctx = {"query": "hello", "collection": "col__stub__v1"}
    out = node(ctx)  # type: ignore[arg-type]
    assert "bm25_results" in out
    assert "dense_results" in out
    assert isinstance(out["bm25_results"], list)
    assert isinstance(out["dense_results"], list)
