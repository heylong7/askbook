"""Integration test: HybridRetriever with real Chroma + BM25 on small corpus."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import Chunk
from askbook.embeddings.stub import StubEmbedder
from askbook.query.fusion import RRFFusionNode
from askbook.query.retriever import HybridRetriever
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore


def _make_trace() -> MagicMock:
    span = TraceSpan(name="test")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def test_hybrid_retriever_integration(tmp_path: Path) -> None:
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    collection = store.make_collection_name(
        namespace="test", embed_model=embedder.model_name
    )

    # Ingest 3 chunks
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="doc1",
            content=f"alpha beta gamma chunk {i}",
            embedding=embedder.embed_passage(f"alpha beta gamma chunk {i}"),
        )
        for i in range(3)
    ]
    store.upsert(chunks, collection)
    bm25.add([(c.chunk_id, c.content, c.metadata) for c in chunks])

    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    bm25_res, dense_res = asyncio.run(
        retriever.retrieve("alpha", collection, candidate_k=10)
    )
    assert len(bm25_res) > 0
    assert len(dense_res) > 0

    # Run RRF over results
    trace = _make_trace()
    fusion = RRFFusionNode(k=60, top_k=3, trace_writer=trace)
    ctx = {
        "query": "alpha",
        "collection": collection,
        "bm25_results": bm25_res,
        "dense_results": dense_res,
    }
    out = fusion(ctx)  # type: ignore[arg-type]
    assert len(out["retrieval_results"]) <= 3
    assert out["retrieval_results"][0].retrieval_method == "rrf"
