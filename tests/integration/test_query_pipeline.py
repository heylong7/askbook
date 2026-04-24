"""Integration test: QueryPipeline end-to-end with stub LLM and real Chroma+BM25."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from askbook.core.interfaces import TraceSpan
from askbook.core.models import Chunk
from askbook.embeddings.stub import StubEmbedder
from askbook.providers.stub import StubLLMProvider
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.pipeline import QueryPipeline
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode
from askbook.rerankers.stub import StubReranker
from askbook.vectorstores.bm25_index import BM25PersistentIndex
from askbook.vectorstores.chroma_store import ChromaVectorStore


def _make_trace() -> Any:
    span = TraceSpan(name="t")
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=span)
    cm.__exit__ = MagicMock(return_value=False)
    trace = MagicMock()
    trace.span = MagicMock(return_value=cm)
    return trace


def _build_pipeline(tmp_path: Path, *, collection: str) -> QueryPipeline:
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    llm = StubLLMProvider(canned_response="answer with [c0]")
    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    trace = _make_trace()
    return QueryPipeline(
        rewriter=QueryRewriterNode(enabled=False, trace_writer=trace),
        hyde=HyDENode(enabled=False, trace_writer=trace),
        retriever_node=HybridRetrieverNode(
            retriever=retriever, top_k=5, trace_writer=trace
        ),
        fusion_node=RRFFusionNode(k=60, top_k=5, trace_writer=trace),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=StubReranker(), top_k=3, trace_writer=trace
        ),
        llm_rerank_node=LLMFineRerankNode(enabled=False, trace_writer=trace),
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=trace),
    )


def test_query_pipeline_end_to_end(tmp_path: Path) -> None:
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    bm25 = BM25PersistentIndex(path=tmp_path / "bm25.pkl")
    collection = store.make_collection_name(
        namespace="mylib", embed_model=embedder.model_name
    )

    # Ingest small corpus
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="d1",
            content=f"alpha beta gamma {i}",
            embedding=embedder.embed_passage(f"alpha beta gamma {i}"),
        )
        for i in range(5)
    ]
    store.upsert(chunks, collection)
    bm25.add([(c.chunk_id, c.content) for c in chunks])

    pipeline = _build_pipeline(tmp_path, collection=collection)
    answer = pipeline.run(query="alpha", collection=collection)

    assert answer.text != ""
    assert answer.pipeline_trace_id != ""


def test_query_pipeline_empty_corpus_returns_fallback(tmp_path: Path) -> None:
    embedder = StubEmbedder()
    store = ChromaVectorStore(path=str(tmp_path / "chroma"))
    collection = store.make_collection_name(
        namespace="empty", embed_model=embedder.model_name
    )

    pipeline = _build_pipeline(tmp_path, collection=collection)
    answer = pipeline.run(query="anything", collection=collection)

    assert "资料不足" in answer.text
    assert answer.citations == []
