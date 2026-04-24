"""run_query: builds QueryPipeline from settings and executes a single query."""

from __future__ import annotations

from pathlib import Path

import typer

from askbook.core.registry import ServiceRegistry
from askbook.observability.null_trace import NullTraceWriter
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.pipeline import QueryPipeline
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode
from askbook.vectorstores.bm25_index import BM25PersistentIndex


def run_query(
    question: str,
    collection: str,
    *,
    config_path: Path | None = None,
) -> None:
    from askbook.config import load_settings

    cfg = load_settings(config_path)
    reg = ServiceRegistry()
    embedder = reg.build_embedder(cfg.embedding)
    store = reg.build_vectorstore(cfg.vectorstore)
    llm = reg.build_llm(cfg.llm)
    reranker = reg.build_reranker(cfg.query)
    trace = NullTraceWriter()

    bm25_path = Path(cfg.data_dir).expanduser() / "bm25" / f"{collection}.pkl"
    bm25 = BM25PersistentIndex(path=bm25_path)

    retriever = HybridRetriever(embedder=embedder, store=store, bm25_index=bm25)
    pipeline = QueryPipeline(
        rewriter=QueryRewriterNode(
            enabled=cfg.query.enable_rewrite, llm=llm, trace_writer=trace
        ),
        hyde=HyDENode(enabled=cfg.query.enable_hyde, llm=llm, trace_writer=trace),
        retriever_node=HybridRetrieverNode(
            retriever=retriever, top_k=cfg.query.top_k, trace_writer=trace
        ),
        fusion_node=RRFFusionNode(
            k=cfg.query.rrf_k, top_k=cfg.query.top_k, trace_writer=trace
        ),
        ce_rerank_node=CrossEncoderRerankNode(
            reranker=reranker, top_k=cfg.query.rerank_top_k, trace_writer=trace
        ),
        llm_rerank_node=LLMFineRerankNode(
            enabled=cfg.query.enable_llm_rerank, llm=llm, trace_writer=trace
        ),
        synthesizer_node=AnswerSynthesizerNode(llm=llm, trace_writer=trace),
    )

    collection_full = store.make_collection_name(
        namespace=collection, embed_model=embedder.model_name
    )
    answer = pipeline.run(query=question, collection=collection_full)

    typer.echo(answer.text)
    typer.echo("")
    typer.echo("Sources:")
    for c in answer.citations:
        typer.echo(f"  - {c.chunk_id} ({c.source}) score={c.score:.3f}")


__all__ = ["run_query"]
