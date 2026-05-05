"""Build shared MCP server dependencies (assembled once at startup).

Layer 3 of 3:
  contracts.py  — Pydantic I/O models (done)
  tools.py      — Pure function handlers (done)
  deps.py       — Startup wiring (this file)
  server.py     — MCP SDK wrapping (future task)
"""

from __future__ import annotations

from pathlib import Path

from askbook.config import load_settings
from askbook.config.settings import Settings
from askbook.core.registry import ServiceRegistry
from askbook.mcp_server.tools import ServerDeps
from askbook.observability.registry import build_trace_writer
from askbook.query.fusion import RRFFusionNode
from askbook.query.hyde import HyDENode
from askbook.query.pipeline import QueryPipeline
from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
from askbook.query.rewriter import QueryRewriterNode
from askbook.query.synthesizer import AnswerSynthesizerNode
from askbook.vectorstores.bm25_index import BM25PersistentIndex

_bm25_cache: dict[str, BM25PersistentIndex] = {}


def _get_or_create_bm25(cfg: Settings, collection: str) -> BM25PersistentIndex:
    bm25_path = Path(cfg.data_dir).expanduser() / "bm25" / f"{collection}.pkl"
    if collection not in _bm25_cache:
        _bm25_cache[collection] = BM25PersistentIndex(path=bm25_path)
    return _bm25_cache[collection]


def build_server_deps(
    *,
    config_path: Path | None = None,
    collection_hint: str = "default",
) -> ServerDeps:
    """Assemble MCP server shared dependencies (built once at startup).

    BM25 indices are cached per collection name and created lazily.
    """
    cfg = load_settings(config_path)
    reg = ServiceRegistry()

    embedder = reg.build_embedder(cfg.embedding)
    store = reg.build_vectorstore(cfg.vectorstore)
    llm = reg.build_llm(cfg.llm)
    reranker = reg.build_reranker(cfg.query)
    trace = build_trace_writer(cfg.observability)

    bm25 = _get_or_create_bm25(cfg, collection_hint)

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

    return ServerDeps(
        pipeline=pipeline,
        store=store,
        embedder=embedder,
        fallback_text=AnswerSynthesizerNode.FALLBACK_TEXT,
        trace_writer=trace,
    )


__all__ = ["build_server_deps"]
