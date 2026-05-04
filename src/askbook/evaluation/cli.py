"""askbook eval — CLI entry."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import typer

from askbook.core.models import RetrievalResult
from askbook.evaluation.datasets import QADataset
from askbook.evaluation.report import RetrievalEvalReport
from askbook.evaluation.runner import RetrievalEvalRunner

logger = logging.getLogger(__name__)


class _RetrieveOnlyPipeline(Protocol):
    """Minimal protocol satisfied by QueryPipeline.run_retrieve_only."""

    def run_retrieve_only(
        self, *, query: str, collection: str
    ) -> list[RetrievalResult]: ...


class _Flushable(Protocol):
    """Minimal protocol satisfied by trace writers."""

    def flush(self) -> None: ...


def _build_pipeline(
    config_path: Path | None, collection: str
) -> tuple[_RetrieveOnlyPipeline, str, _Flushable]:
    """Construct a real QueryPipeline from settings (kept thin for monkeypatching)."""
    from askbook.config import load_settings
    from askbook.core.registry import ServiceRegistry
    from askbook.observability.registry import build_trace_writer
    from askbook.query.fusion import RRFFusionNode
    from askbook.query.hyde import HyDENode
    from askbook.query.pipeline import QueryPipeline
    from askbook.query.reranker_stage import CrossEncoderRerankNode, LLMFineRerankNode
    from askbook.query.retriever import HybridRetriever, HybridRetrieverNode
    from askbook.query.rewriter import QueryRewriterNode
    from askbook.query.synthesizer import AnswerSynthesizerNode
    from askbook.vectorstores.bm25_index import BM25PersistentIndex

    cfg = load_settings(config_path)
    reg = ServiceRegistry()
    embedder = reg.build_embedder(cfg.embedding)
    store = reg.build_vectorstore(cfg.vectorstore)
    llm = reg.build_llm(cfg.llm)
    reranker = reg.build_reranker(cfg.query)
    trace = build_trace_writer(cfg.observability)
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
    return pipeline, collection_full, trace


def run_eval(
    *,
    dataset: Path,
    collection: str,
    k: int = 5,
    output: Path | None = None,
    update_baseline: Path | None = None,
    config_path: Path | None = None,
) -> int:
    """Run the retrieval-evaluation suite and write a JSON report."""
    if not dataset.exists():
        typer.echo(f"dataset not found: {dataset}", err=True)
        raise typer.Exit(code=2)

    ds = QADataset.from_yaml(dataset)
    pipeline, collection_full, trace = _build_pipeline(config_path, collection)
    try:
        runner = RetrievalEvalRunner(pipeline=pipeline, k=k)
        report: RetrievalEvalReport = runner.run(dataset=ds, collection=collection_full)
        report.dataset_path = str(dataset)
        report.collection = collection_full
    finally:
        try:
            trace.flush()
        except Exception:  # pragma: no cover
            logger.exception("trace flush failed")

    typer.echo(report.format_table())

    if output is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("eval_runs") / f"{ts}.json"
    report.write_json(output)
    typer.echo(f"report → {output}")

    if update_baseline is not None:
        update_baseline.parent.mkdir(parents=True, exist_ok=True)
        update_baseline.write_text(
            json.dumps(report.aggregate, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        typer.echo(f"baseline updated → {update_baseline}")

    return 0


__all__ = ["run_eval"]
