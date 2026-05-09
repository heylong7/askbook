"""MCP tool handlers for the 4 core askbook tools (DEV_SPEC Ch 30.1.3).

Layer 2 of 3:
  contracts.py  — Pydantic I/O models (done)
  tools.py      — Pure function handlers (this file)
  server.py     — MCP SDK wrapping (future task)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from askbook.core.interfaces import (
    EmbedderProtocol,
    TraceWriterProtocol,
    VectorStoreABC,
)
from askbook.mcp_server.contracts import (
    AskData,
    AskInput,
    CollectionDescriptor,
    CollectionStatsData,
    CollectionStatsInput,
    DocumentSummaryData,
    GetChunkContentData,
    GetChunkContentInput,
    GetDocumentSummaryInput,
    ListCollectionsInput,
    SearchHit,
    SearchInput,
    ToolResponse,
    TraceLookupData,
    TraceLookupInput,
)
from askbook.observability.null_trace import NullTraceWriter
from askbook.query.pipeline import QueryPipeline

# Maximum number of first snippets returned by get_document_summary
_MAX_FIRST_SNIPPETS = 3
# Maximum snippet length (chars) — mirrors RetrievalResult constraint
_SNIPPET_MAX_LEN = 200


@dataclass(frozen=True)
class ServerDeps:
    """Single-startup-built, shared by all handlers.

    Constructed once when the MCP server starts; injected into every handler
    call so handlers remain pure functions with explicit dependencies.
    """

    pipeline: QueryPipeline
    store: VectorStoreABC
    embedder: EmbedderProtocol
    fallback_text: str  # copy of AnswerSynthesizerNode.FALLBACK_TEXT
    trace_writer: TraceWriterProtocol = field(default_factory=NullTraceWriter)


# ---------------------------------------------------------------------------
# handle_search
# ---------------------------------------------------------------------------


def handle_search(inp: SearchInput, deps: ServerDeps) -> ToolResponse:
    """Semantic search over a collection; returns ranked hit snippets."""
    with deps.trace_writer.span("mcp.search") as span:
        span.set_attribute("collection", inp.collection)
        span.set_attribute("query", inp.query)

        collection = deps.store.make_collection_name(
            namespace=inp.collection, embed_model=deps.embedder.model_name
        )
        query_vec = deps.embedder.embed_query(inp.query)
        results = deps.store.search(query_vec, collection, top_k=inp.top_k)

        if not results:
            response = ToolResponse(
                status="warning",
                summary=(
                    f"No results found for query '{inp.query}'"
                    f" in collection '{inp.collection}'."
                ),
                source_ids=[],
            )
            span.set_attribute("status", response.status)
            span.set_attribute("source_ids", response.source_ids)
            return response

        hits = [
            SearchHit(
                chunk_id=r.chunk_id,
                score=r.score,
                snippet=r.snippet[:_SNIPPET_MAX_LEN],
                retrieval_method=r.retrieval_method,
            ).model_dump()
            for r in results
        ]
        source_ids = [r.chunk_id for r in results]

        response = ToolResponse(
            status="success",
            summary=f"Found {len(results)} result(s) for '{inp.query}'.",
            data={"hits": hits},
            source_ids=source_ids,
        )
        span.set_attribute("status", response.status)
        span.set_attribute("source_ids", response.source_ids)
        return response


# ---------------------------------------------------------------------------
# handle_ask
# ---------------------------------------------------------------------------


def handle_ask(inp: AskInput, deps: ServerDeps) -> ToolResponse:
    """Run the full RAG query pipeline and return a synthesized answer."""
    with deps.trace_writer.span("mcp.ask") as span:
        span.set_attribute("collection", inp.collection)
        span.set_attribute("question", inp.question)

        collection = deps.store.make_collection_name(
            namespace=inp.collection, embed_model=deps.embedder.model_name
        )
        answer = deps.pipeline.run(query=inp.question, collection=collection)

        # Treat fallback text or missing citations as a warning (no reliable sources)
        is_fallback = answer.text == deps.fallback_text or not answer.citations
        if is_fallback:
            response = ToolResponse(
                status="warning",
                summary=(
                    "Knowledge base returned no relevant context; answer is a fallback."
                ),
                data={"answer": answer.text, "citations": [], "usage": {}},
                source_ids=[],
            )
            span.set_attribute("status", response.status)
            span.set_attribute("source_ids", response.source_ids)
            return response

        ask_data = AskData(
            answer=answer.text,
            citations=answer.citations,
            usage=answer.usage,
        )
        source_ids = [c.chunk_id for c in answer.citations]

        response = ToolResponse(
            status="success",
            summary=f"Answer synthesized from {len(answer.citations)} source(s).",
            data=ask_data.model_dump(),
            source_ids=source_ids,
        )
        span.set_attribute("status", response.status)
        span.set_attribute("source_ids", response.source_ids)
        return response


# ---------------------------------------------------------------------------
# handle_list_collections
# ---------------------------------------------------------------------------


def handle_list_collections(
    inp: ListCollectionsInput, deps: ServerDeps
) -> ToolResponse:
    """List all collections available in the vector store."""
    with deps.trace_writer.span("mcp.list_collections") as span:
        collections = deps.store.list_collections()

        if not collections:
            response = ToolResponse(
                status="warning",
                summary="No collections found in the vector store.",
                source_ids=[],
            )
            span.set_attribute("status", response.status)
            span.set_attribute("source_ids", response.source_ids)
            return response

        descriptors = [
            CollectionDescriptor(
                namespace=info.name.split("__")[0],
                full_name=info.name,
                chunk_count=info.size,
                embed_model=info.embed_model,
            ).model_dump()
            for info in collections
        ]
        source_ids = [info.name for info in collections]

        response = ToolResponse(
            status="success",
            summary=f"Found {len(collections)} collection(s).",
            data={"collections": descriptors},
            source_ids=source_ids,
        )
        span.set_attribute("status", response.status)
        span.set_attribute("source_ids", response.source_ids)
        return response


# ---------------------------------------------------------------------------
# handle_get_document_summary
# ---------------------------------------------------------------------------


def handle_get_document_summary(
    inp: GetDocumentSummaryInput, deps: ServerDeps
) -> ToolResponse:
    """Return metadata and first few snippets for a specific document."""
    with deps.trace_writer.span("mcp.get_document_summary") as span:
        span.set_attribute("collection", inp.collection)
        span.set_attribute("doc_id", inp.doc_id)

        collection = deps.store.make_collection_name(
            namespace=inp.collection, embed_model=deps.embedder.model_name
        )
        chunks = deps.store.get_document_chunks(
            doc_id=inp.doc_id, collection=collection
        )

        if not chunks:
            response = ToolResponse(
                status="warning",
                summary=(
                    f"No chunks found for document '{inp.doc_id}'"
                    f" in collection '{inp.collection}'."
                ),
                source_ids=[],
            )
            span.set_attribute("status", response.status)
            span.set_attribute("source_ids", response.source_ids)
            return response

        first_snippets = [
            c.content[:_SNIPPET_MAX_LEN] for c in chunks[:_MAX_FIRST_SNIPPETS]
        ]
        source_path: str = str(chunks[0].metadata.get("source_path", ""))
        source_ids = [c.chunk_id for c in chunks]

        summary_data = DocumentSummaryData(
            doc_id=inp.doc_id,
            source_path=source_path,
            chunk_count=len(chunks),
            first_snippets=first_snippets,
        )

        response = ToolResponse(
            status="success",
            summary=f"Document '{inp.doc_id}' has {len(chunks)} chunk(s).",
            data=summary_data.model_dump(),
            source_ids=source_ids,
        )
        span.set_attribute("status", response.status)
        span.set_attribute("source_ids", response.source_ids)
        return response


# ---------------------------------------------------------------------------
# handle_trace_lookup
# ---------------------------------------------------------------------------


def handle_trace_lookup(inp: TraceLookupInput, deps: ServerDeps) -> ToolResponse:
    """Look up trace events by trace_id or return recent events."""
    from pathlib import Path

    from askbook.dashboard.loader import load_events

    trace_dir = Path.home() / ".askbook" / "traces"
    events = load_events(trace_dir, days=inp.days)

    if inp.trace_id:
        events = [e for e in events if e.trace_id == inp.trace_id]

    matches = events[: inp.limit]
    data = TraceLookupData(
        events=[e.model_dump(mode="json") for e in matches],
        total_count=len(matches),
    )

    if not matches:
        return ToolResponse(
            status="warning",
            summary=(
                f"未找到匹配的 Trace 事件"
                f"（trace_id={inp.trace_id or 'any'}, days={inp.days}）"
            ),
            source_ids=[],
            data=data.model_dump(),
        )

    return ToolResponse(
        status="success",
        summary=f"找到 {len(matches)} 条 Trace 事件（筛选自 {len(events)} 条）",
        data=data.model_dump(),
        source_ids=[e.trace_id for e in matches],
    )


# ---------------------------------------------------------------------------
# handle_collection_stats
# ---------------------------------------------------------------------------


def handle_collection_stats(
    inp: CollectionStatsInput, deps: ServerDeps
) -> ToolResponse:
    """Return detailed statistics for a collection."""
    full_name = deps.store.make_collection_name(
        namespace=inp.collection, embed_model=deps.embedder.model_name
    )
    stats = deps.store.get_collection_stats(full_name)

    return ToolResponse(
        status="success",
        summary=f"Collection '{inp.collection}': {stats.chunk_count} chunks",
        data=CollectionStatsData(
            namespace=inp.collection,
            full_name=full_name,
            chunk_count=stats.chunk_count,
            embed_model=deps.embedder.model_name,
            document_count=getattr(stats, "doc_count", 0),
            disk_size_bytes=getattr(stats, "disk_size_bytes", 0),
        ).model_dump(),
        source_ids=[full_name],
    )


# ---------------------------------------------------------------------------
# handle_get_chunk_content
# ---------------------------------------------------------------------------


def handle_get_chunk_content(
    inp: GetChunkContentInput, deps: ServerDeps
) -> ToolResponse:
    """Return the full content of a single chunk by chunk_id."""
    with deps.trace_writer.span("mcp.get_chunk_content") as span:
        span.set_attribute("collection", inp.collection)
        span.set_attribute("chunk_id", inp.chunk_id)

        collection = deps.store.make_collection_name(
            namespace=inp.collection, embed_model=deps.embedder.model_name
        )
        chunk = deps.store.get_chunk_by_id(inp.chunk_id, collection)

        if chunk is None:
            response = ToolResponse(
                status="warning",
                summary=(
                    f"Chunk '{inp.chunk_id}' not found"
                    f" in collection '{inp.collection}'."
                ),
                source_ids=[],
            )
            span.set_attribute("status", response.status)
            span.set_attribute("source_ids", response.source_ids)
            return response

        data = GetChunkContentData(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=chunk.content,
            source_path=str(chunk.metadata.get("source_path", "")),
        )

        response = ToolResponse(
            status="success",
            summary=f"Chunk '{inp.chunk_id}' retrieved ({len(chunk.content)} chars).",
            data=data.model_dump(),
            source_ids=[chunk.chunk_id],
        )
        span.set_attribute("status", response.status)
        span.set_attribute("source_ids", response.source_ids)
        return response


# ---------------------------------------------------------------------------
# TOOL_REGISTRY
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[
    str, tuple[type[BaseModel], Callable[[Any, ServerDeps], ToolResponse]]
] = {
    "search": (SearchInput, handle_search),
    "ask": (AskInput, handle_ask),
    "list_collections": (ListCollectionsInput, handle_list_collections),
    "get_document_summary": (GetDocumentSummaryInput, handle_get_document_summary),
    "get_chunk_content": (GetChunkContentInput, handle_get_chunk_content),
    "trace_lookup": (TraceLookupInput, handle_trace_lookup),
    "collection_stats": (CollectionStatsInput, handle_collection_stats),
}


__all__ = [
    "ServerDeps",
    "handle_search",
    "handle_ask",
    "handle_list_collections",
    "handle_get_document_summary",
    "handle_get_chunk_content",
    "handle_trace_lookup",
    "handle_collection_stats",
    "TOOL_REGISTRY",
]
