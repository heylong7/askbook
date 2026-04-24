"""MCP tool handlers for the 4 core askbook tools (DEV_SPEC Ch 30.1.3).

Layer 2 of 3:
  contracts.py  — Pydantic I/O models (done)
  tools.py      — Pure function handlers (this file)
  server.py     — MCP SDK wrapping (future task)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from askbook.core.interfaces import EmbedderProtocol, VectorStoreABC
from askbook.mcp_server.contracts import (
    AskData,
    AskInput,
    CollectionDescriptor,
    DocumentSummaryData,
    GetDocumentSummaryInput,
    ListCollectionsInput,
    SearchHit,
    SearchInput,
    ToolResponse,
)
from askbook.query.pipeline import QueryPipeline

# Maximum number of first snippets returned by get_document_summary
_MAX_FIRST_SNIPPETS = 3
# Maximum snippet length (chars) — mirrors RetrievalResult constraint
_SNIPPET_MAX_LEN = 200


@dataclass
class ServerDeps:
    """Single-startup-built, shared by all handlers.

    Constructed once when the MCP server starts; injected into every handler
    call so handlers remain pure functions with explicit dependencies.
    """

    pipeline: QueryPipeline
    store: VectorStoreABC
    embedder: EmbedderProtocol
    fallback_text: str  # copy of AnswerSynthesizerNode.FALLBACK_TEXT


# ---------------------------------------------------------------------------
# handle_search
# ---------------------------------------------------------------------------


def handle_search(inp: SearchInput, deps: ServerDeps) -> ToolResponse:
    """Semantic search over a collection; returns ranked hit snippets."""
    collection = deps.store.make_collection_name(
        namespace=inp.collection, embed_model=deps.embedder.model_name
    )
    query_vec = deps.embedder.embed_query(inp.query)
    results = deps.store.search(query_vec, collection, top_k=inp.top_k)

    if not results:
        return ToolResponse(
            status="warning",
            summary=(
                f"No results found for query '{inp.query}'"
                f" in collection '{inp.collection}'."
            ),
            source_ids=[],
        )

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

    return ToolResponse(
        status="success",
        summary=f"Found {len(results)} result(s) for '{inp.query}'.",
        data={"hits": hits},
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# handle_ask
# ---------------------------------------------------------------------------


def handle_ask(inp: AskInput, deps: ServerDeps) -> ToolResponse:
    """Run the full RAG query pipeline and return a synthesized answer."""
    collection = deps.store.make_collection_name(
        namespace=inp.collection, embed_model=deps.embedder.model_name
    )
    answer = deps.pipeline.run(query=inp.question, collection=collection)

    # Treat fallback text or missing citations as a warning (no reliable sources)
    is_fallback = answer.text == deps.fallback_text or not answer.citations
    if is_fallback:
        return ToolResponse(
            status="warning",
            summary=(
                "Knowledge base returned no relevant context; answer is a fallback."
            ),
            data={"answer": answer.text, "citations": [], "usage": {}},
            source_ids=[],
        )

    ask_data = AskData(
        answer=answer.text,
        citations=answer.citations,
        usage=answer.usage,
    )
    source_ids = [c.chunk_id for c in answer.citations]

    return ToolResponse(
        status="success",
        summary=f"Answer synthesized from {len(answer.citations)} source(s).",
        data=ask_data.model_dump(),
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# handle_list_collections
# ---------------------------------------------------------------------------


def handle_list_collections(
    inp: ListCollectionsInput, deps: ServerDeps
) -> ToolResponse:
    """List all collections available in the vector store."""
    collections = deps.store.list_collections()

    if not collections:
        return ToolResponse(
            status="warning",
            summary="No collections found in the vector store.",
            source_ids=[],
        )

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

    return ToolResponse(
        status="success",
        summary=f"Found {len(collections)} collection(s).",
        data={"collections": descriptors},
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# handle_get_document_summary
# ---------------------------------------------------------------------------


def handle_get_document_summary(
    inp: GetDocumentSummaryInput, deps: ServerDeps
) -> ToolResponse:
    """Return metadata and first few snippets for a specific document."""
    collection = deps.store.make_collection_name(
        namespace=inp.collection, embed_model=deps.embedder.model_name
    )
    chunks = deps.store.get_document_chunks(doc_id=inp.doc_id, collection=collection)

    if not chunks:
        return ToolResponse(
            status="warning",
            summary=(
                f"No chunks found for document '{inp.doc_id}'"
                f" in collection '{inp.collection}'."
            ),
            source_ids=[],
        )

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

    return ToolResponse(
        status="success",
        summary=f"Document '{inp.doc_id}' has {len(chunks)} chunk(s).",
        data=summary_data.model_dump(),
        source_ids=source_ids,
    )


# ---------------------------------------------------------------------------
# TOOL_REGISTRY
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[
    str, tuple[type[Any], Callable[[Any, ServerDeps], ToolResponse]]
] = {
    "search": (SearchInput, handle_search),
    "ask": (AskInput, handle_ask),
    "list_collections": (ListCollectionsInput, handle_list_collections),
    "get_document_summary": (GetDocumentSummaryInput, handle_get_document_summary),
}


__all__ = [
    "ServerDeps",
    "handle_search",
    "handle_ask",
    "handle_list_collections",
    "handle_get_document_summary",
    "TOOL_REGISTRY",
]
