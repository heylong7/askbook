"""Domain exception hierarchy (DEV_SPEC Ch 5)."""

from __future__ import annotations


class AskbookError(Exception):
    """Project base exception."""


# ---- Ingestion ---------------------------------------------------------
class IngestionError(AskbookError):
    """Base class for ingestion-path failures."""


class DocumentLoadError(IngestionError):
    """Failed to load a source document (PDF/MD/etc.)."""


class ChunkingError(IngestionError):
    """Splitter failed to produce chunks."""


# ---- Query -------------------------------------------------------------
class QueryError(AskbookError):
    """Base class for query-path failures."""


class RetrievalError(QueryError):
    """Hybrid retriever failed."""


class RerankError(QueryError):
    """Reranker failed."""


# ---- Providers ---------------------------------------------------------
class ProviderError(AskbookError):
    """Base class for LLM/embedding provider failures."""

    def __init__(self, provider: str, msg: str) -> None:
        super().__init__(f"[{provider}] {msg}")
        self.provider = provider


class ProviderTimeoutError(ProviderError):
    """Provider call timed out."""


class ProviderQuotaError(ProviderError):
    """Provider rejected the call due to quota."""


class ProviderFallbackExhaustedError(ProviderError):
    """All providers in the fallback chain failed."""


# ---- MCP ---------------------------------------------------------------
class MCPToolError(AskbookError):
    """MCP tool handler raised."""

    def __init__(self, tool: str, msg: str) -> None:
        super().__init__(f"MCP tool '{tool}' failed: {msg}")
        self.tool = tool


__all__ = [
    "AskbookError",
    "IngestionError",
    "DocumentLoadError",
    "ChunkingError",
    "QueryError",
    "RetrievalError",
    "RerankError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderQuotaError",
    "ProviderFallbackExhaustedError",
    "MCPToolError",
]
