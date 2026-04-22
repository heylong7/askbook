"""Verify domain exception hierarchy matches DEV_SPEC Ch 5."""

from __future__ import annotations

import pytest

from askbook.core import exceptions as exc


def test_base_exception() -> None:
    assert issubclass(exc.AskbookError, Exception)


def test_ingestion_hierarchy() -> None:
    assert issubclass(exc.IngestionError, exc.AskbookError)
    assert issubclass(exc.DocumentLoadError, exc.IngestionError)
    assert issubclass(exc.ChunkingError, exc.IngestionError)


def test_query_hierarchy() -> None:
    assert issubclass(exc.QueryError, exc.AskbookError)
    assert issubclass(exc.RetrievalError, exc.QueryError)
    assert issubclass(exc.RerankError, exc.QueryError)


def test_provider_error_format() -> None:
    err = exc.ProviderError(provider="ollama", msg="connection refused")
    assert err.provider == "ollama"
    assert "[ollama]" in str(err)
    assert "connection refused" in str(err)


def test_provider_hierarchy() -> None:
    assert issubclass(exc.ProviderTimeoutError, exc.ProviderError)
    assert issubclass(exc.ProviderQuotaError, exc.ProviderError)
    assert issubclass(exc.ProviderFallbackExhaustedError, exc.ProviderError)


def test_mcp_error_format() -> None:
    err = exc.MCPToolError(tool="ask", msg="boom")
    assert str(err) == "MCP tool 'ask' failed: boom"


def test_raises_with_context() -> None:
    with pytest.raises(exc.DocumentLoadError):
        raise exc.DocumentLoadError("bad pdf")
