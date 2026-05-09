"""Unit tests for AnthropicProvider."""

from __future__ import annotations

import httpx
import pytest
import respx

from askbook.core.exceptions import ProviderError, ProviderTimeoutError
from askbook.core.interfaces import LLMProviderProtocol
from askbook.core.models import LLMResponse
from askbook.providers.anthropic_provider import AnthropicProvider


def test_provider_name() -> None:
    """provider_name returns 'anthropic'."""
    provider = AnthropicProvider(api_key="test-key")
    assert provider.provider_name == "anthropic"


def test_conforms_to_protocol() -> None:
    """AnthropicProvider satisfies LLMProviderProtocol."""
    provider = AnthropicProvider(api_key="test-key")
    assert isinstance(provider, LLMProviderProtocol)


def test_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads ANTHROPIC_API_KEY from environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    provider = AnthropicProvider()
    assert provider._api_key == "env-key"  # pragma: allowlist secret


def test_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit api_key arg overrides env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    provider = AnthropicProvider(api_key="explicit-key")
    assert provider._api_key == "explicit-key"  # pragma: allowlist secret


@respx.mock
def test_complete_returns_llm_response() -> None:
    """complete() returns valid LLMResponse from mock Anthropic API."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Hello from Claude"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "model": "claude-sonnet-4-6",
            },
        )
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    result = provider.complete("Hello")
    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from Claude"
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-6"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.latency_ms >= 0


@respx.mock
def test_complete_raises_on_timeout() -> None:
    """complete() raises ProviderTimeoutError on httpx.TimeoutException."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    with pytest.raises(ProviderTimeoutError, match="anthropic"):
        provider.complete("test")


@respx.mock
def test_complete_raises_on_unauthorized() -> None:
    """complete() raises ProviderError on 401."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )
    provider = AnthropicProvider(api_key="bad-key")
    provider.max_retries = 0
    with pytest.raises(ProviderError, match="anthropic"):
        provider.complete("test")


@respx.mock
async def test_acomplete_matches_complete() -> None:
    """acomplete() returns same content as complete()."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "async response"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "model": "claude-sonnet-4-6",
            },
        )
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    resp = await provider.acomplete("test")
    assert resp.content == "async response"
    assert resp.provider == "anthropic"


@respx.mock
def test_complete_fallback_token_counting() -> None:
    """Falls back to _make_usage when API returns no usage field."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "no usage data"}],
                "model": "claude-sonnet-4-6",
            },
        )
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    result = provider.complete("test prompt")
    assert result.content == "no usage data"
    assert result.usage.prompt_tokens >= 1
    assert result.usage.completion_tokens >= 1
    assert result.usage.total_tokens > 0


@respx.mock
def test_complete_raises_on_malformed_response() -> None:
    """Raises ProviderError when response is missing content."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json={})
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    with pytest.raises(ProviderError, match="anthropic"):
        provider.complete("test")


@respx.mock
async def test_astream_yields_chunks() -> None:
    """astream() yields text chunks from SSE stream."""
    hello = (
        '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}'
    )
    world = (
        '{"type":"content_block_delta","delta":{"type":"text_delta","text":" world"}}'
    )
    sse_lines = (
        "event: content_block_delta\n"
        f"data: {hello}\n"
        "\n"
        "event: content_block_delta\n"
        f"data: {world}\n"
        "\n"
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, content=sse_lines)
    )
    provider = AnthropicProvider(api_key="test-key")
    provider.max_retries = 0
    chunks = [c async for c in provider.astream("test")]
    assert chunks == ["Hello", " world"]
