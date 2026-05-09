"""Unit tests for OpenAIProvider."""

from __future__ import annotations

import httpx
import pytest
import respx

from askbook.core.exceptions import ProviderError, ProviderTimeoutError
from askbook.core.interfaces import LLMProviderProtocol
from askbook.core.models import LLMResponse
from askbook.providers.openai_provider import OpenAIProvider


def test_provider_name() -> None:
    """provider_name returns 'openai'."""
    provider = OpenAIProvider(api_key="test-key")
    assert provider.provider_name == "openai"


def test_conforms_to_protocol() -> None:
    """OpenAIProvider satisfies LLMProviderProtocol."""
    provider = OpenAIProvider(api_key="test-key")
    assert isinstance(provider, LLMProviderProtocol)


def test_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads OPENAI_API_KEY from environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = OpenAIProvider()
    assert provider._api_key == "env-key"  # pragma: allowlist secret


def test_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit api_key arg overrides env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    provider = OpenAIProvider(api_key="explicit-key")
    assert provider._api_key == "explicit-key"  # pragma: allowlist secret


@respx.mock
def test_complete_returns_llm_response() -> None:
    """complete() returns valid LLMResponse from mock OpenAI API."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Hello from OpenAI"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model": "gpt-4o",
            },
        )
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    result = provider.complete("Hello")
    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from OpenAI"
    assert result.provider == "openai"
    assert result.model == "gpt-4o"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
    assert result.latency_ms >= 0


@respx.mock
def test_complete_raises_on_timeout() -> None:
    """complete() raises ProviderTimeoutError on httpx.TimeoutException."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    with pytest.raises(ProviderTimeoutError, match="openai"):
        provider.complete("test")


@respx.mock
def test_complete_raises_on_unauthorized() -> None:
    """complete() raises ProviderError on 401."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )
    provider = OpenAIProvider(api_key="bad-key")
    provider.max_retries = 0
    with pytest.raises(ProviderError, match="openai"):
        provider.complete("test")


@respx.mock
async def test_acomplete_matches_complete() -> None:
    """acomplete() returns same content as complete()."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "async response"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
                "model": "gpt-4o",
            },
        )
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    resp = await provider.acomplete("test")
    assert resp.content == "async response"
    assert resp.provider == "openai"


@respx.mock
def test_complete_fallback_token_counting() -> None:
    """complete() falls back to _make_usage when API returns no usage field."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "no usage data"}}],
                "model": "gpt-4o",
            },
        )
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    result = provider.complete("test prompt")
    assert result.content == "no usage data"
    assert result.usage.prompt_tokens >= 1
    assert result.usage.completion_tokens >= 1
    assert result.usage.total_tokens > 0


@respx.mock
def test_complete_raises_on_malformed_response() -> None:
    """complete() raises ProviderError when response is missing choices."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={})
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    with pytest.raises(ProviderError, match="openai"):
        provider.complete("test")


@respx.mock
async def test_astream_yields_chunks() -> None:
    """astream() yields content chunks from SSE stream."""
    sse_lines = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}\n'
        "\n"
        'data: {"choices":[{"delta":{"content":" world"}}]}\n'
        "\n"
        "data: [DONE]\n"
        "\n"
    )
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_lines)
    )
    provider = OpenAIProvider(api_key="test-key")
    provider.max_retries = 0
    chunks = [c async for c in provider.astream("test")]
    assert chunks == ["Hello", " world"]
