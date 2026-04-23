"""Unit tests for LLM provider infrastructure."""

from __future__ import annotations

import httpx
import pytest
import respx

from askbook.core.interfaces import LLMProviderProtocol
from askbook.core.models import LLMResponse
from askbook.providers.base import (
    ProviderTimeoutError,
    RetryMixin,
    TokenCountingMixin,
)
from askbook.providers.stub import StubLLMProvider

# ---------------------------------------------------------------------------
# StubLLMProvider tests
# ---------------------------------------------------------------------------


def test_stub_conforms_to_protocol() -> None:
    provider = StubLLMProvider()
    assert isinstance(provider, LLMProviderProtocol)


def test_stub_complete_returns_llm_response() -> None:
    provider = StubLLMProvider(canned_response="hello", model="test-model")
    resp = provider.complete("some prompt")
    assert isinstance(resp, LLMResponse)
    assert resp.content == "hello"
    assert resp.model == "test-model"
    assert resp.provider == "stub"
    assert resp.latency_ms == 0.0


def test_stub_complete_usage_is_populated() -> None:
    provider = StubLLMProvider(canned_response="hi")
    resp = provider.complete("a prompt")
    assert resp.usage.prompt_tokens >= 1
    assert resp.usage.completion_tokens >= 1
    expected = resp.usage.prompt_tokens + resp.usage.completion_tokens
    assert resp.usage.total_tokens == expected


async def test_stub_acomplete_matches_complete() -> None:
    provider = StubLLMProvider(canned_response="async-answer")
    sync_resp = provider.complete("test")
    async_resp = await provider.acomplete("test")
    assert sync_resp.content == async_resp.content
    assert sync_resp.provider == async_resp.provider


async def test_stub_astream_yields_canned_response() -> None:
    provider = StubLLMProvider(canned_response="streamed")
    chunks = [chunk async for chunk in provider.astream("prompt")]
    assert chunks == ["streamed"]


# ---------------------------------------------------------------------------
# RetryMixin tests
# ---------------------------------------------------------------------------


async def test_retry_mixin_succeeds_after_one_failure() -> None:
    class _TestProvider(RetryMixin):
        max_retries = 2
        backoff_seconds = 0.0
        call_count = 0

        async def _flaky(self) -> str:
            self.call_count += 1
            if self.call_count < 2:
                raise ProviderTimeoutError("test", "transient")
            return "ok"

    provider = _TestProvider()
    result = await provider._retry_async(provider._flaky)
    assert result == "ok"
    assert provider.call_count == 2


async def test_retry_mixin_raises_after_max_retries() -> None:
    class _AlwaysTimeout(RetryMixin):
        max_retries = 1
        backoff_seconds = 0.0

        async def _always_fail(self) -> str:
            raise ProviderTimeoutError("test", "always fails")

    provider = _AlwaysTimeout()
    with pytest.raises(ProviderTimeoutError, match="always fails"):
        await provider._retry_async(provider._always_fail)


# ---------------------------------------------------------------------------
# TokenCountingMixin tests
# ---------------------------------------------------------------------------


def test_token_counting_mixin_counts_tokens() -> None:
    mixin = TokenCountingMixin()
    usage = mixin._make_usage("hello world", "response text here")
    assert usage.prompt_tokens >= 1
    assert usage.completion_tokens >= 1
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_token_counting_mixin_minimum_one_token() -> None:
    mixin = TokenCountingMixin()
    # Very short strings should still yield at least 1 token
    usage = mixin._make_usage("a", "b")
    assert usage.prompt_tokens == 1
    assert usage.completion_tokens == 1


# ---------------------------------------------------------------------------
# OllamaQwenProvider tests (respx mock)
# ---------------------------------------------------------------------------


@respx.mock
async def test_ollama_acomplete_parses_response() -> None:
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": "hello",
                "done": True,
                "prompt_eval_count": 3,
                "eval_count": 2,
                "model": "qwen2.5:7b",
            },
        )
    )
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    provider = OllamaQwenProvider()
    resp = await provider.acomplete("prompt")
    assert resp.content == "hello"
    assert resp.provider == "ollama"
    assert resp.model == "qwen2.5:7b"
    assert resp.usage.prompt_tokens == 3
    assert resp.usage.completion_tokens == 2
    assert resp.usage.total_tokens == 5


@respx.mock
async def test_ollama_acomplete_raises_on_timeout() -> None:
    respx.post("http://localhost:11434/api/generate").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    # Disable retries so the test runs quickly
    provider = OllamaQwenProvider()
    provider.max_retries = 0

    with pytest.raises(ProviderTimeoutError, match="ollama"):
        await provider.acomplete("prompt")


@respx.mock
async def test_ollama_provider_name() -> None:
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    provider = OllamaQwenProvider()
    assert provider.provider_name == "ollama"


@respx.mock
async def test_ollama_conforms_to_protocol() -> None:
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    provider = OllamaQwenProvider()
    assert isinstance(provider, LLMProviderProtocol)
