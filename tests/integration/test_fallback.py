"""Integration tests for FallbackProvider."""

from __future__ import annotations

import contextlib

import pytest

from askbook.core.exceptions import ProviderFallbackExhaustedError
from askbook.core.models import LLMResponse
from askbook.providers.base import FallbackProvider, ProviderTimeoutError
from askbook.providers.stub import StubLLMProvider


class FailingProvider(StubLLMProvider):
    """A provider that always times out."""

    def __init__(self, name: str = "failing") -> None:
        super().__init__(model=name)
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise ProviderTimeoutError(self._name, "simulated timeout")


def test_fallback_chain_exhausted() -> None:
    """All providers fail -> ProviderFallbackExhaustedError."""
    fb = FallbackProvider([FailingProvider("p1"), FailingProvider("p2")])
    with pytest.raises(ProviderFallbackExhaustedError, match="2 providers failed"):
        fb.complete("test")


def test_fallback_chain_switches_to_next() -> None:
    """First provider fails, second succeeds."""
    fb = FallbackProvider([FailingProvider("bad"), StubLLMProvider(model="good")])
    result = fb.complete("test prompt")
    assert isinstance(result, LLMResponse)
    assert result.content  # stub returns something


def test_fallback_chain_empty_providers() -> None:
    """Empty provider list raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        FallbackProvider([])


def test_fallback_provider_name() -> None:
    fb = FallbackProvider([StubLLMProvider()])
    assert fb.provider_name == "fallback"


def test_fallback_count_incremented() -> None:
    """fallback_count reflects number of providers skipped before
    a successful (or exhausted) call."""
    # One failure before success => fallback_count == 1
    fb = FallbackProvider(
        [FailingProvider("bad1"), FailingProvider("bad2"), StubLLMProvider(model="ok")]
    )
    fb.complete("test")
    assert fb.fallback_count == 2

    # All fail => fallback_count still reflects each fallback attempt
    fb2 = FallbackProvider([FailingProvider("x"), FailingProvider("y")])
    with contextlib.suppress(ProviderFallbackExhaustedError):
        fb2.complete("test")
    assert fb2.fallback_count == 1  # 2 providers, 1 fallback after first fails

    # Success on first try => 0
    fb3 = FallbackProvider([StubLLMProvider(model="direct")])
    fb3.complete("test")
    assert fb3.fallback_count == 0


@pytest.mark.requires_dashscope
def test_dashscope_primary_falls_back_to_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """DashScopeQwenProvider times out -> FallbackProvider uses StubLLMProvider.

    Uses monkeypatch to simulate a ProviderTimeoutError from the real
    DashScopeQwenProvider, then verifies the stub fallback returns a result.
    """
    from askbook.providers.dashscope_qwen import DashScopeQwenProvider

    primary = DashScopeQwenProvider(api_key="test-ds-key")
    # Simulate DashScope timeout
    monkeypatch.setattr(
        primary,
        "complete",
        lambda *a, **kw: (_ for _ in ()).throw(
            ProviderTimeoutError("dashscope", "simulated gateway timeout")
        ),
    )
    fallback = StubLLMProvider(canned_response="fallback-answer", model="fallback")
    fb = FallbackProvider([primary, fallback])

    result = fb.complete("test prompt")
    assert result.content == "fallback-answer"
    assert fb.fallback_count == 1


@pytest.mark.requires_ollama
def test_dashscope_falls_back_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """DashScope timeout -> OllamaQwenProvider returns a valid response.

    Simulates a DashScope failure and verifies the real Ollama fallback
    path produces a non-empty result with fallback_count == 1.
    """
    from askbook.providers.dashscope_qwen import DashScopeQwenProvider
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    primary = DashScopeQwenProvider(api_key="test-ds-key")
    monkeypatch.setattr(
        primary,
        "complete",
        lambda *a, **kw: (_ for _ in ()).throw(
            ProviderTimeoutError("dashscope", "simulated gateway timeout")
        ),
    )
    fallback = OllamaQwenProvider()
    fb = FallbackProvider([primary, fallback])

    result = fb.complete("Say hello in one word.")
    assert result.content
    assert fb.fallback_count == 1
    assert fb.retries_per_task == 1


def test_registry_build_fallback_chain() -> None:
    """ServiceRegistry.build_fallback_chain returns FallbackProvider when
    fallback_chain configured."""
    from askbook.config.schema import LLMConfig, LLMFallbackItem
    from askbook.core.registry import ServiceRegistry

    reg = ServiceRegistry()
    config = LLMConfig(
        provider="stub",
        model="test",
        fallback_chain=[
            LLMFallbackItem(provider="stub", model="fallback-model"),
        ],
    )
    result = reg.build_fallback_chain(config)
    assert isinstance(result, FallbackProvider)
