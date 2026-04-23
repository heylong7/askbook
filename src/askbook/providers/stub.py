"""Deterministic stub LLM provider for unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from askbook.core.models import LLMResponse
from askbook.providers.base import BaseLLMProvider


class StubLLMProvider(BaseLLMProvider):
    """A predictable, side-effect-free LLM provider for testing."""

    def __init__(
        self,
        *,
        canned_response: str = "stub-answer",
        model: str = "stub-model",
    ) -> None:
        self._canned = canned_response
        self._model = model

    @property
    def provider_name(self) -> str:
        return "stub"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        usage = self._make_usage(prompt, self._canned)
        return LLMResponse(
            content=self._canned,
            usage=usage,
            model=self._model,
            provider="stub",
            latency_ms=0.0,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return self.complete(prompt, temperature=temperature, max_tokens=max_tokens)

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        yield self._canned


__all__ = ["StubLLMProvider"]
