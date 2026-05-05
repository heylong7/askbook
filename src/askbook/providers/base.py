"""Base classes for LLM providers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from askbook.core.exceptions import ProviderTimeoutError
from askbook.core.models import LLMResponse, TokenUsage

T = TypeVar("T")


class RetryMixin:
    """Mixin that adds async retry logic for transient timeout errors."""

    max_retries: int = 2
    backoff_seconds: float = 0.2

    async def _retry_async(self, func: Callable[[], Awaitable[T]]) -> T:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await func()
            except ProviderTimeoutError as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(self.backoff_seconds * (2**attempt))
        raise last_exc  # type: ignore[misc]


class TokenCountingMixin:
    """Mixin that provides naive token counting via character heuristics."""

    def _make_usage(self, prompt: str, completion: str) -> TokenUsage:
        pt = max(1, len(prompt) // 4)
        ct = max(1, len(completion) // 4)
        return TokenUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        )


class BaseLLMProvider(RetryMixin, TokenCountingMixin):
    """Abstract base for all LLM provider implementations."""

    @property
    def provider_name(self) -> str:
        raise NotImplementedError

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class FallbackProvider:
    """Tries providers in order; raises ProviderFallbackExhaustedError if all fail.

    Callers can inspect ``fallback_count`` after ``complete()`` returns
    to observe how many times the chain fell back to the next provider.
    """

    def __init__(self, providers: list[BaseLLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self._providers = providers
        self._fallback_count = 0

    @property
    def provider_name(self) -> str:
        return "fallback"

    @property
    def fallback_count(self) -> int:
        """How many providers were skipped in the last ``complete()`` call."""
        return self._fallback_count

    @property
    def retries_per_task(self) -> int:
        """Alias for ``fallback_count`` — matches trace attribute convention."""
        return self._fallback_count

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from askbook.core.exceptions import ProviderFallbackExhaustedError

        self._fallback_count = 0
        last_error: Exception | None = None
        for i, provider in enumerate(self._providers):
            try:
                return provider.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                if i < len(self._providers) - 1:
                    self._fallback_count += 1
                last_error = exc
                continue
        raise ProviderFallbackExhaustedError(
            self.provider_name,
            f"All {len(self._providers)} providers failed",
        ) from last_error


__all__ = [
    "BaseLLMProvider",
    "FallbackProvider",
    "ProviderTimeoutError",
    "RetryMixin",
    "TokenCountingMixin",
]
