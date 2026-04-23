"""Base classes for LLM providers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from askbook.core.models import TokenUsage

T = TypeVar("T")


class ProviderTimeoutError(Exception):
    """Raised when a provider call times out."""

    def __init__(self, provider: str, detail: str = "") -> None:
        super().__init__(f"[{provider}] timeout: {detail}")
        self.provider = provider


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


__all__ = [
    "BaseLLMProvider",
    "ProviderTimeoutError",
    "RetryMixin",
    "TokenCountingMixin",
]
