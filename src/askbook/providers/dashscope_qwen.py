"""DashScope Qwen API provider via HTTP (dashscope SDK)."""

from __future__ import annotations

import os
import time

from askbook.core.models import LLMResponse
from askbook.providers.base import BaseLLMProvider, ProviderTimeoutError


class DashScopeQwenProvider(BaseLLMProvider):
    """DashScope Qwen API provider via HTTP (dashscope SDK)."""

    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        # Try to import and configure dashscope
        try:
            import dashscope

            dashscope.api_key = self._api_key
        except ImportError:
            pass
        self.max_retries = 1
        self.backoff_seconds = 1.0

    @property
    def provider_name(self) -> str:
        return "dashscope"

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            from dashscope import Generation
        except ImportError as exc:
            raise ProviderTimeoutError(
                self.provider_name, "dashscope SDK not installed"
            ) from exc

        t0 = time.perf_counter()
        try:
            resp = Generation.call(
                model=self._model,
                prompt=prompt,
                temperature=(
                    temperature if temperature is not None else self._temperature
                ),
                max_tokens=(max_tokens if max_tokens is not None else self._max_tokens),
                result_format="message",
            )
        except Exception as exc:
            raise ProviderTimeoutError(self.provider_name, str(exc)) from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise ProviderTimeoutError(
                self.provider_name,
                f"HTTP {resp.status_code}: {resp.message}",
            )

        output = resp.output
        text = ""
        if output and output.choices:
            text = output.choices[0].message.content or ""
        usage = self._make_usage(prompt, text)
        return LLMResponse(
            content=text,
            model=self._model,
            usage=usage,
            provider=self.provider_name,
            latency_ms=latency_ms,
        )


__all__ = ["DashScopeQwenProvider"]
