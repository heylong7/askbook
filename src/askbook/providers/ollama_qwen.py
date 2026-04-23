"""Ollama provider using Qwen models via raw HTTP (no SDK)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import httpx

from askbook.core.models import LLMResponse, TokenUsage
from askbook.providers.base import BaseLLMProvider, ProviderTimeoutError

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_TIMEOUT = 60.0


class OllamaQwenProvider(BaseLLMProvider):
    """LLM provider that calls a local Ollama daemon with a Qwen model."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return asyncio.run(
            self.acomplete(prompt, temperature=temperature, max_tokens=max_tokens)
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._retry_async(
            lambda: self._do_acomplete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        payload = self._build_payload(prompt, stream=True)
        url = f"{self._base_url}/api/generate"
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_name, f"stream timed out: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict[str, object]:
        eff_temp = temperature if temperature is not None else self._temperature
        eff_tokens = max_tokens if max_tokens is not None else self._max_tokens
        return {
            "model": self._model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": eff_temp,
                "num_predict": eff_tokens,
            },
        }

    async def _do_acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(
            prompt, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        url = f"{self._base_url}/api/generate"
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.provider_name, str(exc)) from exc
        latency_ms = (time.monotonic() - t0) * 1000.0

        data = response.json()
        content: str = data.get("response", "")
        model: str = data.get("model", self._model)
        fallback_pt = max(1, len(prompt) // 4)
        fallback_ct = max(1, len(content) // 4)
        prompt_tokens: int = int(data.get("prompt_eval_count", fallback_pt))
        completion_tokens: int = int(data.get("eval_count", fallback_ct))

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return LLMResponse(
            content=content,
            usage=usage,
            model=model,
            provider="ollama",
            latency_ms=latency_ms,
        )


__all__ = ["OllamaQwenProvider"]
