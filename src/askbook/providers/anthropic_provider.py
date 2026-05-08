"""Anthropic Messages API provider via HTTP."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator

import httpx

from askbook.core.exceptions import ProviderError, ProviderTimeoutError
from askbook.core.models import LLMResponse, TokenUsage
from askbook.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """LLM provider using Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = (
            api_key if api_key is not None else os.environ["ANTHROPIC_API_KEY"]
        )
        self._timeout = timeout
        self._endpoint = "https://api.anthropic.com/v1/messages"

    @property
    def provider_name(self) -> str:
        return "anthropic"

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
                prompt, temperature=temperature, max_tokens=max_tokens
            )
        )

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        eff_temp: object = kwargs.get("temperature", self._temperature)
        eff_tokens: object = kwargs.get("max_tokens", self._max_tokens)
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": eff_temp,
            "max_tokens": eff_tokens,
            "stream": True,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST", self._endpoint, json=body, headers=headers
                ) as resp,
            ):
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise ProviderError(
                        self.provider_name,
                        f"HTTP {exc.response.status_code}: {exc.response.text}",
                    ) from exc
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                yield text
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                self.provider_name, f"stream timed out: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _do_acomplete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        eff_temp = temperature if temperature is not None else self._temperature
        eff_tokens = max_tokens if max_tokens is not None else self._max_tokens
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": eff_temp,
            "max_tokens": eff_tokens,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self.provider_name, str(exc)) from exc

        latency_ms = (time.monotonic() - t0) * 1000.0

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                self.provider_name,
                f"HTTP {exc.response.status_code}: {exc.response.text}",
            ) from exc

        data = resp.json()
        try:
            content: str = data["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                self.provider_name, f"unexpected response shape: {exc}"
            ) from exc
        usage_data = data.get("usage", {})
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("input_tokens", 0)
                + usage_data.get("output_tokens", 0),
            )
        else:
            usage = self._make_usage(prompt, content)
        return LLMResponse(
            content=content,
            usage=usage,
            model=data.get("model", self._model),
            provider="anthropic",
            latency_ms=latency_ms,
        )


__all__ = ["AnthropicProvider"]
