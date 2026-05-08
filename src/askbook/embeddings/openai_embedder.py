"""OpenAI Embeddings API provider via HTTP."""

from __future__ import annotations

import os

import httpx

from askbook.core.exceptions import ProviderError


class OpenAIEmbedder:
    """OpenAI text-embedding-3 provider.

    Lazily computes dimension from the first API response, so
    dimension is not available until an embed_* method is called.
    """

    model_name: str

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model
        self._api_key = api_key if api_key is not None else os.environ["OPENAI_API_KEY"]
        self._timeout = timeout
        self._endpoint = "https://api.openai.com/v1/embeddings"
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("dimension not available -- call embed_* first")
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def embed_passage(self, text: str) -> list[float]:
        return self._call_api([text])[0]

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        return self._call_api(texts)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model_name, "input": texts}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(
            self._endpoint, json=body, headers=headers, timeout=self._timeout
        )
        if resp.status_code != 200:
            raise ProviderError(
                "openai", f"HTTP {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings


__all__ = ["OpenAIEmbedder"]
