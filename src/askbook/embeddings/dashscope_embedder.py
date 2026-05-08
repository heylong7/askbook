"""DashScope TextEmbedding API provider."""

from __future__ import annotations

import os

from askbook.core.exceptions import ProviderError


class DashScopeEmbedder:
    """DashScope text-embedding provider.

    Lazily computes dimension from the first API response, so
    dimension is not available until an embed_* method is called.
    """

    model_name: str

    def __init__(
        self,
        *,
        model: str = "text-embedding-v3",
        api_key: str | None = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._dimension: int | None = None
        try:
            import dashscope

            dashscope.api_key = self._api_key
        except ImportError:
            pass

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
        try:
            from dashscope import TextEmbedding
        except ImportError as exc:
            raise ProviderError(
                "dashscope", "dashscope SDK not installed"
            ) from exc

        resp = TextEmbedding.call(model=self.model_name, input=texts)
        if resp.status_code != 200:
            raise ProviderError(
                "dashscope",
                f"HTTP {resp.status_code}: {getattr(resp, 'message', '')}",
            )
        embeddings = [
            item["embedding"] for item in resp.output["embeddings"]
        ]
        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
        return embeddings


__all__ = ["DashScopeEmbedder"]
