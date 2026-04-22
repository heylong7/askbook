"""BGE-M3 embedder — lazy model load so unit tests never trigger downloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel


class BGEM3Embedder:
    """DEV_SPEC Ch 20: query/passage prefix strategy separated."""

    model_name: str = "BAAI/bge-m3"
    QUERY_PREFIX: str = ""
    PASSAGE_PREFIX: str = ""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        normalize: bool = True,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model
        self._normalize = normalize
        self._device = device
        self._batch_size = batch_size
        self._model: BGEM3FlagModel | None = None
        self._dimension = 1024

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_model(self) -> BGEM3FlagModel:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            device = None if self._device == "auto" else self._device
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,
                device=device,
            )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        out = model.encode(
            texts,
            batch_size=self._batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = out["dense_vecs"]
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.QUERY_PREFIX + text])[0]

    def embed_passage(self, text: str) -> list[float]:
        return self._encode([self.PASSAGE_PREFIX + text])[0]

    def embed_batch(
        self, texts: list[str], is_query: bool = False
    ) -> list[list[float]]:
        prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
        return self._encode([prefix + t for t in texts])


__all__ = ["BGEM3Embedder"]
