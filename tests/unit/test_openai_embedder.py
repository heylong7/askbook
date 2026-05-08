"""Unit tests for OpenAIEmbedder."""

from __future__ import annotations

import httpx
import pytest
import respx

from askbook.core.interfaces import EmbedderProtocol
from askbook.embeddings.openai_embedder import OpenAIEmbedder

EMBED_ENDPOINT = "https://api.openai.com/v1/embeddings"


def test_protocol_conformance() -> None:
    """OpenAIEmbedder satisfies EmbedderProtocol."""
    e = OpenAIEmbedder(api_key="test-key")
    assert isinstance(e, EmbedderProtocol)


def test_model_name_default() -> None:
    """Default model_name is text-embedding-3-small."""
    e = OpenAIEmbedder(api_key="test-key")
    assert e.model_name == "text-embedding-3-small"


def test_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reads OPENAI_API_KEY from environment when no explicit key given."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    e = OpenAIEmbedder()
    assert e._api_key == "env-key"  # pragma: allowlist secret


@respx.mock
def test_embed_query_returns_vector() -> None:
    """embed_query() returns a float list from the API response."""
    respx.post(EMBED_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
            },
        )
    )
    e = OpenAIEmbedder(api_key="test-key")
    result = e.embed_query("hello")
    assert len(result) == 3
    assert result == [0.1, 0.2, 0.3]
    assert e.dimension == 3


@respx.mock
def test_embed_passage_returns_vector() -> None:
    """embed_passage() returns a float list from the API response."""
    respx.post(EMBED_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [1.0, 2.0]}],
                "model": "text-embedding-3-small",
            },
        )
    )
    e = OpenAIEmbedder(api_key="test-key")
    result = e.embed_passage("some passage")
    assert len(result) == 2


@respx.mock
def test_embed_batch_returns_multiple_vectors() -> None:
    """embed_batch() returns a list of float lists."""
    respx.post(EMBED_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [1.0, 2.0]},
                    {"embedding": [3.0, 4.0]},
                ],
                "model": "text-embedding-3-small",
            },
        )
    )
    e = OpenAIEmbedder(api_key="test-key")
    results = e.embed_batch(["a", "b"])
    assert len(results) == 2
    assert results[0] == [1.0, 2.0]
    assert results[1] == [3.0, 4.0]


@respx.mock
def test_dimension_lazy_init() -> None:
    """dimension is None until first API call sets it from the response."""
    e = OpenAIEmbedder(api_key="test-key")
    assert e._dimension is None
    respx.post(EMBED_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.0] * 1536}],
                "model": "text-embedding-3-small",
            },
        )
    )
    e.embed_query("test")
    assert e.dimension == 1536
