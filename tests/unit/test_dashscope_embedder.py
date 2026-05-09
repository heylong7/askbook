"""Tests for DashScopeEmbedder."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from askbook.core.interfaces import EmbedderProtocol
from askbook.embeddings.dashscope_embedder import DashScopeEmbedder


def test_protocol_conformance():
    e = DashScopeEmbedder(api_key="test-key")
    assert isinstance(e, EmbedderProtocol)


def test_model_name_default():
    e = DashScopeEmbedder(api_key="test-key")
    assert e.model_name == "text-embedding-v3"


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    e = DashScopeEmbedder()
    assert e._api_key == "env-key"


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    e = DashScopeEmbedder(api_key="explicit-key")
    assert e._api_key == "explicit-key"


def test_embed_query_returns_vector():
    e = DashScopeEmbedder(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.output = {"embeddings": [{"embedding": [0.1, 0.2, 0.3]}]}

    with patch("dashscope.TextEmbedding") as mock_te:
        mock_te.call.return_value = mock_resp
        result = e.embed_query("hello")
    assert result == [0.1, 0.2, 0.3]
    assert e.dimension == 3


def test_embed_batch_returns_multiple_vectors():
    e = DashScopeEmbedder(api_key="test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.output = {
        "embeddings": [
            {"embedding": [1.0, 2.0]},
            {"embedding": [3.0, 4.0]},
        ]
    }

    with patch("dashscope.TextEmbedding") as mock_te:
        mock_te.call.return_value = mock_resp
        results = e.embed_batch(["a", "b"])
    assert len(results) == 2
    assert results[0] == [1.0, 2.0]
    assert results[1] == [3.0, 4.0]


def test_dimension_lazy_init():
    """dimension is None until first API call."""
    e = DashScopeEmbedder(api_key="test-key")
    assert e._dimension is None
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.output = {"embeddings": [{"embedding": [0.0] * 1024}]}

    with patch("dashscope.TextEmbedding") as mock_te:
        mock_te.call.return_value = mock_resp
        e.embed_query("test")
    assert e.dimension == 1024
