"""Unit tests for DashScopeQwenProvider."""

from __future__ import annotations

import builtins

import pytest

from askbook.providers.dashscope_qwen import DashScopeQwenProvider


def test_provider_name_is_dashscope() -> None:
    provider = DashScopeQwenProvider(api_key="test-key")
    assert provider.provider_name == "dashscope"


def test_init_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    provider = DashScopeQwenProvider()
    assert provider._api_key == "env-key"  # pragma: allowlist secret


def test_init_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    provider = DashScopeQwenProvider(api_key="explicit-key")
    assert provider._api_key == "explicit-key"  # pragma: allowlist secret


def test_complete_raises_without_dashscope_sdk() -> None:
    """complete() raises ProviderTimeoutError when dashscope SDK not installed."""
    provider = DashScopeQwenProvider(api_key="test-key")
    # Mock the import to fail inside complete()
    original_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "dashscope" or name.startswith("dashscope."):
            raise ImportError("No module named 'dashscope'")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mock_import  # type: ignore[assignment]
    try:
        from askbook.providers.base import ProviderTimeoutError

        with pytest.raises(ProviderTimeoutError, match="dashscope SDK not installed"):
            provider.complete("test prompt")
    finally:
        builtins.__import__ = original_import  # type: ignore[assignment]
