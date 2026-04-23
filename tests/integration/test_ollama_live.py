"""Live integration tests for the Ollama provider.

Run only when a local Ollama daemon is available:
    pytest -m requires_ollama
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_ollama


async def test_ollama_live_hello() -> None:
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    provider = OllamaQwenProvider()
    resp = await provider.acomplete("Say 'ok' and nothing else.")
    assert "ok" in resp.content.lower()


async def test_ollama_live_response_fields() -> None:
    from askbook.providers.ollama_qwen import OllamaQwenProvider

    provider = OllamaQwenProvider()
    resp = await provider.acomplete("Say 'hi'.")
    assert resp.provider == "ollama"
    assert resp.model != ""
    assert resp.usage.total_tokens > 0
    assert resp.latency_ms > 0
