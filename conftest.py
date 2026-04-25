"""Root pytest configuration.

Automatically skips tests that require external services when those services
are not available.
"""

from __future__ import annotations

import socket

import pytest


def _ollama_reachable() -> bool:
    """Return True if a local Ollama daemon is listening on port 11434."""
    try:
        with socket.create_connection(("localhost", 11434), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live-service tests when the required service is not available."""
    skip_ollama = pytest.mark.skip(
        reason="Ollama daemon not reachable on localhost:11434"
    )

    ollama_available = _ollama_reachable()

    for item in items:
        if item.get_closest_marker("requires_ollama") and not ollama_available:
            item.add_marker(skip_ollama)
