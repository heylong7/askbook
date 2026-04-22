"""ServiceRegistry placeholder behavior — concrete builds land in later phases."""

from __future__ import annotations

import pytest

from askbook.core.registry import ServiceRegistry


def test_registry_singletons_not_yet_bound() -> None:
    reg = ServiceRegistry()
    with pytest.raises(LookupError):
        reg.get_llm()


def test_registry_supports_manual_binding() -> None:
    reg = ServiceRegistry()

    sentinel = object()
    reg.bind("llm", sentinel)
    assert reg.get("llm") is sentinel


def test_registry_build_llm_not_implemented_yet() -> None:
    reg = ServiceRegistry()
    with pytest.raises(NotImplementedError, match="Phase 2"):
        reg.build_llm(config=None)
