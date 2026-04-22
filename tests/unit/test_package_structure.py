"""Verify src-layout package skeleton matches DEV_SPEC Ch 17."""

from __future__ import annotations

import importlib

SUBPACKAGES = [
    "askbook",
    "askbook.core",
    "askbook.config",
    "askbook.providers",
    "askbook.embeddings",
    "askbook.rerankers",
    "askbook.vectorstores",
    "askbook.splitters",
    "askbook.ingestion",
    "askbook.query",
    "askbook.mcp_server",
    "askbook.observability",
    "askbook.evaluation",
    "askbook.evaluation.metrics",
    "askbook.dashboard",
    "askbook.dashboard.pages",
    "askbook.prompts",
    "askbook.utils",
]


def test_all_subpackages_importable() -> None:
    for pkg in SUBPACKAGES:
        importlib.import_module(pkg)


def test_version_exposed() -> None:
    import askbook

    assert isinstance(askbook.__version__, str)
    assert askbook.__version__.count(".") >= 2
