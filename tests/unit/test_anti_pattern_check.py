"""Tests for scripts/anti_pattern_check.py — 8 tests per Task 7.2 Step 2."""

from __future__ import annotations

from pathlib import Path

from scripts.anti_pattern_check import (
    MAX_MCP_TOOLS,
    check_llm_judge_parallel,
    check_mcp_tool_count,
    check_query_span_original_query,
    check_retrieval_result,
    check_tool_response_source_ids,
    run_checks,
)


def _write(file_path: Path, content: str) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Check 1 — RetrievalResult forbidden fields
# ---------------------------------------------------------------------------


def test_retrieval_result_clean_passes(tmp_path: Path) -> None:
    """A clean RetrievalResult (no forbidden fields) should pass."""
    src = _write(
        tmp_path / "core" / "models.py",
        "class RetrievalResult:\n"
        "    chunk_id: str\n"
        "    score: float\n"
        "    snippet: str\n"
        "    metadata: dict\n"
        "    retrieval_method: str\n",
    )
    assert check_retrieval_result(src) == []


def test_retrieval_result_forbidden_field_detected(tmp_path: Path) -> None:
    """A RetrievalResult carrying a forbidden field should be flagged."""
    src = _write(
        tmp_path / "core" / "models.py",
        "class RetrievalResult:\n"
        "    chunk_id: str\n"
        "    raw_text: str\n"  # forbidden field
        "    score: float\n",
    )
    violations = check_retrieval_result(src)
    assert len(violations) == 1
    assert violations[0].pattern == "RetrievalResult forbidden fields"
    assert "raw_text" in violations[0].detail


# ---------------------------------------------------------------------------
# Check 2 — ToolResponse source_ids
# ---------------------------------------------------------------------------


def test_tool_response_source_ids_present(tmp_path: Path) -> None:
    """File containing ``\"source_ids\"`` should pass."""
    src = _write(tmp_path / "contracts.py", "source_ids: list[str] = []\n")
    assert check_tool_response_source_ids(src) == []


# ---------------------------------------------------------------------------
# Check 3 — QuerySpan original_query
# ---------------------------------------------------------------------------


def test_query_span_original_query_present(tmp_path: Path) -> None:
    """File containing ``\"original_query\"`` should pass."""
    src = _write(tmp_path / "schema.py", 'original_query: str = ""\n')
    assert check_query_span_original_query(src) == []


# ---------------------------------------------------------------------------
# Check 5 — MCP tool count
# ---------------------------------------------------------------------------


def test_mcp_tool_count_within_cap(tmp_path: Path) -> None:
    """TOOL_REGISTRY with ≤ MAX_MCP_TOOLS keys should pass."""
    keys = "\n    ".join(f'"{chr(97 + i)}": (int, str),' for i in range(MAX_MCP_TOOLS))
    src = _write(
        tmp_path / "tools.py",
        f"TOOL_REGISTRY = {{\n    {keys}\n}}\n",
    )
    assert check_mcp_tool_count(src) == []


def test_mcp_tool_count_exceeds_cap(tmp_path: Path) -> None:
    """TOOL_REGISTRY with > MAX_MCP_TOOLS keys should be flagged."""
    keys = "\n    ".join(
        f'"{chr(97 + i)}": (int, str),' for i in range(MAX_MCP_TOOLS + 1)
    )
    src = _write(
        tmp_path / "tools.py",
        f"TOOL_REGISTRY = {{\n    {keys}\n}}\n",
    )
    violations = check_mcp_tool_count(src)
    assert len(violations) == 1
    assert violations[0].pattern == "MCP tool count"
    assert "exceeds cap" in violations[0].detail


# ---------------------------------------------------------------------------
# Check 6 — LLM-judge parallelism
# ---------------------------------------------------------------------------


def test_llm_judge_parallel_detected(tmp_path: Path) -> None:
    """File containing both asyncio.gather and Semaphore should pass."""
    src = _write(
        tmp_path / "llm_judge.py",
        "import asyncio\n\n"
        "async def foo():\n"
        "    sem = asyncio.Semaphore(4)\n"
        "    await asyncio.gather(*tasks)\n",
    )
    assert check_llm_judge_parallel(src) == []


# ---------------------------------------------------------------------------
# Main runner — clean codebase returns zero violations
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_clean(tmp_path: Path) -> None:
    """run_checks() against a clean temporary source tree should all PASS."""
    # Check 1: clean RetrievalResult
    _write(
        tmp_path / "core" / "models.py",
        "class RetrievalResult:\n"
        "    chunk_id: str\n"
        "    score: float\n"
        "    snippet: str\n"
        "    metadata: dict\n"
        "    retrieval_method: str\n",
    )
    # Check 2: source_ids present
    _write(tmp_path / "mcp_server" / "contracts.py", "source_ids: list[str] = []\n")
    # Check 3: original_query present
    _write(tmp_path / "observability" / "schema.py", 'original_query: str = ""\n')
    # Check 4: pipeline file with only constants (clean)
    _write(
        tmp_path / "query" / "pipeline.py",
        "CONFIG = 'clean'\nMAX_RETRIES = 3\n",
    )
    # Check 5: TOOL_REGISTRY within cap
    keys = "\n    ".join(f'"{chr(97 + i)}": (int, str),' for i in range(MAX_MCP_TOOLS))
    _write(
        tmp_path / "mcp_server" / "tools.py",
        f"TOOL_REGISTRY = {{\n    {keys}\n}}\n",
    )
    # Check 6: asyncio.gather + Semaphore present
    _write(
        tmp_path / "evaluation" / "metrics" / "llm_judge.py",
        "import asyncio\n"
        "async def foo():\n"
        "    sem = asyncio.Semaphore(4)\n"
        "    await asyncio.gather(*tasks)\n",
    )

    results = run_checks(tmp_path)
    for name, status, violations in results:
        assert status == "PASS", f"{name} FAILED: {violations}"
    assert len(results) == 6
