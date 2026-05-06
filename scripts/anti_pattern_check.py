"""Static analysis script scanning for 6 anti-patterns per DEV_SPEC 30.3.

Usage:
    uv run python scripts/anti_pattern_check.py

Exit codes:
    0 — all checks pass (no violations)
    1 — violations found (details printed to stdout)
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "askbook"

FORBIDDEN_RETRIEVAL_FIELDS = {"raw_text", "full_content", "page_content"}
MAX_MCP_TOOLS = 6


@dataclass(frozen=True)
class Violation:
    pattern: str
    location: str
    detail: str


# ---------------------------------------------------------------------------
# Check 1: RetrievalResult carrying forbidden fields
# ---------------------------------------------------------------------------


def check_retrieval_result(file_path: Path) -> list[Violation]:
    """AST-parse *models.py* and verify RetrievalResult has no forbidden fields."""
    if not file_path.exists():
        return [
            Violation(
                "RetrievalResult forbidden fields", str(file_path), "File not found"
            )
        ]

    source = file_path.read_text("utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [
            Violation(
                "RetrievalResult forbidden fields",
                str(file_path),
                f"Syntax error: {exc}",
            )
        ]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RetrievalResult":
            fields: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(
                    item.target, ast.Name
                ):
                    fields.add(item.target.id)
            found = FORBIDDEN_RETRIEVAL_FIELDS & fields
            if found:
                return [
                    Violation(
                        "RetrievalResult forbidden fields",
                        str(file_path),
                        f"Forbidden field(s) found: {', '.join(sorted(found))}",
                    )
                ]
            return []  # Clean

    return [
        Violation(
            "RetrievalResult forbidden fields",
            str(file_path),
            "class RetrievalResult not found",
        )
    ]


# ---------------------------------------------------------------------------
# Check 2: ToolResponse missing source_ids on status:success
# ---------------------------------------------------------------------------


def check_tool_response_source_ids(file_path: Path) -> list[Violation]:
    """Check that ``source_ids`` identifier appears in *contracts.py*."""
    if not file_path.exists():
        return [Violation("ToolResponse source_ids", str(file_path), "File not found")]

    content = file_path.read_text("utf-8")
    if "source_ids" in content:
        return []
    return [
        Violation(
            "ToolResponse source_ids",
            str(file_path),
            "Missing source_ids in ToolResponse definition",
        )
    ]


# ---------------------------------------------------------------------------
# Check 3: Missing original_query in QuerySpan
# ---------------------------------------------------------------------------


def check_query_span_original_query(file_path: Path) -> list[Violation]:
    """Check that ``original_query`` identifier appears in *schema.py*."""
    if not file_path.exists():
        return [Violation("QuerySpan original_query", str(file_path), "File not found")]

    content = file_path.read_text("utf-8")
    if "original_query" in content:
        return []
    return [
        Violation(
            "QuerySpan original_query",
            str(file_path),
            "Missing original_query in QuerySpan definition",
        )
    ]


# ---------------------------------------------------------------------------
# Check 4: Mutable global state shared between pipeline nodes
# ---------------------------------------------------------------------------


def _get_pipeline_node_files(src_root: Path) -> list[Path]:
    """Collect *.py* files whose path contains *pipeline* or stem contains *node*."""
    files: list[Path] = []
    for py_file in sorted(src_root.rglob("*.py")):
        try:
            rel = py_file.relative_to(src_root)
        except ValueError:
            continue
        if "pipeline" in str(rel).lower() or "node" in py_file.stem.lower():
            files.append(py_file)
    return files


def check_mutable_global_state(file_paths: list[Path]) -> list[Violation]:
    """AST-scan pipeline/node files for module-level non-constant assignments.

    Skips:
    - All-uppercase names (conventional constants)
    - Names starting with ``_`` (private/internal convention)
    - Dunder names (``__all__``, ``__version__``, etc.)
    """
    violations: list[Violation] = []
    for file_path in file_paths:
        if not file_path.exists():
            continue
        source = file_path.read_text("utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.target is not None:
                targets = [node.target]

            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                name = t.id
                # Skip constants, private names, and dunders
                if (
                    name.isupper()
                    or name.startswith("_")
                    or (name.startswith("__") and name.endswith("__"))
                ):
                    continue
                violations.append(
                    Violation(
                        "Mutable global state",
                        str(file_path),
                        f"Module-level non-constant assignment: {name}",
                    )
                )
    return violations


# ---------------------------------------------------------------------------
# Check 5: MCP tool count exceeding cap
# ---------------------------------------------------------------------------


def check_mcp_tool_count(file_path: Path) -> list[Violation]:
    """AST-count ``TOOL_REGISTRY`` dict keys in *tools.py*.

    Handles both plain (``Assign``) and type-annotated (``AnnAssign``) assignments.
    """
    if not file_path.exists():
        return [Violation("MCP tool count", str(file_path), "File not found")]

    source = file_path.read_text("utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Violation("MCP tool count", str(file_path), f"Syntax error: {exc}")]

    for node in ast.walk(tree):
        # Handle Assign: TOOL_REGISTRY = { ... }
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TOOL_REGISTRY":
                    return _count_tool_registry_dict(node.value)
        # Handle AnnAssign: TOOL_REGISTRY: dict[...] = { ... }
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "TOOL_REGISTRY"
        ):
            if node.value is not None:
                return _count_tool_registry_dict(node.value)
            return []  # Type-only annotation, no value — assume pass
    return [Violation("MCP tool count", str(file_path), "TOOL_REGISTRY not found")]


def _count_tool_registry_dict(value: ast.expr) -> list[Violation]:
    """Return violations if *value* is a dict with > MAX_MCP_TOOLS keys."""
    if isinstance(value, ast.Dict):
        count = len(value.keys)
        if count > MAX_MCP_TOOLS:
            return [
                Violation(
                    "MCP tool count",
                    "",
                    f"TOOL_REGISTRY has {count} tools, exceeds cap of {MAX_MCP_TOOLS}",
                )
            ]
        return []  # Within cap
    return []  # Non-dict value — can't determine, assume pass


# ---------------------------------------------------------------------------
# Check 6: Serial LLM-judge execution (should use asyncio.gather + Semaphore)
# ---------------------------------------------------------------------------


def check_llm_judge_parallel(file_path: Path) -> list[Violation]:
    """Check *llm_judge.py* contains both ``asyncio.gather`` and ``Semaphore``."""
    if not file_path.exists():
        return [Violation("LLM-judge parallelism", str(file_path), "File not found")]

    content = file_path.read_text("utf-8")
    has_gather = "asyncio.gather" in content
    has_semaphore = "Semaphore" in content

    if has_gather and has_semaphore:
        return []

    missing = []
    if not has_gather:
        missing.append("asyncio.gather")
    if not has_semaphore:
        missing.append("Semaphore")
    return [
        Violation(
            "LLM-judge parallelism",
            str(file_path),
            f"Missing parallel execution patterns: {', '.join(missing)}",
        )
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_checks(src_root: Path | None = None) -> list[tuple[str, str, list[Violation]]]:
    """Run all 6 anti-pattern checks.

    Returns a list of *(check_name, status, violations)* tuples.
    """
    if src_root is None:
        src_root = SRC_ROOT

    results: list[tuple[str, str, list[Violation]]] = []

    # Check 1
    violations = check_retrieval_result(src_root / "core" / "models.py")
    results.append(
        (
            "RetrievalResult forbidden fields",
            "PASS" if not violations else "FAIL",
            violations,
        )
    )

    # Check 2
    violations = check_tool_response_source_ids(
        src_root / "mcp_server" / "contracts.py"
    )
    results.append(
        ("ToolResponse source_ids", "PASS" if not violations else "FAIL", violations)
    )

    # Check 3
    violations = check_query_span_original_query(
        src_root / "observability" / "schema.py"
    )
    results.append(
        ("QuerySpan original_query", "PASS" if not violations else "FAIL", violations)
    )

    # Check 4
    node_files = _get_pipeline_node_files(src_root)
    violations = check_mutable_global_state(node_files)
    results.append(
        ("Mutable global state", "PASS" if not violations else "FAIL", violations)
    )

    # Check 5
    violations = check_mcp_tool_count(src_root / "mcp_server" / "tools.py")
    results.append(("MCP tool count", "PASS" if not violations else "FAIL", violations))

    # Check 6
    violations = check_llm_judge_parallel(
        src_root / "evaluation" / "metrics" / "llm_judge.py"
    )
    results.append(
        ("LLM-judge parallelism", "PASS" if not violations else "FAIL", violations)
    )

    return results


def main() -> int:
    """Run all checks, print results, return exit code (0 = all pass)."""
    results = run_checks()
    all_passed = True

    for name, status, violations in results:
        if status == "PASS":
            print(f"[PASS] {name}")
        else:
            all_passed = False
            print(f"[FAIL] {name}")
            for v in violations:
                print(f"       Location: {v.location}")
                print(f"       Detail:   {v.detail}")

    print()
    total = len(results)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
