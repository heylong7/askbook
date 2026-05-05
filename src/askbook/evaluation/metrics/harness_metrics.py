"""Harness health metrics computed from trace events and evaluation results."""

from __future__ import annotations


def completion_rate(success_count: int, total_count: int) -> float:
    """MCP tool calls with status:success / total calls. Target: >= 0.95."""
    if total_count == 0:
        return 1.0
    return success_count / total_count


def retries_per_task(total_retries: int, total_tasks: int) -> float:
    """Average retry count per task. Target: <= 1.2."""
    if total_tasks == 0:
        return 0.0
    return total_retries / total_tasks


def pass_at_1(pass_count: int, total_count: int) -> float:
    """First-pass rate on golden set. Target: >= 0.85."""
    if total_count == 0:
        return 1.0
    return pass_count / total_count


def cost_per_task(total_cost_cny: float, total_tasks: int) -> float:
    """Average token cost (CNY) per successful ask call. Target: <= 0.05."""
    if total_tasks == 0:
        return 0.0
    return total_cost_cny / total_tasks


def compute_harness_metrics(
    *,
    mcp_success_count: int,
    mcp_total_count: int,
    total_retries: int,
    task_count: int,
    golden_pass_count: int,
    golden_total_count: int,
    total_cost_cny: float,
) -> dict[str, float]:
    """Compute all four harness metrics in one call."""
    return {
        "completion_rate": completion_rate(mcp_success_count, mcp_total_count),
        "retries_per_task": retries_per_task(total_retries, task_count),
        "pass_at_1": pass_at_1(golden_pass_count, golden_total_count),
        "cost_per_task": cost_per_task(total_cost_cny, task_count),
    }


__all__ = [
    "completion_rate",
    "retries_per_task",
    "pass_at_1",
    "cost_per_task",
    "compute_harness_metrics",
]
