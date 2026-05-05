"""Unit tests for harness health metrics."""

from __future__ import annotations

import pytest

from askbook.evaluation.metrics.harness_metrics import (
    completion_rate,
    compute_harness_metrics,
    cost_per_task,
    pass_at_1,
    retries_per_task,
)


class TestCompletionRate:
    def test_perfect(self) -> None:
        assert completion_rate(10, 10) == 1.0

    def test_partial(self) -> None:
        assert completion_rate(7, 10) == 0.7

    def test_zero_total(self) -> None:
        assert completion_rate(0, 0) == 1.0


class TestRetriesPerTask:
    def test_with_retries(self) -> None:
        assert retries_per_task(5, 10) == 0.5

    def test_no_retries(self) -> None:
        assert retries_per_task(0, 10) == 0.0

    def test_zero_tasks(self) -> None:
        assert retries_per_task(0, 0) == 0.0


class TestPassAt1:
    def test_all_pass(self) -> None:
        assert pass_at_1(8, 8) == 1.0

    def test_some_pass(self) -> None:
        assert pass_at_1(6, 10) == 0.6

    def test_zero_total(self) -> None:
        assert pass_at_1(0, 0) == 1.0


class TestCostPerTask:
    def test_typical(self) -> None:
        assert cost_per_task(0.03, 3) == pytest.approx(0.01)

    def test_zero_tasks(self) -> None:
        assert cost_per_task(0.0, 0) == 0.0


class TestComputeHarnessMetrics:
    def test_returns_all_four(self) -> None:
        result = compute_harness_metrics(
            mcp_success_count=9,
            mcp_total_count=10,
            total_retries=3,
            task_count=10,
            golden_pass_count=8,
            golden_total_count=10,
            total_cost_cny=0.50,
        )
        assert result == {
            "completion_rate": 0.9,
            "retries_per_task": 0.3,
            "pass_at_1": 0.8,
            "cost_per_task": 0.05,
        }
