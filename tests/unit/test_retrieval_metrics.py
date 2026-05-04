"""Tests for retrieval-quality metrics (hit_rate, mrr, recall_at_k, ndcg_at_k)."""

from __future__ import annotations

import math

import pytest

from askbook.evaluation.metrics.retrieval import hit_rate, mrr, ndcg_at_k, recall_at_k


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    [
        (["a", "b", "c"], {"b"}, 3, 1.0),  # hit
        (["a", "b", "c"], {"d"}, 3, 0.0),  # miss
        (["a", "b", "c"], {"c"}, 2, 0.0),  # outside k
        ([], {"a"}, 5, 0.0),  # empty retrieved
        (["a", "b"], set(), 5, 0.0),  # empty relevant -> defined as 0
    ],
)
def test_hit_rate(retrieved, relevant, k, expected):
    assert hit_rate(retrieved, relevant, k) == expected


@pytest.mark.parametrize(
    "retrieved,relevant,k,expected",
    [
        (["a", "b", "c"], {"a"}, 3, 1.0),  # rank 1
        (["a", "b", "c"], {"b"}, 3, 0.5),  # rank 2
        (["a", "b", "c"], {"c"}, 3, 1 / 3),  # rank 3
        (["a", "b", "c"], {"d"}, 3, 0.0),
        (["a", "b", "c"], {"b", "c"}, 3, 0.5),  # first relevant at rank 2
        (["a"], set(), 3, 0.0),
    ],
)
def test_mrr(retrieved, relevant, k, expected):
    assert mrr(retrieved, relevant, k) == pytest.approx(expected)


def test_recall_at_k_partial():
    # 2 of 3 relevant in top-3 -> 2/3
    assert recall_at_k(["a", "b", "c"], {"a", "b", "x"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant_returns_zero():
    assert recall_at_k(["a", "b"], set(), 5) == 0.0


def test_ndcg_at_k_perfect_order_equals_one():
    # All relevant at the top, identical to ideal
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_at_k_no_match_zero():
    assert ndcg_at_k(["a", "b", "c"], {"x"}, 3) == 0.0


def test_ndcg_at_k_known_value():
    # relevant = {"b"}, retrieved order = ["a", "b", "c"], k=3
    # DCG = 1/log2(2+1) = 1/log2(3); IDCG = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(["a", "b", "c"], {"b"}, 3) == pytest.approx(expected)
