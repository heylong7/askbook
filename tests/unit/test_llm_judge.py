"""Unit tests for LLM-judge scoring."""

from __future__ import annotations

from askbook.evaluation.metrics.llm_judge import (
    JudgeScore,
    _parse_judge_response,
    judge_faithfulness,
)
from askbook.providers.stub import StubLLMProvider


def test_parse_judge_response_extracts_first_digit() -> None:
    """_parse_judge_response extracts the first digit from LLM output."""
    result = _parse_judge_response("Score: 4 - the answer is faithful.")
    assert result.score == 0.8  # 4/5
    assert "Score: 4" in result.reasoning


def test_parse_judge_response_no_digit_returns_zero() -> None:
    """No digit in response -> score 0.0."""
    result = _parse_judge_response("The answer appears correct.")
    assert result.score == 0.0


def test_parse_judge_response_caps_at_one() -> None:
    """Score above 1.0 is capped."""
    result = _parse_judge_response("7")  # 7/5 = 1.4
    assert result.score == 1.0


def test_judge_faithfulness_uses_semaphore_parallelism() -> None:
    """judge_faithfulness uses asyncio.gather for concurrent LLM calls."""
    llm = StubLLMProvider(model="test", canned_response="Score: 5")
    questions = ["q1", "q2", "q3"]
    answers = ["a1", "a2", "a3"]
    contexts = [["c1"], ["c2"], ["c3"]]

    results = judge_faithfulness(llm, questions, answers, contexts, max_concurrency=2)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, JudgeScore)
        assert r.score == 1.0  # "Score: 5" -> 5/5 = 1.0


def test_judge_faithfulness_empty_input() -> None:
    """Empty input returns empty list."""
    llm = StubLLMProvider(model="test")
    results = judge_faithfulness(llm, [], [], [])
    assert results == []


def test_judge_faithfulness_with_llm_none_context() -> None:
    """Context with empty ctx returns score zero (stub response)."""
    llm = StubLLMProvider(model="test", canned_response="Score: 3")
    results = judge_faithfulness(
        llm,
        questions=["q1"],
        answers=["a1"],
        contexts=[[]],
        max_concurrency=1,
    )
    # stub returns "Score: 3", parsed as 3/5 = 0.6
    assert results[0].score == 0.6
