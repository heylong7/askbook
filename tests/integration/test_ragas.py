"""Integration tests for Ragas metrics wrapper."""

from __future__ import annotations

from askbook.evaluation.metrics.ragas_wrapper import RagasScores, compute_ragas_scores
from askbook.providers.stub import StubLLMProvider


def test_compute_ragas_scores_returns_zeros_when_ragas_not_installed() -> None:
    """When ragas is not installed, all scores return 0.0."""
    llm = StubLLMProvider(model="test")
    scores = compute_ragas_scores(
        questions=["q1"],
        answers=["a1"],
        contexts=[["c1"]],
        ground_truths=["gt1"],
        llm=llm,
    )
    assert isinstance(scores, RagasScores)
    assert scores.faithfulness == 0.0
    assert scores.answer_relevancy == 0.0
    assert scores.context_precision == 0.0


def test_compute_ragas_scores_empty_inputs() -> None:
    """Empty inputs return zeros (ragas not installed fallback)."""
    llm = StubLLMProvider(model="test")
    scores = compute_ragas_scores(
        questions=[],
        answers=[],
        contexts=[],
        ground_truths=[],
        llm=llm,
    )
    assert scores.faithfulness == 0.0
    assert scores.answer_relevancy == 0.0
    assert scores.context_precision == 0.0


def test_answer_runner_integration() -> None:
    """run_answer_eval produces report with ragas + llm_judge scores."""
    from askbook.evaluation.answer_runner import run_answer_eval
    from askbook.evaluation.datasets import QADataset

    llm = StubLLMProvider(model="test", canned_response="Score: 4")
    dataset = QADataset.model_validate(
        {
            "_meta": {"version": "1.0", "collection": "demo"},
            "items": [
                {
                    "qid": "Q001",
                    "question": "What is RAG?",
                    "ground_truth": "RAG is Retrieval Augmented Generation.",
                    "relevant_chunk_ids": ["c1"],
                    "tags": ["empty_source"],
                },
                {
                    "qid": "Q002",
                    "question": "What is Python?",
                    "ground_truth": "Python is a programming language.",
                    "relevant_chunk_ids": ["c2"],
                    "tags": ["empty_source"],
                },
            ],
        }
    )

    report = run_answer_eval(
        llm=llm,
        dataset=dataset,
        answers=["RAG is...", "Python is..."],
        contexts=[["context1"], ["context2"]],
    )

    assert report.ragas is not None
    assert report.ragas.faithfulness == 0.0  # ragas not installed
    assert len(report.llm_judge_scores) == 2
    assert report.mean_faithfulness > 0.0
