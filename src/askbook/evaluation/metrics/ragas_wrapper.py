"""Ragas answer-quality metrics wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from askbook.core.interfaces import LLMProviderProtocol


@dataclass
class RagasScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float


def compute_ragas_scores(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    llm: LLMProviderProtocol,
) -> RagasScores:
    """Compute Ragas metrics. Returns all zeros when ragas is not installed."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError:
        return RagasScores(0.0, 0.0, 0.0)

    from datasets import Dataset as HFDataset  # type: ignore[attr-defined]

    ds = HFDataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    try:
        eval_result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        result = dict(eval_result)
    except Exception:
        return RagasScores(0.0, 0.0, 0.0)

    return RagasScores(
        faithfulness=float(result.get("faithfulness", 0.0)),
        answer_relevancy=float(result.get("answer_relevancy", 0.0)),
        context_precision=float(result.get("context_precision", 0.0)),
    )
