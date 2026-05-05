"""Answer quality evaluation orchestrator — Ragas + LLM-judge unified entry point."""

from __future__ import annotations

from dataclasses import dataclass, field

from askbook.core.interfaces import LLMProviderProtocol
from askbook.evaluation.datasets import QADataset
from askbook.evaluation.metrics.llm_judge import JudgeScore, judge_faithfulness
from askbook.evaluation.metrics.ragas_wrapper import RagasScores, compute_ragas_scores


@dataclass
class AnswerEvalReport:
    ragas: RagasScores | None = None
    llm_judge_scores: list[JudgeScore] = field(default_factory=list)
    mean_faithfulness: float = 0.0
    mean_answer_relevancy: float = 0.0


def run_answer_eval(
    *,
    llm: LLMProviderProtocol,
    dataset: QADataset,
    answers: list[str],
    contexts: list[list[str]],
) -> AnswerEvalReport:
    """Run both Ragas and LLM-judge evaluation on a dataset."""
    questions = [item.question for item in dataset.items]
    ground_truths = [item.ground_truth for item in dataset.items]

    ragas_scores = compute_ragas_scores(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        llm=llm,
    )

    judge_scores = judge_faithfulness(llm, questions, answers, contexts)

    report = AnswerEvalReport(ragas=ragas_scores, llm_judge_scores=judge_scores)
    if judge_scores:
        report.mean_faithfulness = sum(s.score for s in judge_scores) / len(
            judge_scores
        )
    return report
