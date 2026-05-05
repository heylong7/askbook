"""LLM-judge scoring with semaphore-based parallelism (Ch 30.iv)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from askbook.core.interfaces import LLMProviderProtocol

JUDGE_FAITHFULNESS_PROMPT = """\
Rate the faithfulness of this answer to the provided context on a scale of 1-5.

Context: {context}
Question: {question}
Answer: {answer}

Score (1-5):"""

JUDGE_RELEVANCY_PROMPT = """\
Rate how relevant this answer is to the question on a scale of 1-5.

Question: {question}
Answer: {answer}

Score (1-5):"""


@dataclass
class JudgeScore:
    score: float
    reasoning: str = ""


async def _judge_single(
    llm: LLMProviderProtocol,
    prompt: str,
    sem: asyncio.Semaphore,
) -> str:
    async with sem:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: llm.complete(prompt).content)


async def _batch_judge(
    llm: LLMProviderProtocol,
    prompts: list[str],
    max_concurrency: int = 4,
) -> list[str]:
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [_judge_single(llm, p, sem) for p in prompts]
    return await asyncio.gather(*tasks)


def _parse_judge_response(text: str) -> JudgeScore:
    for ch in text.strip():
        if ch.isdigit():
            score = int(ch) / 5.0
            return JudgeScore(score=min(score, 1.0), reasoning=text.strip())
    return JudgeScore(score=0.0, reasoning=text.strip())


def judge_faithfulness(
    llm: LLMProviderProtocol,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    max_concurrency: int = 4,
) -> list[JudgeScore]:
    """Judge faithfulness per QA pair using asyncio.Semaphore for parallelism."""
    if not questions:
        return []
    prompts = [
        JUDGE_FAITHFULNESS_PROMPT.format(
            context="\n".join(ctx) if ctx else "(no context)",
            question=q,
            answer=a,
        )
        for q, a, ctx in zip(questions, answers, contexts, strict=True)
    ]
    raw_scores = asyncio.run(_batch_judge(llm, prompts, max_concurrency))
    return [_parse_judge_response(r) for r in raw_scores]
