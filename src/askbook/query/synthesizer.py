"""AnswerSynthesizerNode: renders jinja prompt + calls LLM to produce Answer."""

from __future__ import annotations

from importlib.resources import files

from jinja2 import Template

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    TraceWriterProtocol,
)
from askbook.core.models import Answer, Citation, RetrievalResult, TokenUsage


def _render_synthesis(query: str, results: list[RetrievalResult]) -> str:
    text = files("askbook.prompts").joinpath("synthesis.jinja").read_text("utf-8")
    rendered: str = Template(text).render(query=query, results=results)
    return rendered


class AnswerSynthesizerNode(BasePipelineNode):
    """Synthesizes an Answer from retrieval results using an LLM."""

    FALLBACK_TEXT = "资料不足，未能基于当前知识库回答该问题。"

    def __init__(
        self,
        *,
        llm: LLMProviderProtocol,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="answer_synthesizer", trace_writer=trace_writer)
        self._llm = llm

    def run(self, context: PipelineContext) -> PipelineContext:
        results: list[RetrievalResult] = context.get("retrieval_results", [])
        trace_id: str = context.get("pipeline_trace_id", "")
        query: str = context.get("rewritten_query") or context["query"]

        if not results:
            answer = Answer(
                text=self.FALLBACK_TEXT,
                citations=[],
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                pipeline_trace_id=trace_id,
            )
            return {**context, "answer": answer}

        prompt = _render_synthesis(query, results)
        resp = self._llm.complete(prompt)
        citations = [
            Citation(
                chunk_id=r.chunk_id,
                source=str(r.metadata.get("source_path", "")),
                score=r.score,
            )
            for r in results
        ]
        answer = Answer(
            text=resp.content,
            citations=citations,
            usage=resp.usage,
            pipeline_trace_id=trace_id,
        )
        return {**context, "answer": answer}


__all__ = ["AnswerSynthesizerNode"]
