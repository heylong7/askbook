"""HyDENode: generates hypothetical document embeddings for improved dense retrieval."""

from __future__ import annotations

from importlib.resources import files

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    TraceWriterProtocol,
)

_HYDE_PROMPT = files("askbook.prompts").joinpath("hyde.jinja").read_text("utf-8")


class HyDENode(BasePipelineNode):
    """Generates a hypothetical document to improve dense retrieval."""

    def __init__(
        self,
        *,
        llm: LLMProviderProtocol | None = None,
        enabled: bool = False,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="hyde", trace_writer=trace_writer)
        self._llm = llm
        self._enabled = enabled

    def run(self, context: PipelineContext) -> PipelineContext:
        if not self._enabled:
            return context
        if self._llm is None:
            return context

        from jinja2 import Template

        query = context.get("query", "")
        prompt = Template(_HYDE_PROMPT).render(query=query)
        response = self._llm.complete(prompt)
        hyde_doc = response.content.strip()

        if hyde_doc:
            context["hyde_query"] = hyde_doc  # type: ignore[typeddict-unknown-key]

        return context


__all__ = ["HyDENode"]
