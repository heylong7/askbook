"""QueryRewriterNode: passthrough by default, optional LLM rewrite (Phase 2)."""

from __future__ import annotations

from importlib.resources import files

from jinja2 import Template

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    TraceWriterProtocol,
)


def _load_prompt(template_name: str, **kwargs: object) -> str:
    text = files("askbook.prompts").joinpath(template_name).read_text("utf-8")
    rendered: str = Template(text).render(**kwargs)
    return rendered


class QueryRewriterNode(BasePipelineNode):
    """Rewrites query via LLM when enabled; passthrough otherwise."""

    def __init__(
        self,
        *,
        llm: LLMProviderProtocol | None = None,
        enabled: bool = False,
        trace_writer: TraceWriterProtocol,
    ) -> None:
        super().__init__(name="query_rewriter", trace_writer=trace_writer)
        self._llm = llm
        self._enabled = enabled

    def run(self, context: PipelineContext) -> PipelineContext:
        original: str = context.get("query", "")
        if not self._enabled or self._llm is None:
            return {**context, "rewritten_query": original}
        rendered = _load_prompt("query_rewrite.jinja", query=original)
        rewritten = self._llm.complete(rendered).content.strip() or original
        return {**context, "rewritten_query": rewritten}


__all__ = ["QueryRewriterNode"]
