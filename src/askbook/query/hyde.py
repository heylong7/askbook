"""HyDENode: disabled placeholder for hypothetical document embedding.

Phase 1 opt-in feature — not yet enabled.
"""

from __future__ import annotations

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    TraceWriterProtocol,
)


class HyDENode(BasePipelineNode):
    """Generates a hypothetical document to improve dense retrieval (v1.0 opt-in)."""

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
        # v1.0: generate hypothetical document and fill context["hyde_query"]
        return context


__all__ = ["HyDENode"]
