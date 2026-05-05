"""Vision LLM enrichment node — injects image descriptions into chunks."""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path

from jinja2 import Template

from askbook.core.interfaces import (
    BasePipelineNode,
    LLMProviderProtocol,
    PipelineContext,
    TraceWriterProtocol,
)
from askbook.core.models import Chunk
from askbook.observability.null_trace import NullTraceWriter

logger = logging.getLogger(__name__)

_IMG_PATTERN = re.compile(r"!\[.*?\]\((.*?)\)")


def _image_to_data_uri(image_path: Path) -> str | None:
    """Read image file from disk and return a base64 data URI.

    Returns ``None`` when the file is missing or unreadable (logged as warning).
    """
    try:
        raw = image_path.read_bytes()
    except (FileNotFoundError, OSError):
        logger.warning("Image file not found or unreadable: %s", image_path)
        return None

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


class LLMEnrichmentNode(BasePipelineNode):
    """Scans chunks for markdown image references, describes each image via an
    LLM, and appends the descriptions back into chunk content.

    When no LLM is configured the node acts as a transparent passthrough.
    """

    def __init__(
        self,
        llm: LLMProviderProtocol | None = None,
        trace_writer: TraceWriterProtocol | None = None,
        *,
        vision_concurrency: int = 2,
    ) -> None:
        super().__init__("enrich", trace_writer or NullTraceWriter())
        self._llm = llm
        self._vision_concurrency = vision_concurrency

    # ------------------------------------------------------------------
    # PipelineNode interface
    # ------------------------------------------------------------------

    def run(self, context: PipelineContext) -> PipelineContext:
        chunks = list(context.get("chunks", []))

        if self._llm is None:
            return {**context, "chunks": chunks}

        source_path = context.get("source_path", ".")
        source_dir = Path(source_path)
        if source_dir.is_file():
            source_dir = source_dir.parent

        if self._vision_concurrency > 1:
            enriched = self._enrich_concurrent(chunks, source_dir)
        else:
            enriched = [self._enrich_chunk(c, source_dir) for c in chunks]
        return {**context, "chunks": enriched}

    def _enrich_concurrent(self, chunks: list[Chunk], source_dir: Path) -> list[Chunk]:
        """Enrich chunks in parallel using a thread pool.

        Each chunk is processed in a separate thread to overlap otherwise-
        blocking LLM completions. Results are placed back in the original
        order so downstream nodes see the same sequence.
        """
        enriched: list[Chunk | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=self._vision_concurrency) as executor:
            futures = {
                executor.submit(self._enrich_chunk, c, source_dir): i
                for i, c in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                enriched[idx] = future.result()
        # All slots are guaranteed filled at this point because every
        # future completes (either successfully or by raising).
        return enriched  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Cached at module level to avoid re-reading/re-compiling per image
    _TEMPLATE: Template | None = None

    def _render_prompt(self, data_uri: str) -> str:
        """Parse a ``data:…;base64,…`` URI and render the Jinja2 template."""
        header, b64_data = data_uri.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]

        if LLMEnrichmentNode._TEMPLATE is None:
            text = (
                files("askbook.prompts")
                .joinpath("img_description.jinja")
                .read_text("utf-8")
            )
            LLMEnrichmentNode._TEMPLATE = Template(text)

        result = LLMEnrichmentNode._TEMPLATE.render(
            mime_type=mime_type, base64_data=b64_data
        )
        return str(result)

    def _enrich_chunk(self, chunk: Chunk, source_dir: Path) -> Chunk:
        """Find every ``![alt](path)`` reference in *chunk*, describe each
        image with the LLM, and return a new chunk with descriptions appended.

        Missing images are silently skipped (the helper logs a warning).
        """
        matches = _IMG_PATTERN.findall(chunk.content)
        if not matches:
            return chunk

        descriptions: list[str] = []
        for rel_path in matches:
            img_path = source_dir / rel_path
            data_uri = _image_to_data_uri(img_path)
            if data_uri is None:
                continue
            prompt = self._render_prompt(data_uri)
            # self._llm is guaranteed non-None by caller
            response = self._llm.complete(prompt)  # type: ignore[union-attr]
            descriptions.append(f"- **{rel_path}**: {response.content.strip()}")

        if not descriptions:
            return chunk

        new_content = (
            chunk.content + "\n\n## Image Descriptions\n" + "\n".join(descriptions)
        )
        return chunk.model_copy(update={"content": new_content})


__all__ = ["LLMEnrichmentNode", "_image_to_data_uri"]
