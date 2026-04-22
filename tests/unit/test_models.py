"""Pydantic model contracts + Harness 30.1.1 whitelist enforcement."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from askbook.core import models


def test_token_usage() -> None:
    u = models.TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.total_tokens == 15
    assert u.estimated_cost_usd is None


def test_llm_response() -> None:
    r = models.LLMResponse(
        content="hi",
        usage=models.TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="qwen2.5:7b",
        provider="ollama",
        latency_ms=42.0,
    )
    assert r.content == "hi"


def test_document() -> None:
    d = models.Document(
        doc_id="abc",
        source_path="/tmp/x.pdf",
        content="# title",
        metadata={"pages": 3},
    )
    assert d.doc_id == "abc"


def test_chunk_defaults() -> None:
    c = models.Chunk(chunk_id="h1", doc_id="d1", content="hello")
    assert c.embedding is None
    assert c.metadata == {}


def test_retrieval_result_harness_fields() -> None:
    """Harness 30.1.1: snippet<=200, no full_text fields."""
    r = models.RetrievalResult(
        chunk_id="c1",
        score=0.9,
        snippet="a" * 200,
        metadata={"source": "x.pdf"},
        retrieval_method="bm25",
    )
    assert r.chunk_id == "c1"
    assert len(r.snippet) == 200


def test_retrieval_result_snippet_max_length() -> None:
    with pytest.raises(ValidationError):
        models.RetrievalResult(
            chunk_id="c1",
            score=0.9,
            snippet="x" * 201,
            metadata={},
            retrieval_method="bm25",
        )


@pytest.mark.parametrize("banned", ["raw_text", "full_content", "page_content"])
def test_retrieval_result_rejects_banned_fields(banned: str) -> None:
    """Harness 30.1.1: banned fields must be rejected at construction."""
    payload = {
        "chunk_id": "c1",
        "score": 0.5,
        "snippet": "s",
        "metadata": {},
        "retrieval_method": "bm25",
        banned: "some leaked full text",
    }
    with pytest.raises(ValidationError):
        models.RetrievalResult(**payload)


def test_retrieval_result_field_names_are_canonical() -> None:
    fields = set(models.RetrievalResult.model_fields.keys())
    assert fields == {"chunk_id", "score", "snippet", "metadata", "retrieval_method"}


def test_citation_and_answer() -> None:
    cit = models.Citation(chunk_id="c1", source="x.pdf", score=0.8)
    ans = models.Answer(
        text="the answer",
        citations=[cit],
        usage=models.TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        pipeline_trace_id="trace-1",
    )
    assert ans.citations[0].chunk_id == "c1"


def test_collection_info() -> None:
    info = models.CollectionInfo(name="ns__bge__v1", size=100, embed_model="bge-m3")
    assert info.size == 100


def test_collection_stats() -> None:
    stats = models.CollectionStats(
        collection="ns__bge__v1",
        chunk_count=42,
        doc_count=5,
        last_updated="2026-04-21T10:00:00Z",
    )
    assert stats.chunk_count == 42


def test_qa_pair() -> None:
    qa = models.QAPair(question="q", answer="a", source_ids=["c1", "c2"])
    assert qa.source_ids == ["c1", "c2"]


def test_ingestion_result_defaults() -> None:
    from askbook.core.models import IngestionResult

    r = IngestionResult(
        docs_processed=3,
        chunks_added=10,
        chunks_reused=2,
        chunks_deleted=1,
        collection="kb_demo",
        duration_seconds=1.5,
    )
    assert r.errors == []
    assert r.collection == "kb_demo"


def test_ingestion_result_forbids_extra_fields() -> None:
    from askbook.core.models import IngestionResult

    with pytest.raises(ValidationError):
        IngestionResult(
            docs_processed=1,
            chunks_added=1,
            chunks_reused=0,
            chunks_deleted=0,
            collection="x",
            duration_seconds=0.1,
            unknown_field="boom",  # type: ignore[call-arg]
        )
