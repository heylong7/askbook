"""Sub-config pydantic models loaded by Settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMFallbackItem(BaseModel):
    provider: str
    model: str


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    temperature: float = 0.1
    max_tokens: int = 4096
    fallback_chain: list[LLMFallbackItem] = Field(default_factory=list)
    token_limit_per_call: int = 8000
    quota_per_hour: int = 100


class EmbeddingConfig(BaseModel):
    provider: str = "bge-m3"
    model: str = "BAAI/bge-m3"
    normalize: bool = True
    device: str = "auto"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    path: str = "~/.askbook/chroma"


class IngestionConfig(BaseModel):
    chunk_size: int = 600
    chunk_overlap: int = 80
    embed_concurrency: int = 4
    vision_concurrency: int = 2
    chroma_concurrency: int = 4


class QueryConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
    enable_rewrite: bool = False
    enable_hyde: bool = False
    enable_llm_rerank: bool = False


class MCPConfig(BaseModel):
    transport: str = "stdio"
    system_prompt_max_tokens: int = 500


class ObservabilityConfig(BaseModel):
    trace_dir: str = "~/.askbook/traces"
    flush_interval_seconds: float = 1.0
    pii_redaction: bool = True


__all__ = [
    "LLMFallbackItem",
    "LLMConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "IngestionConfig",
    "QueryConfig",
    "MCPConfig",
    "ObservabilityConfig",
]
