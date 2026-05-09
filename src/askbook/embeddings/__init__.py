"""askbook.embeddings — embedding provider implementations."""

from askbook.embeddings.dashscope_embedder import DashScopeEmbedder
from askbook.embeddings.openai_embedder import OpenAIEmbedder

__all__ = ["OpenAIEmbedder", "DashScopeEmbedder"]
