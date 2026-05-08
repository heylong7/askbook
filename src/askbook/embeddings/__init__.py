"""askbook.embeddings — embedding provider implementations."""

from askbook.embeddings.openai_embedder import OpenAIEmbedder
from askbook.embeddings.dashscope_embedder import DashScopeEmbedder

__all__ = ["OpenAIEmbedder", "DashScopeEmbedder"]
