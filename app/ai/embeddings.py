"""Embedding service for generating dense vector representations of reflections."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ai.providers.mock import MockEmbeddingProvider
from app.ai.schemas import EmbeddingResult

if TYPE_CHECKING:
    from app.ai.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service generating vector embeddings for semantic search and memory indexing."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider or MockEmbeddingProvider()

    @property
    def dimension(self) -> int:
        """Vector dimension produced by the configured provider."""
        return self._provider.dimension

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a float vector."""
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimension

        result = self._provider.embed([text.strip()])
        return result.vectors[0] if result.vectors else [0.0] * self.dimension

    def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of text strings in batch.

        Args:
            texts: List of strings to embed.

        Returns:
            EmbeddingResult containing list of float vectors.
        """
        if not texts:
            return EmbeddingResult(
                vectors=[],
                dimension=self.dimension,
                model_name=self.model_name,
            )

        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        return self._provider.embed(cleaned_texts)

    def embed(self, input_data: str | list[str]) -> list[float] | list[list[float]]:
        """Convenience method accepting either a string or list of strings."""
        if isinstance(input_data, str):
            return self.embed_text(input_data)
        batch_result = self.embed_batch(input_data)
        return batch_result.vectors
