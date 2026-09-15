"""Abstract provider interfaces and protocols for AI services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.ai.schemas import (
        EmbeddingResult,
        EmotionResult,
        ValueResult,
        VectorRecord,
        VectorSearchResult,
    )


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for text generation LLM providers."""

    @property
    def model_name(self) -> str:
        """Name or identifier of the underlying model."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Generate text completion given a prompt.

        Args:
            prompt: Structured text prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text string.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for vector embedding providers."""

    @property
    def model_name(self) -> str:
        """Name or identifier of the underlying embedding model."""
        ...

    @property
    def dimension(self) -> int:
        """Dimension length of the generated embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Generate dense vector embeddings for input texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            EmbeddingResult containing list of vector floats.
        """
        ...


@runtime_checkable
class EmotionClassifier(Protocol):
    """Protocol for emotion classification providers."""

    @property
    def model_name(self) -> str:
        """Name of the classifier model."""
        ...

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> EmotionResult:
        """Classify emotional tone of journal text.

        Args:
            text: Journal text to classify.
            candidate_labels: Optional custom candidate emotion labels.

        Returns:
            EmotionResult with top emotion and full score distribution.
        """
        ...


@runtime_checkable
class ValueClassifier(Protocol):
    """Protocol for core value theme classification providers."""

    @property
    def model_name(self) -> str:
        """Name of the value classifier model."""
        ...

    def classify(
        self,
        text: str,
        candidate_labels: list[str] | None = None,
    ) -> ValueResult:
        """Classify core value theme in journal text.

        Args:
            text: Journal text to classify.
            candidate_labels: Optional custom candidate value labels.

        Returns:
            ValueResult with top value and full score distribution.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage and semantic search providers."""

    @property
    def dimension(self) -> int:
        """Declared vector dimension for this store."""
        ...

    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update vector records in the index."""
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search the index for nearest vectors using normalized cosine similarity.

        Returns:
            List of VectorSearchResult ordered by score descending. Score in [0.0, 1.0].
        """
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove vector records by their IDs."""
        ...

    def clear(self) -> None:
        """Clear all records from the index."""
        ...

    def count(self) -> int:
        """Return the number of vectors stored in the index."""
        ...
