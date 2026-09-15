"""Unit tests for domain AI services (emotions, values, embeddings)."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.ai.embeddings import EmbeddingService
from app.ai.emotion_classifier import EmotionClassificationService
from app.ai.providers.mock import (
    MockEmbeddingProvider,
    MockEmotionClassifier,
    MockValueClassifier,
)
from app.ai.value_classifier import ValueClassificationService


class TestEmotionClassificationService:
    """Tests for EmotionClassificationService."""

    def test_default_labels_and_classification(self):
        service = EmotionClassificationService(MockEmotionClassifier())
        assert len(service.candidate_labels) == 8
        assert "happy" in service.candidate_labels

        res = service.classify("Today was joyous and peaceful.")
        assert res.top_emotion == "happy"
        assert res.confidence > 0.5

    def test_empty_text_returns_default_calm_safely(self):
        service = EmotionClassificationService(MockEmotionClassifier())
        res = service.classify("   ")
        assert res.top_emotion == "calm"
        assert res.confidence == 0.0

    def test_custom_candidate_labels(self):
        service = EmotionClassificationService(MockEmotionClassifier())
        res = service.classify("I feel great", custom_labels=["excited", "tired"])
        assert res.top_emotion in ["excited", "tired"]

    def test_graceful_fallback_on_classifier_exception(self):
        mock_provider = MagicMock()
        mock_provider.classify.side_effect = RuntimeError("Inference crash")
        mock_provider.model_name = "failing-model"

        service = EmotionClassificationService(mock_provider)
        res = service.classify("Some journal text")

        assert res.model_name == "fallback-error"
        assert res.confidence == 0.0


class TestValueClassificationService:
    """Tests for ValueClassificationService."""

    def test_default_labels_and_classification(self):
        service = ValueClassificationService(MockValueClassifier())
        assert len(service.candidate_labels) == 7
        assert "honesty" in service.candidate_labels

        res = service.classify("I need to speak honestly and openly.")
        assert res.top_value == "honesty"
        assert res.confidence > 0.5

    def test_empty_text_returns_default_growth_safely(self):
        service = ValueClassificationService(MockValueClassifier())
        res = service.classify("")
        assert res.top_value == "growth"
        assert res.confidence == 0.0

    def test_graceful_fallback_on_classifier_exception(self):
        mock_provider = MagicMock()
        mock_provider.classify.side_effect = RuntimeError("Inference crash")
        mock_provider.model_name = "failing-model"

        service = ValueClassificationService(mock_provider)
        res = service.classify("Some journal text")

        assert res.model_name == "fallback-error"
        assert res.confidence == 0.0


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_embed_single_text(self):
        service = EmbeddingService(MockEmbeddingProvider(dimension=384))
        vec = service.embed_text("Reflecting on gratitude.")
        assert len(vec) == 384
        assert isinstance(vec[0], float)

    def test_embed_batch(self):
        service = EmbeddingService(MockEmbeddingProvider(dimension=384))
        res = service.embed_batch(["Text 1", "Text 2", "Text 3"])
        assert len(res.vectors) == 3
        assert res.dimension == 384

    def test_embed_empty_text_returns_zero_vector(self):
        service = EmbeddingService(MockEmbeddingProvider(dimension=384))
        vec = service.embed_text("")
        assert len(vec) == 384
        assert all(x == 0.0 for x in vec)

    def test_embed_convenience_dispatcher(self):
        service = EmbeddingService(MockEmbeddingProvider(dimension=384))
        single = service.embed("One string")
        assert isinstance(single, list)
        assert len(single) == 384

        batch = service.embed(["Str 1", "Str 2"])
        assert isinstance(batch, list)
        assert len(batch) == 2
