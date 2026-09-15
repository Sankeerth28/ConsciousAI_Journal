"""Unit tests for deterministic Mock AI providers."""

from __future__ import annotations

import math

from app.ai.interfaces import (
    EmbeddingProvider,
    EmotionClassifier,
    LLMProvider,
    ValueClassifier,
)
from app.ai.providers.mock import (
    MockEmbeddingProvider,
    MockEmotionClassifier,
    MockLLMProvider,
    MockValueClassifier,
)


class TestMockProvidersProtocolCompliance:
    """Verify that Mock providers satisfy runtime-checkable protocols."""

    def test_mock_llm_satisfies_protocol(self):
        llm = MockLLMProvider()
        assert isinstance(llm, LLMProvider)
        assert isinstance(llm.model_name, str)

    def test_mock_embedding_satisfies_protocol(self):
        emb = MockEmbeddingProvider()
        assert isinstance(emb, EmbeddingProvider)
        assert emb.dimension == 384

    def test_mock_emotion_satisfies_protocol(self):
        emo = MockEmotionClassifier()
        assert isinstance(emo, EmotionClassifier)

    def test_mock_value_satisfies_protocol(self):
        val = MockValueClassifier()
        assert isinstance(val, ValueClassifier)


class TestMockLLMBehavior:
    """Tests for MockLLMProvider response generation."""

    def test_mock_llm_generates_supportive_response(self):
        llm = MockLLMProvider()
        resp = llm.generate("User says: I felt overwhelmed today.")
        assert isinstance(resp, str)
        assert len(resp) > 10

    def test_mock_llm_adapts_to_persona_in_prompt(self):
        llm = MockLLMProvider()
        coach_resp = llm.generate("Persona: Coach. User entry: Need to finish my project.")
        assert "action" in coach_resp.lower()

        therapist_resp = llm.generate("Persona: Therapist. User entry: Feeling mixed feelings.")
        assert "emotions" in therapist_resp.lower()


class TestMockEmbeddingBehavior:
    """Tests for MockEmbeddingProvider vectors."""

    def test_mock_embeddings_dimensions_and_norm(self):
        emb = MockEmbeddingProvider(dimension=384)
        result = emb.embed(["I had a tranquil morning walk.", "Writing in my journal."])

        assert len(result.vectors) == 2
        assert len(result.vectors[0]) == 384
        assert len(result.vectors[1]) == 384

        # Verify unit vector norm
        norm = math.sqrt(sum(x * x for x in result.vectors[0]))
        assert abs(norm - 1.0) < 1e-4

    def test_mock_embeddings_deterministic(self):
        emb = MockEmbeddingProvider()
        v1 = emb.embed(["Consistent sentence."]).vectors[0]
        v2 = emb.embed(["Consistent sentence."]).vectors[0]
        assert v1 == v2


class TestMockClassifierBehavior:
    """Tests for MockEmotionClassifier and MockValueClassifier."""

    def test_mock_emotion_keyword_detection(self):
        emo = MockEmotionClassifier()
        res_happy = emo.classify("I feel so joyful and excited about this new chapter!")
        assert res_happy.top_emotion == "happy"
        assert res_happy.confidence >= 0.7

        res_sad = emo.classify("I feel down, grieving the loss.")
        assert res_sad.top_emotion == "sad"

        res_calm = emo.classify("Just an ordinary Tuesday.")
        assert res_calm.top_emotion == "calm"

    def test_mock_value_keyword_detection(self):
        val = MockValueClassifier()
        res_honest = val.classify("I need to speak the truth and be completely honest.")
        assert res_honest.top_value == "honesty"

        res_courage = val.classify("I will be brave and take a bold leap.")
        assert res_courage.top_value == "courage"

        res_growth = val.classify("Learning from everyday routines.")
        assert res_growth.top_value == "growth"
