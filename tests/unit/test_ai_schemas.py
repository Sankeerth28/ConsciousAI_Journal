"""Unit tests for AI Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.schemas import (
    EmbeddingResult,
    EmotionResult,
    EmotionScore,
    OutputSafetyCheckResult,
    PersonaEnum,
    ReflectionResult,
    SafetyCheckResult,
    ValueResult,
    ValueScore,
)


class TestAISchemas:
    """Tests for validating structured AI output schemas."""

    def test_persona_enum_and_normalization(self):
        assert PersonaEnum.SUPPORTIVE == "Supportive"
        assert PersonaEnum.COACH == "Coach"
        assert PersonaEnum.THERAPIST == "Therapist-Style Reflection"
        assert PersonaEnum.NEUTRAL == "Neutral"

        assert PersonaEnum.from_str("Supportive") == PersonaEnum.SUPPORTIVE
        assert PersonaEnum.from_str("coach") == PersonaEnum.COACH
        assert PersonaEnum.from_str("Therapist-Style Reflection") == PersonaEnum.THERAPIST
        assert PersonaEnum.from_str("therapist") == PersonaEnum.THERAPIST
        assert PersonaEnum.from_str("NEUTRAL") == PersonaEnum.NEUTRAL
        assert PersonaEnum.from_str("unknown") == PersonaEnum.SUPPORTIVE
        assert PersonaEnum.from_str("") == PersonaEnum.SUPPORTIVE

    def test_emotion_score_and_result(self):
        score = EmotionScore(label="calm", score=0.85)
        assert score.label == "calm"
        assert score.score == 0.85

        result = EmotionResult(
            top_emotion="calm",
            emotions=[score, EmotionScore(label="happy", score=0.15)],
            confidence=0.85,
            model_name="test-model",
        )
        assert result.top_emotion == "calm"
        assert len(result.emotions) == 2
        assert result.confidence == 0.85
        assert result.model_name == "test-model"
        assert result.is_low_confidence is False

    def test_emotion_score_bounds_validation(self):
        with pytest.raises(ValidationError):
            EmotionScore(label="calm", score=1.5)  # > 1.0

        with pytest.raises(ValidationError):
            EmotionScore(label="calm", score=-0.1)  # < 0.0

    def test_low_confidence_flagged(self):
        result = EmotionResult(
            top_emotion="neutral",
            emotions=[EmotionScore(label="neutral", score=0.25)],
            confidence=0.25,
            model_name="test-model",
        )
        assert result.is_low_confidence is True

        val_result = ValueResult(
            top_value="peace",
            values=[ValueScore(label="peace", score=0.20)],
            confidence=0.20,
            model_name="test-model",
        )
        assert val_result.is_low_confidence is True

    def test_value_score_and_result(self):
        score = ValueScore(label="growth", score=0.9)
        result = ValueResult(
            top_value="growth",
            values=[score],
            confidence=0.9,
            model_name="test-val-model",
        )
        assert result.top_value == "growth"
        assert result.values[0].score == 0.9
        assert result.is_low_confidence is False

    def test_emotion_confidence_bounds_validation(self):
        with pytest.raises(ValidationError):
            EmotionResult(
                top_emotion="calm",
                confidence=1.2,  # > 1.0
                model_name="test-model",
            )
        with pytest.raises(ValidationError):
            EmotionResult(
                top_emotion="calm",
                confidence=-0.1,  # < 0.0
                model_name="test-model",
            )

    def test_value_confidence_bounds_validation(self):
        with pytest.raises(ValidationError):
            ValueResult(
                top_value="growth",
                confidence=1.5,  # > 1.0
                model_name="test-model",
            )
        with pytest.raises(ValidationError):
            ValueResult(
                top_value="growth",
                confidence=-0.5,  # < 0.0
                model_name="test-model",
            )

    def test_empty_results_explicitly_handled(self):
        # Empty vectors allowed when embedding empty input
        res = EmbeddingResult(
            vectors=[],
            dimension=384,
            model_name="test-model",
        )
        assert res.vectors == []
        assert res.dimension == 384

        # Empty emotion scores list allowed
        emo_res = EmotionResult(
            top_emotion="calm",
            emotions=[],
            confidence=0.0,
            model_name="test-model",
        )
        assert emo_res.emotions == []
        assert emo_res.is_low_confidence is True

        # Empty value scores list allowed
        val_res = ValueResult(
            top_value="growth",
            values=[],
            confidence=0.0,
            model_name="test-model",
        )
        assert val_res.values == []
        assert val_res.is_low_confidence is True

    def test_embedding_result_valid(self):
        res = EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dimension=3,
            model_name="test-embed-model",
        )
        assert len(res.vectors) == 2
        assert res.dimension == 3

    def test_embedding_result_rejects_ragged_vectors(self):
        with pytest.raises(ValidationError, match="Ragged embedding"):
            EmbeddingResult(
                vectors=[[0.1, 0.2, 0.3], [0.4, 0.5]],  # Length mismatch
                dimension=3,
                model_name="test-embed-model",
            )

    def test_embedding_result_rejects_invalid_dimension(self):
        with pytest.raises(ValidationError):
            EmbeddingResult(
                vectors=[[]],
                dimension=0,  # Must be gt=0
                model_name="test-embed-model",
            )

    def test_reflection_result(self):
        res = ReflectionResult(
            response="I hear you. What would help?",
            persona=PersonaEnum.SUPPORTIVE,
            model_name="test-llm",
            safety_flag=False,
            fallback_used=False,
        )
        assert res.response == "I hear you. What would help?"
        assert res.persona == PersonaEnum.SUPPORTIVE
        assert res.safety_flag is False

    def test_safety_check_results(self):
        safe = SafetyCheckResult(is_safe=True)
        assert safe.is_safe is True
        assert safe.reason is None

        crisis = SafetyCheckResult(
            is_safe=False,
            reason="crisis_detected",
            crisis_response="Please call 988",
        )
        assert crisis.is_safe is False
        assert crisis.reason == "crisis_detected"
        assert crisis.crisis_response == "Please call 988"

        output_safe = OutputSafetyCheckResult(is_safe=True)
        assert output_safe.is_safe is True
