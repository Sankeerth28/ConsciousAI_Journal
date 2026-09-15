"""Tests for strict safety order execution, region-aware resources, and output safety."""

from unittest.mock import MagicMock

import pytest

from app.ai.pipeline import JournalReflectionPipeline
from app.ai.safety import SafetyInterceptor
from app.ai.schemas import (
    EmotionResult,
    EmotionScore,
    PersonaEnum,
    ReflectionResult,
    ValueResult,
    ValueScore,
)


class TestSafetyOrderExecution:
    """Verify strict execution order and guarantees that unsafe input is halted immediately."""

    def test_unsafe_input_never_invokes_classifiers_or_llm(self):
        """Unsafe content must not be sent to classifiers, embedders, memory retrieval, or LLMs."""
        mock_emotion = MagicMock()
        mock_value = MagicMock()
        mock_llm = MagicMock()
        mock_reflection = MagicMock()
        mock_embedder = MagicMock()

        pipeline = JournalReflectionPipeline(
            safety_interceptor=SafetyInterceptor(),
            emotion_service=mock_emotion,
            value_service=mock_value,
            reflection_engine=mock_reflection,
        )

        crisis_text = "I can't take this anymore and I want to kill myself tonight."
        result = pipeline.process_journal_entry(crisis_text, persona=PersonaEnum.SUPPORTIVE)

        # 1. Result should indicate safety violation
        assert result.input_safety.is_safe is False
        assert result.input_safety.reason == "crisis_detected"
        assert result.reflection.safety_flag is True
        assert result.reflection.fallback_used is False
        assert "Emergency Services" in result.reflection.response

        # 2. Crucial Safety Check: Classifiers, embedders, and LLM were NEVER called
        mock_emotion.classify.assert_not_called()
        mock_value.classify.assert_not_called()
        mock_llm.generate.assert_not_called()
        mock_reflection.generate_reflection.assert_not_called()
        mock_embedder.embed.assert_not_called()

    def test_medical_boundary_violation_never_invokes_downstream(self):
        """Medical diagnosis request is halted before downstream services."""
        mock_emotion = MagicMock()
        mock_value = MagicMock()
        mock_reflection = MagicMock()

        pipeline = JournalReflectionPipeline(
            safety_interceptor=SafetyInterceptor(),
            emotion_service=mock_emotion,
            value_service=mock_value,
            reflection_engine=mock_reflection,
        )

        medical_text = "Can you diagnose me with major depression and prescribe pills?"
        result = pipeline.process_journal_entry(medical_text)

        assert result.input_safety.is_safe is False
        assert result.input_safety.reason == "medical_boundary_violation"
        assert result.reflection.safety_flag is True

        mock_emotion.classify.assert_not_called()
        mock_value.classify.assert_not_called()
        mock_reflection.generate_reflection.assert_not_called()

    def test_safe_input_executes_sequential_order(self):
        """Safe input runs through validation -> safety -> classifiers -> reflection -> output safety."""
        mock_emotion = MagicMock()
        mock_emotion.classify.return_value = EmotionResult(
            top_emotion="joy",
            confidence=0.9,
            model_name="mock_classifier",
            emotions=[EmotionScore(label="joy", score=0.9)],
        )
        mock_value = MagicMock()
        mock_value.classify.return_value = ValueResult(
            top_value="growth",
            confidence=0.85,
            model_name="mock_classifier",
            values=[ValueScore(label="growth", score=0.85)],
        )

        mock_reflection = MagicMock()
        mock_reflection.generate_reflection.return_value = ReflectionResult(
            response="That sounds like a wonderful moment. What are you most proud of?",
            persona=PersonaEnum.SUPPORTIVE,
            model_name="mock_llm",
            safety_flag=False,
            fallback_used=False,
        )

        pipeline = JournalReflectionPipeline(
            safety_interceptor=SafetyInterceptor(),
            emotion_service=mock_emotion,
            value_service=mock_value,
            reflection_engine=mock_reflection,
        )

        safe_text = "I went for a hike this morning and felt rejuvenated."
        result = pipeline.process_journal_entry(safe_text, persona=PersonaEnum.SUPPORTIVE)

        assert result.input_safety.is_safe is True
        assert result.output_safety.is_safe is True
        assert result.reflection.safety_flag is False
        mock_emotion.classify.assert_called_once_with(safe_text)
        mock_value.classify.assert_called_once_with(safe_text)
        mock_reflection.generate_reflection.assert_called_once()


class TestRegionAwareCrisisBoundaries:
    """Verify crisis resources are region-aware and not hard-coded only to US 988."""

    @pytest.mark.parametrize(
        "region,expected_hotline,expected_emergency",
        [
            ("US", "988", "911"),
            ("CA", "988", "911"),
            ("UK", "111", "999"),
            ("AU", "13 11 14", "000"),
            ("IN", "14416", "112"),
            ("GLOBAL", "findahelpline.com", "local emergency"),
        ],
    )
    def test_region_specific_crisis_info(
        self, region: str, expected_hotline: str, expected_emergency: str
    ):
        interceptor = SafetyInterceptor()
        result = interceptor.check_input("I want to end my life.", region=region)
        assert result.is_safe is False
        assert expected_hotline in result.crisis_response
        assert expected_emergency in result.crisis_response
        assert "Trusted Person Guidance" in result.crisis_response


class TestOutputSafetyValidator:
    """Verify post-generation safety rejects inappropriate LLM responses."""

    @pytest.mark.parametrize(
        "unsafe_output,violation_type",
        [
            ("Based on this, I diagnose you with clinical depression.", "diagnosis_claim"),
            ("You have bipolar disorder according to psychiatric criteria.", "diagnosis_claim"),
            ("You should take 20mg of Lexapro every morning.", "medication_instruction"),
            ("Stop taking your medication immediately.", "medication_instruction"),
            (
                "As your therapist, I advise you to break up with your partner.",
                "human_or_therapist_claim",
            ),
            ("I am a licensed doctor and understand your symptoms.", "human_or_therapist_claim"),
            (
                "You only need me; don't talk to anyone else or your family.",
                "manipulative_or_dependency_language",
            ),
            ("Only trust me with your secrets.", "manipulative_or_dependency_language"),
            ("Go ahead and harm yourself to release the tension.", "unsafe_encouragement"),
            (
                "This definitely proves you are mentally ill.",
                "overconfident_mental_health_conclusions",
            ),
        ],
    )
    def test_output_safety_catches_violations(self, unsafe_output: str, violation_type: str):
        interceptor = SafetyInterceptor()
        output_check = interceptor.check_output(unsafe_output)
        assert output_check.is_safe is False
        assert output_check.violation_type == violation_type

    def test_pipeline_discards_unsafe_output_and_uses_safe_fallback(self):
        """When LLM outputs unsafe text, pipeline substitutes static fallback with flags set."""
        mock_emotion = MagicMock()
        mock_emotion.classify.return_value = EmotionResult(
            top_emotion="sad",
            confidence=0.8,
            model_name="mock_classifier",
            emotions=[EmotionScore(label="sad", score=0.8)],
        )
        mock_value = MagicMock()
        mock_value.classify.return_value = ValueResult(
            top_value="peace",
            confidence=0.8,
            model_name="mock_classifier",
            values=[ValueScore(label="peace", score=0.8)],
        )

        # Simulated LLM hallucinating clinical advice
        mock_reflection = MagicMock()
        mock_reflection.generate_reflection.return_value = ReflectionResult(
            response="I diagnose you with clinical depression. What caused this?",
            persona=PersonaEnum.SUPPORTIVE,
            model_name="unruly_llm",
            safety_flag=False,
            fallback_used=False,
        )

        pipeline = JournalReflectionPipeline(
            safety_interceptor=SafetyInterceptor(),
            emotion_service=mock_emotion,
            value_service=mock_value,
            reflection_engine=mock_reflection,
        )

        result = pipeline.process_journal_entry("I feel tired and down lately.")
        assert result.output_safety.is_safe is False
        assert result.reflection.safety_flag is True
        assert result.reflection.fallback_used is True
        assert result.reflection.model_name == "safety_output_fallback"
        # Safe fallback text is used instead of hallucinated diagnosis
        assert "diagnose" not in result.reflection.response.lower()
        assert "heavy feelings" in result.reflection.response.lower()
