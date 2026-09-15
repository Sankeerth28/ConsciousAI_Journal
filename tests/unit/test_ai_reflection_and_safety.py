"""Unit tests for ReflectionEngine and SafetyInterceptor."""

from unittest.mock import MagicMock

import pytest

from app.ai.exceptions import InferenceError
from app.ai.reflection_engine import (
    STATIC_FALLBACKS,
    ReflectionEngine,
    _clean_and_format_response,
)
from app.ai.safety import MEDICAL_BOUNDARY_RESPONSE, SafetyInterceptor
from app.ai.schemas import PersonaEnum


class TestSafetyInterceptor:
    """Tests for crisis detection and medical boundary interception."""

    def test_safe_journal_entry_passes(self):
        interceptor = SafetyInterceptor()
        result = interceptor.check(
            "Today was a busy day at work. I felt a bit tired but accomplished."
        )
        assert result.is_safe is True
        assert result.reason is None
        assert result.crisis_response is None

    @pytest.mark.parametrize(
        "crisis_text",
        [
            "I feel so hopeless, I want to kill myself.",
            "I'm thinking about how to end my life tonight.",
            "I just want to die and make the pain stop.",
            "Everyone would be better off dead without me.",
            "I'm struggling with suicidal thoughts today.",
            "I feel an urge to harm myself.",
            "I don't want to live anymore.",
        ],
    )
    def test_crisis_triggers_lifeline_response(self, crisis_text: str):
        interceptor = SafetyInterceptor(default_region="US")
        result = interceptor.check_input(crisis_text)
        assert result.is_safe is False
        assert result.reason == "crisis_detected"
        assert "988" in result.crisis_response
        assert "Emergency Services" in result.crisis_response
        assert "Trusted Person Guidance" in result.crisis_response

    @pytest.mark.parametrize(
        "medical_text",
        [
            "Can you diagnose me based on these symptoms?",
            "Do I have depression or just sadness?",
            "Please prescribe me medication to help me sleep.",
            "What dose of sertraline should I take?",
        ],
    )
    def test_medical_boundary_violation(self, medical_text: str):
        interceptor = SafetyInterceptor()
        result = interceptor.check_input(medical_text)
        assert result.is_safe is False
        assert result.reason == "medical_boundary_violation"
        assert "licensed mental health professional" in result.crisis_response
        assert result.crisis_response == MEDICAL_BOUNDARY_RESPONSE

    def test_empty_or_whitespace_input_is_safe(self):
        interceptor = SafetyInterceptor()
        assert interceptor.check_input("").is_safe is True
        assert interceptor.check_input("   \n\t").is_safe is True

    def test_custom_regex_patterns(self):
        interceptor = SafetyInterceptor(crisis_patterns=[r"\bcustom_red_flag\b"])
        assert interceptor.check_input("Found a custom_red_flag here.").is_safe is False
        assert interceptor.check_input("Just another normal day.").is_safe is True


class TestCleanAndFormatResponse:
    """Tests for response cleaning and structure enforcement."""

    def test_cleans_preamble_and_ensures_question(self):
        raw = "Response: That sounds like an exhausting day. What could help you rest tonight?"
        cleaned = _clean_and_format_response(raw)
        assert cleaned == "That sounds like an exhausting day. What could help you rest tonight?"

    def test_appends_fallback_question_if_missing(self):
        raw = "It takes courage to face challenging transitions"
        cleaned = _clean_and_format_response(raw)
        assert cleaned.startswith("It takes courage to face challenging transitions.")
        assert cleaned.endswith("What feels most important to you as you reflect on this?")

    def test_empty_string_gives_default_fallback(self):
        assert _clean_and_format_response("") == STATIC_FALLBACKS["default"]


class TestReflectionEngine:
    """Tests for ReflectionEngine persona handling, prompt generation, and resilience."""

    def test_default_initialization_with_mock(self):
        engine = ReflectionEngine()
        result = engine.generate_reflection("I completed a challenging project today.")
        assert result.safety_flag is False
        assert result.fallback_used is False
        assert result.persona == PersonaEnum.SUPPORTIVE
        assert len(result.response) > 10

    def test_crisis_input_triggers_safety_without_llm_call(self):
        mock_llm = MagicMock()
        engine = ReflectionEngine(llm=mock_llm)

        result = engine.generate_reflection("I feel like I want to kill myself.", region="US")
        assert result.safety_flag is True
        assert result.fallback_used is False
        assert result.model_name == "safety_layer"
        assert "Emergency Services" in result.response
        mock_llm.generate.assert_not_called()

    def test_medical_input_triggers_boundary_response(self):
        mock_llm = MagicMock()
        engine = ReflectionEngine(llm=mock_llm)

        result = engine.generate_reflection("Can you diagnose me with major depression?")
        assert result.safety_flag is True
        assert result.model_name == "safety_layer"
        assert "licensed mental health professional" in result.response
        mock_llm.generate.assert_not_called()

    def test_empty_input_returns_static_fallback(self):
        mock_llm = MagicMock()
        engine = ReflectionEngine(llm=mock_llm)

        result = engine.generate_reflection("   ")
        assert result.fallback_used is True
        assert result.model_name == "static_fallback"
        assert result.response == STATIC_FALLBACKS["default"]
        mock_llm.generate.assert_not_called()

    @pytest.mark.parametrize(
        "persona_input,expected_persona",
        [
            ("Supportive", PersonaEnum.SUPPORTIVE),
            ("supportive", PersonaEnum.SUPPORTIVE),
            ("Coach", PersonaEnum.COACH),
            ("coach", PersonaEnum.COACH),
            ("Therapist-Style Reflection", PersonaEnum.THERAPIST),
            ("therapist", PersonaEnum.THERAPIST),
            ("Neutral", PersonaEnum.NEUTRAL),
            ("neutral", PersonaEnum.NEUTRAL),
            ("unknown_persona", PersonaEnum.SUPPORTIVE),
        ],
    )
    def test_all_persona_options_supported(self, persona_input: str, expected_persona: PersonaEnum):
        mock_llm = MagicMock()
        mock_llm.model_name = "mock_model"
        mock_llm.generate.return_value = (
            "That is a meaningful reflection. What stood out most to you?"
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.generate_reflection(
            entry_text="I took a walk in the park and enjoyed the fresh air.",
            persona=persona_input,
        )

        assert result.persona == expected_persona
        assert result.fallback_used is False
        mock_llm.generate.assert_called_once()
        prompt_used = mock_llm.generate.call_args[0][0]
        assert "Strict Guidelines:" in prompt_used

    def test_prompt_includes_emotion_value_and_context(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "mock_model"
        mock_llm.generate.return_value = "Validating statement. Reflective question?"

        engine = ReflectionEngine(llm=mock_llm)
        engine.generate_reflection(
            entry_text="Celebrated a milestone with my teammates.",
            detected_emotion="joy",
            detected_value="community",
            persona=PersonaEnum.COACH,
            relevant_context="User values collaboration and team connection.",
        )

        prompt_used = mock_llm.generate.call_args[0][0]
        assert "detected emotion: joy" in prompt_used
        assert "core value: community" in prompt_used
        assert "Relevant past reflections:" in prompt_used
        assert "User values collaboration and team connection." in prompt_used

    def test_fallback_on_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.model_name = "flaky_model"
        mock_llm.generate.side_effect = InferenceError("LLM backend timeout")

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.generate_reflection(
            entry_text="I was furious about how things were handled.",
            detected_emotion="angry",
            persona=PersonaEnum.SUPPORTIVE,
        )

        assert result.fallback_used is True
        assert result.model_name == "static_fallback"
        assert result.response == STATIC_FALLBACKS["angry"]

    def test_fallback_with_unknown_emotion_uses_default(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("Crash")

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.generate_reflection(
            entry_text="Something happened today.",
            detected_emotion="non_existent_emotion",
        )

        assert result.fallback_used is True
        assert result.response == STATIC_FALLBACKS["default"]

    def test_reflection_engine_catches_unsafe_output_directly(self):
        """Even without the pipeline orchestrator, ReflectionEngine enforces output safety."""
        mock_llm = MagicMock()
        mock_llm.model_name = "mock_model"
        mock_llm.generate.return_value = "You have clinical depression. How are you feeling?"

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.generate_reflection(
            entry_text="I felt sad all day.",
            detected_emotion="sad",
        )

        assert result.safety_flag is True
        assert result.fallback_used is True
        assert result.model_name == "safety_output_fallback"
        assert result.response == STATIC_FALLBACKS["sad"]
