"""Tests proving raw journal text, reflection prompts, and credentials never leak into logs."""

from __future__ import annotations

import contextlib
import logging
import sys
from unittest.mock import MagicMock, patch

from app.ai.emotion_classifier import EmotionClassificationService
from app.ai.pipeline import JournalReflectionPipeline
from app.ai.providers.huggingface import HuggingFaceLLMProvider
from app.ai.reflection_engine import ReflectionEngine
from app.ai.schemas import PersonaEnum
from app.ai.value_classifier import ValueClassificationService


class TestPrivacyAndLoggingGuarantees:
    """Audit all logging pathways to ensure strict confidentiality of journal data and secrets."""

    SECRET_JOURNAL_TEXT = "MY_HIGHLY_CONFIDENTIAL_JOURNAL_ENTRY_CONTENT_XYZ_987"
    SECRET_CRISIS_TEXT = "I feel like ending my life secret crisis abc 123"
    SECRET_HF_TOKEN = "hf_SUPER_SECRET_TOKEN_VALUE_DO_NOT_LEAK"

    def test_safe_journal_processing_does_not_log_entry_or_response(self, caplog):
        """Standard pipeline flow must not log the entry text or the generated response."""
        caplog.set_level(logging.DEBUG)

        pipeline = JournalReflectionPipeline()
        result = pipeline.process_journal_entry(
            self.SECRET_JOURNAL_TEXT,
            persona=PersonaEnum.SUPPORTIVE,
        )

        log_records = caplog.text
        assert self.SECRET_JOURNAL_TEXT not in log_records
        assert result.reflection.response not in log_records

    def test_crisis_interception_does_not_log_crisis_entry(self, caplog):
        """When crisis content is detected, the raw text must never appear in log output."""
        caplog.set_level(logging.DEBUG)

        pipeline = JournalReflectionPipeline()
        result = pipeline.process_journal_entry(self.SECRET_CRISIS_TEXT)

        assert result.input_safety.is_safe is False
        log_records = caplog.text
        assert self.SECRET_CRISIS_TEXT not in log_records
        # Verify metadata like reason is logged safely without leaking text
        assert "crisis_detected" in log_records

    def test_llm_exception_does_not_log_prompt_or_journal_text(self, caplog):
        """When the LLM fails during inference, prompt and journal text must not appear in log records."""
        caplog.set_level(logging.DEBUG)

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("Inference backend connection failure")

        engine = ReflectionEngine(llm=mock_llm)
        engine.generate_reflection(
            entry_text=self.SECRET_JOURNAL_TEXT,
            persona=PersonaEnum.SUPPORTIVE,
        )

        log_records = caplog.text
        assert self.SECRET_JOURNAL_TEXT not in log_records
        assert "Strict Guidelines:" not in log_records

    def test_hf_provider_does_not_log_api_token(self, caplog):
        """HuggingFace provider must never log the HF API token on loading or error."""
        caplog.set_level(logging.DEBUG)

        provider = HuggingFaceLLMProvider(
            model_name="nonexistent/model-name-for-test",
            hf_token=self.SECRET_HF_TOKEN,
        )

        mock_transformers = MagicMock()
        mock_transformers.pipeline.side_effect = Exception("Failed to download model weights")
        with (
            patch.dict(sys.modules, {"transformers": mock_transformers}),
            contextlib.suppress(Exception),
        ):
            provider._ensure_pipeline()

        log_records = caplog.text
        assert self.SECRET_HF_TOKEN not in log_records

    def test_classifier_exceptions_do_not_log_entry_text(self, caplog):
        """When emotion or value classifier encounters an error, journal text is not logged."""
        caplog.set_level(logging.DEBUG)

        mock_classifier = MagicMock()
        mock_classifier.classify.side_effect = RuntimeError("Classification service timeout")

        emo_service = EmotionClassificationService(classifier=mock_classifier)
        val_service = ValueClassificationService(classifier=mock_classifier)

        emo_service.classify(self.SECRET_JOURNAL_TEXT)
        val_service.classify(self.SECRET_JOURNAL_TEXT)

        log_records = caplog.text
        assert self.SECRET_JOURNAL_TEXT not in log_records

    def test_output_safety_violation_does_not_log_unsafe_llm_text(self, caplog):
        """When output safety intercepts an unsafe LLM completion, the unsafe text is not logged."""
        caplog.set_level(logging.DEBUG)

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            "I diagnose you with clinical depression. You should stop taking your medication."
        )

        engine = ReflectionEngine(llm=mock_llm)
        result = engine.generate_reflection(
            entry_text=self.SECRET_JOURNAL_TEXT,
            detected_emotion="sad",
        )

        assert result.safety_flag is True
        assert result.fallback_used is True

        log_records = caplog.text
        assert self.SECRET_JOURNAL_TEXT not in log_records
        assert "clinical depression" not in log_records
        assert "stop taking your medication" not in log_records
