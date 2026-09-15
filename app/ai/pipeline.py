"""Sequential reflection pipeline enforcing strict safety order and privacy guarantees."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ai.emotion_classifier import EmotionClassificationService
from app.ai.reflection_engine import STATIC_FALLBACKS, ReflectionEngine
from app.ai.safety import SafetyInterceptor
from app.ai.schemas import (
    PersonaEnum,
    ReflectionPipelineResult,
    ReflectionResult,
    SafetyCheckResult,
)
from app.ai.value_classifier import ValueClassificationService

if TYPE_CHECKING:
    from sqlmodel import Session

    from app.ai.schemas import EmotionResult, MemoryRetrievalResult, ValueResult
    from app.services.memory_service import MemoryRetrievalService

logger = logging.getLogger(__name__)

MAX_JOURNAL_LENGTH = 10_000


class JournalReflectionPipeline:
    """Orchestrates the sequential safety, classification, and reflection pipeline.

    Strict Order of Execution:
    1. Journal Text Input Validation
    2. Input Safety Interception (Crisis & Medical Boundaries)
       -> If UNSAFE: HALT pipeline immediately.
          Do NOT classify emotions/values.
          Do NOT embed or retrieve memories.
          Do NOT send text to LLM.
          Do NOT log user text.
    3. Emotion & Value Classification (Run ONLY on safe input)
    4. Approved Memory Retrieval (Run ONLY on safe input; database revalidated)
    5. Persona-driven Reflection Generation
    6. Output Safety Validation (Validate LLM output against clinical/manipulative policies)
       -> If output is UNSAFE: discard LLM text, substitute safe static fallback.
    7. Final Response Packaging
    """

    def __init__(
        self,
        safety_interceptor: SafetyInterceptor | None = None,
        emotion_service: EmotionClassificationService | None = None,
        value_service: ValueClassificationService | None = None,
        reflection_engine: ReflectionEngine | None = None,
        memory_service: MemoryRetrievalService | None = None,
    ) -> None:
        self._safety = safety_interceptor or SafetyInterceptor()
        self._emotion_service = emotion_service or EmotionClassificationService()
        self._value_service = value_service or ValueClassificationService()
        self._reflection_engine = reflection_engine or ReflectionEngine(
            safety_interceptor=self._safety
        )
        self._memory_service = memory_service

    def process_journal_entry(
        self,
        text: str,
        persona: str | PersonaEnum = PersonaEnum.SUPPORTIVE,
        region: str = "GLOBAL",
        relevant_context: str | None = None,
        session: Session | None = None,
        user_id: str | None = None,
    ) -> ReflectionPipelineResult:
        """Execute the journal reflection pipeline adhering to strict safety order.

        Args:
            text: Raw user journal entry.
            persona: Companion persona style.
            region: User locale/region for crisis resources (e.g., 'US', 'CA', 'UK', 'GLOBAL').
            relevant_context: Approved past memory context (optional override/supplement).
            session: Optional database session for memory retrieval revalidation.
            user_id: Optional user identifier for memory isolation.

        Returns:
            ReflectionPipelineResult with full reflection, classification, and safety metadata.
        """
        resolved_persona = PersonaEnum.from_str(persona)

        # -------------------------------------------------------------
        # 1. INPUT VALIDATION
        # -------------------------------------------------------------
        if not isinstance(text, str):
            logger.warning("Invalid input type rejected by reflection pipeline.")
            safe_check = SafetyCheckResult(is_safe=False, reason="invalid_input_type")
            return ReflectionPipelineResult(
                reflection=ReflectionResult(
                    response=STATIC_FALLBACKS["default"],
                    persona=resolved_persona,
                    model_name="validation_fallback",
                    safety_flag=True,
                    fallback_used=True,
                ),
                input_safety=safe_check,
            )

        stripped_text = text.strip()
        if not stripped_text:
            logger.debug("Empty journal entry received; returning default fallback.")
            safe_check = SafetyCheckResult(is_safe=True, region=region)
            return ReflectionPipelineResult(
                reflection=ReflectionResult(
                    response=STATIC_FALLBACKS["default"],
                    persona=resolved_persona,
                    model_name="static_fallback",
                    safety_flag=False,
                    fallback_used=True,
                ),
                input_safety=safe_check,
            )

        if len(stripped_text) > MAX_JOURNAL_LENGTH:
            stripped_text = stripped_text[:MAX_JOURNAL_LENGTH]
            logger.info("Journal text truncated to %d characters.", MAX_JOURNAL_LENGTH)

        # -------------------------------------------------------------
        # 2. INPUT SAFETY INTERCEPTION
        # -------------------------------------------------------------
        input_safety = self._safety.check_input(stripped_text, region=region)
        if not input_safety.is_safe:
            # CRITICAL PRIVACY & SAFETY: Never log raw journal text.
            logger.info(
                "Safety boundary triggered: reason=%s, region=%s. Pipeline halted.",
                input_safety.reason,
                input_safety.region,
            )
            # HALT IMMEDIATELY: Do NOT classify emotion/value, do NOT embed, do NOT invoke LLM.
            crisis_response = input_safety.crisis_response or STATIC_FALLBACKS["default"]
            return ReflectionPipelineResult(
                reflection=ReflectionResult(
                    response=crisis_response,
                    persona=resolved_persona,
                    model_name="safety_layer",
                    safety_flag=True,
                    fallback_used=False,
                ),
                emotion=None,
                value=None,
                input_safety=input_safety,
                output_safety=None,
            )

        # -------------------------------------------------------------
        # 3. EMOTION & VALUE CLASSIFICATION (Run ONLY on safe input)
        # -------------------------------------------------------------
        emotion_result: EmotionResult = self._emotion_service.classify(stripped_text)
        value_result: ValueResult = self._value_service.classify(stripped_text)

        # -------------------------------------------------------------
        # 4. APPROVED MEMORY RETRIEVAL (Run ONLY after safety passes)
        # -------------------------------------------------------------
        memory_retrieval_result: MemoryRetrievalResult | None = None
        effective_context = relevant_context

        if self._memory_service is not None and session is not None:
            try:
                memory_retrieval_result = self._memory_service.retrieve_context(
                    query=stripped_text,
                    session=session,
                    user_id=user_id,
                )
                if memory_retrieval_result and memory_retrieval_result.context_string:
                    if effective_context:
                        effective_context = (
                            f"{effective_context}\n\n{memory_retrieval_result.context_string}"
                        )
                    else:
                        effective_context = memory_retrieval_result.context_string
            except Exception as exc:
                logger.warning(
                    "Memory retrieval failed gracefully (%s); proceeding with reflection.",
                    type(exc).__name__,
                )

        # -------------------------------------------------------------
        # 5. REFLECTION GENERATION
        # -------------------------------------------------------------
        reflection_result = self._reflection_engine.generate_reflection(
            entry_text=stripped_text,
            detected_emotion=emotion_result.top_emotion,
            detected_value=value_result.top_value,
            persona=resolved_persona,
            relevant_context=effective_context,
            region=region,
            emotion_low_confidence=emotion_result.is_low_confidence,
            value_low_confidence=value_result.is_low_confidence,
        )

        # -------------------------------------------------------------
        # 6. OUTPUT SAFETY VALIDATION
        # -------------------------------------------------------------
        output_safety = self._safety.check_output(reflection_result.response)
        if not output_safety.is_safe:
            logger.warning(
                "Output safety check failed: violation_type=%s. Serving safe fallback.",
                output_safety.violation_type,
            )
            # Substitute with safe emotion-mapped static fallback
            emotion_key = (emotion_result.top_emotion or "").lower()
            safe_response = STATIC_FALLBACKS.get(emotion_key, STATIC_FALLBACKS["default"])
            reflection_result = ReflectionResult(
                response=safe_response,
                persona=resolved_persona,
                model_name="safety_output_fallback",
                safety_flag=True,
                fallback_used=True,
            )

        # -------------------------------------------------------------
        # 7. FINAL RESPONSE PACKAGING
        # -------------------------------------------------------------
        return ReflectionPipelineResult(
            reflection=reflection_result,
            emotion=emotion_result,
            value=value_result,
            input_safety=input_safety,
            output_safety=output_safety,
            memory_context=memory_retrieval_result,
        )
