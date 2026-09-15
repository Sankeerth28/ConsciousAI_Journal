"""Emotion classification service for reflective journaling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ai.providers.mock import DEFAULT_EMOTIONS, MockEmotionClassifier
from app.ai.schemas import EmotionResult, EmotionScore

if TYPE_CHECKING:
    from app.ai.interfaces import EmotionClassifier

logger = logging.getLogger(__name__)


class EmotionClassificationService:
    """Service providing emotion classification for journal entries.

    NOTE: This service identifies emotional expression in written reflections.
    It does NOT provide clinical mental health diagnoses or psychological evaluations.
    """

    def __init__(
        self,
        classifier: EmotionClassifier | None = None,
        default_labels: list[str] | None = None,
    ) -> None:
        self._classifier = classifier or MockEmotionClassifier()
        self._labels = default_labels or DEFAULT_EMOTIONS

    @property
    def candidate_labels(self) -> list[str]:
        return list(self._labels)

    def classify(
        self,
        text: str,
        custom_labels: list[str] | None = None,
    ) -> EmotionResult:
        """Classify the emotional tone of text into candidate labels.

        Args:
            text: Journal entry text.
            custom_labels: Optional subset or override of labels.

        Returns:
            EmotionResult with top emotion, score distribution, and confidence.
        """
        if not text or not text.strip():
            logger.debug("Empty text received for emotion classification; returning default calm.")
            labels = custom_labels or self._labels
            top = "calm" if "calm" in labels else (labels[0] if labels else "neutral")
            return EmotionResult(
                top_emotion=top,
                emotions=[EmotionScore(label=top, score=1.0)],
                confidence=0.0,
                model_name=f"{self._classifier.model_name}-empty-fallback",
            )

        labels = custom_labels or self._labels
        # Clean labels
        cleaned_labels = [str(lbl).strip().lower() for lbl in labels if str(lbl).strip()]
        if not cleaned_labels:
            cleaned_labels = DEFAULT_EMOTIONS

        try:
            return self._classifier.classify(text, candidate_labels=cleaned_labels)
        except Exception as exc:
            logger.warning(
                "Emotion classification failed (%s); using graceful fallback.",
                type(exc).__name__,
            )
            top = cleaned_labels[0]
            return EmotionResult(
                top_emotion=top,
                emotions=[EmotionScore(label=top, score=0.5)],
                confidence=0.0,
                model_name="fallback-error",
            )
