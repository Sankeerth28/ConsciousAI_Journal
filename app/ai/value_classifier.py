"""Core value theme classification service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.ai.providers.mock import DEFAULT_VALUES, MockValueClassifier
from app.ai.schemas import ValueResult, ValueScore

if TYPE_CHECKING:
    from app.ai.interfaces import ValueClassifier

logger = logging.getLogger(__name__)


class ValueClassificationService:
    """Service detecting underlying core values in self-reflections.

    Identifies broad thematic priorities (e.g., honesty, communication, growth).
    """

    def __init__(
        self,
        classifier: ValueClassifier | None = None,
        default_labels: list[str] | None = None,
    ) -> None:
        self._classifier = classifier or MockValueClassifier()
        self._labels = default_labels or DEFAULT_VALUES

    @property
    def candidate_labels(self) -> list[str]:
        return list(self._labels)

    def classify(
        self,
        text: str,
        custom_labels: list[str] | None = None,
    ) -> ValueResult:
        """Classify underlying value theme of text into candidate labels.

        Args:
            text: Journal entry text.
            custom_labels: Optional subset or override of labels.

        Returns:
            ValueResult with top value, score distribution, and confidence.
        """
        if not text or not text.strip():
            logger.debug("Empty text received for value classification; returning default growth.")
            labels = custom_labels or self._labels
            top = "growth" if "growth" in labels else (labels[0] if labels else "values")
            return ValueResult(
                top_value=top,
                values=[ValueScore(label=top, score=1.0)],
                confidence=0.0,
                model_name=f"{self._classifier.model_name}-empty-fallback",
            )

        labels = custom_labels or self._labels
        cleaned_labels = [str(lbl).strip().lower() for lbl in labels if str(lbl).strip()]
        if not cleaned_labels:
            cleaned_labels = DEFAULT_VALUES

        try:
            return self._classifier.classify(text, candidate_labels=cleaned_labels)
        except Exception as exc:
            logger.warning(
                "Value classification failed (%s); using graceful fallback.",
                type(exc).__name__,
            )
            top = cleaned_labels[0]
            return ValueResult(
                top_value=top,
                values=[ValueScore(label=top, score=0.5)],
                confidence=0.0,
                model_name="fallback-error",
            )
