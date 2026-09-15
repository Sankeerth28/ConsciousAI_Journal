"""Pydantic schemas for structured AI inputs and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PersonaEnum(str, Enum):
    """Supported reflective journal personas."""

    SUPPORTIVE = "Supportive"
    COACH = "Coach"
    THERAPIST = "Therapist-Style Reflection"
    NEUTRAL = "Neutral"

    @classmethod
    def from_str(cls, value: str | PersonaEnum) -> PersonaEnum:
        """Parse string or existing enum safely with fallback to SUPPORTIVE."""
        if isinstance(value, cls):
            return value
        if not value or not str(value).strip():
            return cls.SUPPORTIVE
        normalized = str(value).strip().lower()
        if "therapist" in normalized:
            return cls.THERAPIST
        if "coach" in normalized:
            return cls.COACH
        if "neutral" in normalized:
            return cls.NEUTRAL
        return cls.SUPPORTIVE


class EmotionScore(BaseModel):
    """Score for a specific emotion candidate label."""

    label: str
    score: float = Field(ge=0.0, le=1.0)


class EmotionResult(BaseModel):
    """Structured result from emotion classification."""

    top_emotion: str
    emotions: list[EmotionScore] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    is_low_confidence: bool = False

    @model_validator(mode="after")
    def compute_low_confidence(self) -> EmotionResult:
        """Mark low-confidence classifications (< 0.30) to prevent false certainty."""
        if self.confidence < 0.30:
            self.is_low_confidence = True
        return self


class ValueScore(BaseModel):
    """Score for a specific core value candidate label."""

    label: str
    score: float = Field(ge=0.0, le=1.0)


class ValueResult(BaseModel):
    """Structured result from core value classification."""

    top_value: str
    values: list[ValueScore] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    is_low_confidence: bool = False

    @model_validator(mode="after")
    def compute_low_confidence(self) -> ValueResult:
        """Mark low-confidence classifications (< 0.30) to prevent false certainty."""
        if self.confidence < 0.30:
            self.is_low_confidence = True
        return self


class EmbeddingResult(BaseModel):
    """Structured result from embedding generation."""

    vectors: list[list[float]]
    dimension: int = Field(gt=0)
    model_name: str

    @model_validator(mode="after")
    def validate_embedding_dimensions(self) -> EmbeddingResult:
        """Validate non-ragged embedding vectors matching declared dimension."""
        for idx, vec in enumerate(self.vectors):
            if len(vec) != self.dimension:
                msg = (
                    f"Ragged embedding at index {idx}: vector length {len(vec)} "
                    f"does not match declared dimension {self.dimension}"
                )
                raise ValueError(msg)
        return self


class ReflectionResult(BaseModel):
    """Structured output from the reflection engine."""

    response: str
    persona: PersonaEnum
    model_name: str
    safety_flag: bool = False
    fallback_used: bool = False


class SafetyCheckResult(BaseModel):
    """Result from input safety boundary evaluation."""

    is_safe: bool
    reason: str | None = None
    crisis_response: str | None = None
    region: str | None = None


class OutputSafetyCheckResult(BaseModel):
    """Result from output safety validation."""

    is_safe: bool
    violation_type: str | None = None
    reason: str | None = None


class VectorRecord(BaseModel):
    """A dense vector record with unique identifier and arbitrary metadata."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_vector(self) -> VectorRecord:
        if not self.vector:
            msg = "VectorRecord.vector cannot be empty."
            raise ValueError(msg)
        import math

        for idx, val in enumerate(self.vector):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                msg = f"VectorRecord.vector contains non-finite numeric value at index {idx}."
                raise ValueError(msg)
        return self


class VectorSearchResult(BaseModel):
    """Result of a nearest neighbor vector similarity search."""

    id: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """A database-revalidated memory matched via semantic search."""

    memory_id: int
    content: str
    score: float = Field(ge=0.0, le=1.0)
    memory_type: str = "reflection"
    importance: float = 0.5
    source_entry_id: int | None = None
    created_at: str | None = None


class MemoryRetrievalResult(BaseModel):
    """Structured collection of retrieved memories and formatted context."""

    query_text_length: int
    matches: list[MemorySearchResult] = Field(default_factory=list)
    context_string: str | None = None
    is_empty: bool = True


class ReflectionPipelineResult(BaseModel):
    """Comprehensive output from the end-to-end journal reflection pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reflection: ReflectionResult
    emotion: EmotionResult | None = None
    value: ValueResult | None = None
    input_safety: SafetyCheckResult
    output_safety: OutputSafetyCheckResult | None = None
    memory_context: MemoryRetrievalResult | None = None
