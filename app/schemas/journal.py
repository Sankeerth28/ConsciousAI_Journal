"""Pydantic request and response schemas for journal entries."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.ai.schemas import (
    OutputSafetyCheckResult,
    ReflectionResult,
    SafetyCheckResult,
)


class JournalExportFormat(str, Enum):
    """Supported formats for journal data export."""

    CSV = "csv"
    JSON = "json"


class JournalEntryCreate(BaseModel):
    """Payload for creating a new journal entry."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw journal reflection text.",
    )
    mood_score: float | None = Field(
        default=None,
        ge=1.0,
        le=10.0,
        description="Optional subjective mood score between 1.0 and 10.0.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional thematic or categorical tags.",
    )
    persona: str = Field(
        default="Supportive",
        description="Companion persona style ('Supportive', 'Coach', 'Therapist-Style Reflection', 'Neutral').",
    )
    region: str = Field(
        default="GLOBAL",
        description="User region for crisis resource routing ('US', 'CA', 'GB', 'AU', 'IN', 'GLOBAL').",
    )
    is_private: bool = Field(
        default=False,
        description="Privacy visibility flag for the entry.",
    )


class JournalEntryUpdate(BaseModel):
    """Payload for updating mutable fields of an existing journal entry."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] | None = Field(
        default=None,
        description="Updated list of tags.",
    )
    mood_score: float | None = Field(
        default=None,
        ge=1.0,
        le=10.0,
        description="Updated subjective mood score between 1.0 and 10.0.",
    )
    is_private: bool | None = Field(
        default=None,
        description="Updated privacy visibility flag.",
    )


class JournalEntryRead(BaseModel):
    """Public read model for a journal entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: str | None = None
    text: str
    mood_score: float | None = None
    top_emotion: str | None = None
    top_value: str | None = None
    detected_emotions: list[str] = Field(default_factory=list)
    detected_values: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    ai_response: str | None = None
    is_private: bool = False
    created_at: datetime
    updated_at: datetime


class JournalEntryCreateResponse(BaseModel):
    """Response returned when a journal entry is submitted.

    For safe input: includes the persisted entry and reflection results.
    For unsafe input: entry is None (zero persistence) and crisis guidance is returned.
    """

    entry: JournalEntryRead | None = Field(
        default=None,
        description="The persisted journal entry record, or None if halted by safety boundaries.",
    )
    reflection: ReflectionResult = Field(
        ...,
        description="Synthesized AI reflection or safe crisis response envelope.",
    )
    input_safety: SafetyCheckResult = Field(
        ...,
        description="Input safety classification and crisis resource routing metadata.",
    )
    output_safety: OutputSafetyCheckResult | None = Field(
        default=None,
        description="Output safety validation results.",
    )


class JournalListResponse(BaseModel):
    """Paginated collection of journal entries."""

    items: list[JournalEntryRead] = Field(default_factory=list)
    total: int = Field(..., ge=0)
    skip: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    has_more: bool = Field(...)
