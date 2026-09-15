"""Pydantic request and response schemas for user feedback."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    """Payload for submitting feedback on a journal entry's AI response."""

    model_config = ConfigDict(extra="forbid")

    feedback_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Type of feedback (e.g. 'Insightful', 'Helpful', 'Irrelevant').",
    )
    comment: str | None = Field(
        default=None,
        max_length=1_000,
        description="Optional qualitative user commentary.",
    )


class FeedbackRead(BaseModel):
    """Public read model for feedback records."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    journal_entry_id: int
    feedback_type: str
    comment: str | None = None
    created_at: datetime
