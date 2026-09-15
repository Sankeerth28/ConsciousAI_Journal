"""Feedback SQLModel model."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlmodel import Field, Relationship, SQLModel

from app.models.base import utcnow

if TYPE_CHECKING:
    from app.models.journal import JournalEntry


class Feedback(SQLModel, table=True):
    """Canonical model for user feedback on AI responses.

    This table is the single source of truth for all feedback.
    The `feedback` field on `JournalEntry` is maintained strictly for legacy compatibility.
    """

    __tablename__ = "feedbacks"

    id: int | None = Field(default=None, primary_key=True)
    journal_entry_id: int = Field(foreign_key="journal_entries.id", index=True, nullable=False)
    feedback_type: str = Field(
        index=True, nullable=False
    )  # e.g., "Insightful", "Helpful", "Irrelevant"
    comment: str | None = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=utcnow, index=True, nullable=False)

    # Relationship
    journal_entry: Optional["JournalEntry"] = Relationship(back_populates="feedbacks")
