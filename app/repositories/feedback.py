"""Feedback repository for canonical feedback access."""

from __future__ import annotations

from sqlmodel import select

from app.models.feedback import Feedback
from app.repositories.base import BaseRepository


class FeedbackRepository(BaseRepository[Feedback]):
    """Repository handling persistence for canonical Feedback records."""

    def create(self, feedback: Feedback) -> Feedback:
        """Persist a new feedback record.

        The Feedback table is the single source of truth for user feedback.
        """
        self.session.add(feedback)
        self.session.commit()
        self.session.refresh(feedback)
        return feedback

    def get_by_id(self, feedback_id: int) -> Feedback | None:
        """Fetch feedback by primary key."""
        return self.session.get(Feedback, feedback_id)

    def list_by_entry(self, journal_entry_id: int) -> list[Feedback]:
        """List all feedback events for a specific journal entry."""
        statement = (
            select(Feedback)
            .where(Feedback.journal_entry_id == journal_entry_id)
            .order_by(Feedback.created_at.desc(), Feedback.id.desc())
        )
        return list(self.session.exec(statement).all())

    def list(
        self,
        feedback_type: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Feedback]:
        """List feedback records with filtering and pagination."""
        statement = select(Feedback)
        if feedback_type:
            statement = statement.where(Feedback.feedback_type == feedback_type)

        statement = statement.order_by(Feedback.created_at.desc(), Feedback.id.desc())
        statement = statement.offset(skip).limit(limit)
        return list(self.session.exec(statement).all())

    def delete(self, feedback_id: int) -> bool:
        """Remove a feedback record."""
        feedback = self.get_by_id(feedback_id)
        if not feedback:
            return False

        self.session.delete(feedback)
        self.session.commit()
        return True
