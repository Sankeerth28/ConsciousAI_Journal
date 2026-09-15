"""Base repository with shared database session handling."""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlmodel import Session, SQLModel

ModelType = TypeVar("ModelType", bound=SQLModel)


class BaseRepository(Generic[ModelType]):
    """Base repository providing explicit session access."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with an active database session.

        Args:
            session: Active SQLModel Session.
        """
        self.session = session
