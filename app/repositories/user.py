"""User repository for account persistence and retrieval."""

from __future__ import annotations

import threading
from typing import Any

from sqlmodel import select

from app.models.base import utcnow
from app.models.user import User
from app.repositories.base import BaseRepository

_user_write_lock = threading.Lock()


def normalize_email(email: str) -> str:
    """Normalize email address consistently by trimming and lowercasing."""
    if not email:
        return ""
    return email.strip().lower()


class UserRepository(BaseRepository[User]):
    """Repository handling persistence operations for User accounts."""

    def create(self, user: User) -> User:
        """Persist a new user account with normalized email."""
        user.email = normalize_email(user.email)
        with _user_write_lock:
            self.session.add(user)
            try:
                self.session.commit()
                self.session.refresh(user)
            except Exception:
                self.session.rollback()
                raise
        return user

    def get_by_id(self, user_id: str) -> User | None:
        """Fetch a user account by its primary key ID."""
        if not user_id:
            return None
        return self.session.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        """Fetch a user account by normalized email address."""
        if not email:
            return None
        norm_email = normalize_email(email)
        statement = select(User).where(User.email == norm_email)
        return self.session.exec(statement).first()

    def update(self, user_id: str, **updates: Any) -> User | None:
        """Update mutable fields on a user account."""
        user = self.get_by_id(user_id)
        if not user:
            return None

        for field, value in updates.items():
            if field == "email" and isinstance(value, str):
                value = normalize_email(value)
            if hasattr(user, field):
                setattr(user, field, value)

        user.updated_at = utcnow()
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user
