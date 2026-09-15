"""User settings repository."""

from __future__ import annotations

from typing import Any

from app.models.base import utcnow
from app.models.settings import UserSettings
from app.repositories.base import BaseRepository

IMMUTABLE_FIELDS = {"id", "created_at"}


class UserSettingsRepository(BaseRepository[UserSettings]):
    """Repository handling persistence for user configuration and preferences."""

    def get(self, settings_id: int = 1) -> UserSettings | None:
        """Fetch user settings by ID (defaults to 1 for single-user mode)."""
        return self.session.get(UserSettings, settings_id)

    def get_or_create(self, settings_id: int = 1) -> UserSettings:
        """Fetch existing settings or create default settings if absent."""
        settings = self.get(settings_id)
        if not settings:
            settings = UserSettings(id=settings_id)
            self.session.add(settings)
            self.session.commit()
            self.session.refresh(settings)
        return settings

    def update(self, settings_id: int = 1, **updates: Any) -> UserSettings:
        """Update user preferences with safety checks."""
        settings = self.get_or_create(settings_id)

        for field, value in updates.items():
            if field in IMMUTABLE_FIELDS:
                msg = f"Cannot update immutable field '{field}'"
                raise ValueError(msg)
            if not hasattr(settings, field):
                msg = f"Unknown field '{field}' on UserSettings"
                raise ValueError(msg)
            setattr(settings, field, value)

        settings.updated_at = utcnow()
        self.session.add(settings)
        self.session.commit()
        self.session.refresh(settings)
        return settings
