"""Base models and UTC datetime utilities."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import SQLModel


def utcnow() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class TimestampMixin(SQLModel):
    """Mixin providing created_at and updated_at UTC timestamps."""

    pass
