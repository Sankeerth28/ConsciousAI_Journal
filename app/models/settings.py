"""UserSettings SQLModel model."""

from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.base import utcnow


class UserSettings(SQLModel, table=True):
    """User preferences and configuration settings.

    Designed for single-user local-first operation, ready to be linked to a
    user authentication ID in future milestones.
    """

    __tablename__ = "user_settings"

    id: int | None = Field(default=1, primary_key=True)
    persona: str = Field(
        default="Supportive", nullable=False
    )  # Supportive, Therapist-like, Coach, Neutral
    memory_enabled: bool = Field(default=True, nullable=False)
    analytics_enabled: bool = Field(default=True, nullable=False)
    preferred_response_length: str = Field(default="medium", nullable=False)  # short, medium, long
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)
