"""JournalEntry SQLModel model."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import field_validator
from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

from app.models.base import utcnow

if TYPE_CHECKING:
    from app.models.feedback import Feedback
    from app.models.memory import Memory


def _normalize_string_list(v: Any) -> list[str]:
    """Normalize a list of strings: strip whitespace, remove empty items, deduplicate."""
    if v is None:
        return []
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        return list(dict.fromkeys(parts))
    if not isinstance(v, (list, tuple, set)):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in v:
        if item is None:
            continue
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


class JournalEntry(SQLModel, table=True):
    """Core journal entry model representing a user's reflective journal record."""

    __tablename__ = "journal_entries"

    id: int | None = Field(default=None, primary_key=True)
    text: str = Field(nullable=False)
    mood_score: float | None = Field(default=None, nullable=True)

    # Aggregation & quick-filter fields
    top_emotion: str | None = Field(default=None, index=True, nullable=True)
    top_value: str | None = Field(default=None, index=True, nullable=True)

    # Rich list attributes stored portably as JSON
    detected_emotions: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, default=list),
    )
    detected_values: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, default=list),
    )
    tags: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, default=list),
    )

    # Generated AI reflection
    ai_response: str | None = Field(default=None, nullable=True)

    # LEGACY COMPATIBILITY ONLY: The canonical source of truth for new feedback is the Feedback table.
    feedback: str | None = Field(default=None, nullable=True)

    # Fingerprint for deduplicating legacy imports (SHA-256 hash of normalized text + timestamp)
    legacy_source_hash: str | None = Field(
        default=None,
        unique=True,
        index=True,
        nullable=True,
    )

    # Privacy, Multi-Tenancy & Soft-deletion flags
    user_id: str | None = Field(default=None, index=True, nullable=True)
    is_private: bool = Field(default=False, index=True, nullable=False)
    is_deleted: bool = Field(default=False, index=True, nullable=False)
    deleted_at: datetime | None = Field(default=None, nullable=True)

    # Timestamps
    created_at: datetime = Field(default_factory=utcnow, index=True, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)

    # Relationships
    feedbacks: list["Feedback"] = Relationship(
        back_populates="journal_entry",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    memories: list["Memory"] = Relationship(
        back_populates="journal_entry",
    )

    def __init__(self, **data: Any):
        if "detected_emotions" in data:
            data["detected_emotions"] = _normalize_string_list(data.get("detected_emotions"))
        else:
            data["detected_emotions"] = []
        if "detected_values" in data:
            data["detected_values"] = _normalize_string_list(data.get("detected_values"))
        else:
            data["detected_values"] = []
        if "tags" in data:
            data["tags"] = _normalize_string_list(data.get("tags"))
        else:
            data["tags"] = []
        super().__init__(**data)

    @field_validator("detected_emotions", "detected_values", "tags", mode="before")
    @classmethod
    def validate_string_lists(cls, v: Any) -> list[str]:
        return _normalize_string_list(v)
