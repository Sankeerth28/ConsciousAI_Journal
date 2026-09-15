"""Memory SQLModel model."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel

from app.models.base import utcnow

if TYPE_CHECKING:
    from app.models.journal import JournalEntry


class Memory(SQLModel, table=True):
    """Extracted memory or key insight from journal reflections.

    Memories are distinct from raw journal entries. New memories default to
    `is_approved=False` to ensure user control and consent.
    """

    __tablename__ = "memories"

    id: int | None = Field(default=None, primary_key=True)
    content: str = Field(nullable=False)
    source_entry_id: int | None = Field(
        default=None,
        foreign_key="journal_entries.id",
        index=True,
        nullable=True,
    )
    memory_type: str = Field(
        default="reflection", index=True, nullable=False
    )  # reflection, fact, summary, goal
    importance: float = Field(default=0.5, nullable=False)  # 0.0 to 1.0
    is_approved: bool = Field(
        default=False, index=True, nullable=False
    )  # Defaults to False for privacy/consent
    is_deleted: bool = Field(default=False, index=True, nullable=False)
    deleted_at: datetime | None = Field(default=None, nullable=True)
    user_id: str | None = Field(default=None, index=True, nullable=True)
    created_at: datetime = Field(default_factory=utcnow, index=True, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)

    # Relationships
    journal_entry: Optional["JournalEntry"] = Relationship(back_populates="memories")
    embedding: Optional["MemoryEmbedding"] = Relationship(
        back_populates="memory",
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "uselist": False},
    )


class MemoryEmbedding(SQLModel, table=True):
    """Stores dense vector representation and provenance metadata for an approved memory."""

    __tablename__ = "memory_embeddings"

    id: int | None = Field(default=None, primary_key=True)
    memory_id: int = Field(
        foreign_key="memories.id",
        unique=True,
        index=True,
        nullable=False,
    )
    embedding_json: list[float] = Field(
        sa_column=Column(JSON, nullable=False),
    )
    dimension: int = Field(nullable=False, index=True)
    provider: str = Field(default="mock", nullable=False)
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        nullable=False,
        index=True,
    )
    version: str = Field(default="1.0", nullable=False)
    created_at: datetime = Field(default_factory=utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=utcnow, nullable=False)

    # Relationship
    memory: Optional["Memory"] = Relationship(back_populates="embedding")
