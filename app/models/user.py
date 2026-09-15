"""User SQLModel database model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlmodel import Field, SQLModel

from app.models.base import utcnow


class User(SQLModel, table=True):
    """User account entity for authentication and multi-tenant authorization."""

    __tablename__ = "users"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True,
        nullable=False,
    )
    email: str = Field(
        unique=True,
        index=True,
        nullable=False,
        description="Normalized user email address (unique, lowercased).",
    )
    hashed_password: str = Field(
        nullable=False,
        description="Bcrypt password hash.",
    )
    is_active: bool = Field(
        default=True,
        nullable=False,
        description="Whether this account is currently active.",
    )
    is_superuser: bool = Field(
        default=False,
        nullable=False,
        description="Whether this account has administrative privileges.",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        nullable=False,
        description="Account creation timestamp.",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        nullable=False,
        description="Account last update timestamp.",
    )
