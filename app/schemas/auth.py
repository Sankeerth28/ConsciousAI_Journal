"""Pydantic schemas for user authentication and authorization."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """Payload for registering a new user account."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(..., description="User's primary email address.")
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Plaintext account password (min 8 chars, max 72 bytes).",
    )


class UserLogin(BaseModel):
    """Payload for authenticating and obtaining an access token."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(..., description="User's registered email address.")
    password: str = Field(..., description="User's plaintext password.")


class TokenResponse(BaseModel):
    """Bearer access token response envelope."""

    access_token: str = Field(..., description="Signed cryptographic JWT access token.")
    token_type: str = Field(default="bearer", description="Token authorization type.")
    expires_in: int = Field(..., description="Token lifespan in seconds.")


class UserRead(BaseModel):
    """Public read model for user identity without sensitive credentials."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    is_active: bool
    created_at: datetime
