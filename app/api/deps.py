"""FastAPI dependencies for database sessions, repositories, and AI services."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from app.ai.pipeline import JournalReflectionPipeline
from app.core.config import get_settings
from app.core.security import AuthenticationError, decode_access_token
from app.database.session import get_engine
from app.models.user import User
from app.repositories.feedback import FeedbackRepository
from app.repositories.journal import JournalRepository
from app.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy import Engine

oauth2_bearer = HTTPBearer(auto_error=False)


def _get_engine() -> Engine:
    """Database engine configured from application settings."""
    return get_engine()


def get_db() -> Generator[Session, None, None]:
    """Provide an isolated database session per request."""
    engine = _get_engine()
    with Session(engine) as session:
        yield session


def get_user_repo(
    session: Session = Depends(get_db),
) -> UserRepository:
    """Dependency providing a UserRepository instance."""
    return UserRepository(session)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(oauth2_bearer),
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-ID",
        description="Development-only tenant identifier (ignored in production mode).",
    ),
    user_repo: UserRepository = Depends(get_user_repo),
) -> User:
    """Resolve the authenticated principal from credentials.

    Enforces:
    1. If a Bearer token is provided, it is cryptographically verified.
       Invalid or expired tokens strictly result in HTTP 401 Unauthorized.
    2. In production mode (APP_ENV=production), missing Bearer tokens strictly
       result in HTTP 401 Unauthorized; X-User-ID spoofing is rejected.
    3. In development/test mode without a Bearer token, falls back to X-User-ID
       (defaulting to 'default_user') for local developer ergonomics and test isolation.
    """
    settings = get_settings()

    # Case 1: Bearer token is present -> strictly authenticate via JWT
    if credentials is not None and credentials.credentials:
        token = credentials.credentials.strip()
        try:
            payload = decode_access_token(token)
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        user_id = payload.get("sub")
        user = user_repo.get_by_id(str(user_id))
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user

    # Case 2: No Bearer token provided in production/staging mode -> reject
    if settings.is_production_like:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Case 3: Development / Test fallback mode
    dev_user_id = x_user_id.strip() if x_user_id and x_user_id.strip() else "default_user"
    user = user_repo.get_by_id(dev_user_id)
    if not user:
        user = User(
            id=dev_user_id,
            email=f"{dev_user_id}@local.dev",
            hashed_password="dev_unusable_password_hash",
            is_active=True,
        )
    return user


def get_current_user_id(
    current_user: User = Depends(get_current_user),
) -> str:
    """Extract the current user identity from the authenticated principal.

    Derives identity exclusively from get_current_user, preserving repository
    scoping and API contract compatibility for existing endpoints.
    """
    return current_user.id


def get_journal_repo(
    session: Session = Depends(get_db),
) -> JournalRepository:
    """Dependency providing a JournalRepository instance."""
    return JournalRepository(session)


def get_feedback_repo(
    session: Session = Depends(get_db),
) -> FeedbackRepository:
    """Dependency providing a FeedbackRepository instance."""
    return FeedbackRepository(session)


@lru_cache(maxsize=1)
def _get_shared_pipeline() -> JournalReflectionPipeline:
    """Cached reflection pipeline instance configured with default services."""
    return JournalReflectionPipeline()


def get_pipeline() -> JournalReflectionPipeline:
    """Dependency providing the reflection pipeline."""
    return _get_shared_pipeline()
