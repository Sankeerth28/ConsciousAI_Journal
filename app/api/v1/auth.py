"""Authentication REST API endpoints for user registration, login, and identity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_current_user, get_user_repo
from app.core.config import get_settings
from app.core.rate_limit import RateLimiter
from app.core.security import (
    create_access_token,
    hash_password,
    verify_dummy_password,
    verify_password,
)
from app.models.user import User
from app.repositories.user import UserRepository
from app.schemas.auth import TokenResponse, UserLogin, UserRead, UserRegister

router = APIRouter(tags=["authentication"])


def check_register_rate_limit(request: Request) -> None:
    """Enforce rate limits on user registration requests."""
    settings = get_settings()
    limiter = RateLimiter(
        key_prefix="auth_register",
        limit=settings.rate_limit_register_per_minute,
        window_seconds=60,
    )
    limiter(request)


def check_login_rate_limit(request: Request) -> None:
    """Enforce rate limits on user login requests."""
    settings = get_settings()
    limiter = RateLimiter(
        key_prefix="auth_login",
        limit=settings.rate_limit_login_per_minute,
        window_seconds=60,
    )
    limiter(request)


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_register_rate_limit)],
    summary="Register a new user account",
)
def register_user(
    payload: UserRegister,
    repo: UserRepository = Depends(get_user_repo),
) -> UserRead:
    """Register a new user with email and secure password hashing.

    Enforces:
    - Email uniqueness (HTTP 409 on duplicate, with DB race-condition safety)
    - Normalized email format (lowercased, trimmed)
    - Salted bcrypt password hashing (never stores plaintext)
    """
    existing_user = repo.get_by_email(payload.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email address already exists.",
        )

    hashed = hash_password(payload.password)
    user = User(
        email=payload.email,
        hashed_password=hashed,
        is_active=True,
    )
    try:
        created_user = repo.create(user)
    except Exception as exc:
        repo.session.rollback()
        err_str = str(exc).lower()
        if (
            "unique" in err_str
            or "integrity" in err_str
            or "could not refresh" in err_str
            or "invalidrequest" in err_str
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email address already exists.",
            ) from exc
        raise

    return UserRead.model_validate(created_user)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_login_rate_limit)],
    summary="Authenticate and receive a JWT access token",
)
def login_user(
    payload: UserLogin,
    repo: UserRepository = Depends(get_user_repo),
) -> TokenResponse:
    """Authenticate user credentials and issue a signed JWT access token.

    Enforces:
    - Constant-time verification on unknown emails to prevent timing enumeration
    - Active account check
    - Cryptographic HS256 JWT access token generation
    """
    settings = get_settings()
    user = repo.get_by_email(payload.email)
    if not user:
        verify_dummy_password()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(subject=user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )


@router.get(
    "/me",
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary="Retrieve current authenticated user identity",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserRead:
    """Return profile details for the currently authenticated principal."""
    return UserRead.model_validate(current_user)
