"""Security utilities for password hashing and signed JWT authentication."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

# Pre-computed dummy hash for constant-time email enumeration mitigation
_DUMMY_BCRYPT_HASH = "$2b$12$e8YvA5f9L9L7/1s9o2pB9.u1eTj9Y2s.4F6V0s9m9e5k6l7m8n9o0"


class AuthenticationError(Exception):
    """Raised when authentication credentials or tokens cannot be validated."""


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password securely using bcrypt with a random salt.

    Args:
        plain_password: Raw plaintext password string.

    Returns:
        Hashed password string.

    Raises:
        ValueError: If password exceeds bcrypt maximum length (72 bytes) or is empty.
    """
    if not plain_password:
        msg = "Password cannot be empty"
        raise ValueError(msg)
    pwd_bytes = plain_password.encode("utf-8")
    if len(pwd_bytes) > 72:
        msg = "Password cannot exceed 72 bytes"
        raise ValueError(msg)

    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash.

    Handles malformed or incompatible hashes safely without raising unhandled exceptions.

    Args:
        plain_password: The plaintext password provided by the user.
        hashed_password: The stored bcrypt hash string.

    Returns:
        True if password matches hash, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        pwd_bytes = plain_password.encode("utf-8")
        if len(pwd_bytes) > 72:
            return False
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except (ValueError, TypeError):
        return False


def verify_dummy_password() -> None:
    """Perform a dummy bcrypt verification to equalize response timing on non-existent users."""
    with contextlib.suppress(Exception):
        bcrypt.checkpw(b"dummy_password_timing_pad", _DUMMY_BCRYPT_HASH.encode("utf-8"))


def create_access_token(
    subject: str,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Generate a cryptographically signed JWT access token.

    Args:
        subject: The unique user identifier (sub claim).
        expires_delta: Optional custom duration until token expiry.
        extra_claims: Optional dictionary of additional public claims.

    Returns:
        Encoded JWT token string.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)

    claims: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        claims.update(extra_claims)

    encoded = jwt.encode(
        claims,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return encoded


def decode_access_token(token: str, leeway: int = 0) -> dict[str, Any]:
    """Decode and validate a JWT access token against server configuration.

    Strictly enforces:
    - Signature validation using server's secret key
    - Server-configured algorithm only (never trusts token header)
    - Expiration (exp claim)
    - Not-before time (nbf claim)
    - Presence of valid non-empty subject (sub claim)
    - Leeway for clock skew (default 0s)

    Args:
        token: Raw JWT string.
        leeway: Optional clock-skew tolerance in seconds (default 0).

    Returns:
        The validated payload dictionary.

    Raises:
        AuthenticationError: If token is invalid, expired, tampered with, or missing required claims.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            leeway=leeway,
            options={
                "require": ["sub", "exp", "iat", "nbf"],
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
                "verify_signature": True,
            },
        )
    except jwt.ExpiredSignatureError as exc:
        msg = "Token has expired"
        raise AuthenticationError(msg) from exc
    except jwt.ImmatureSignatureError as exc:
        msg = "Token not yet valid"
        raise AuthenticationError(msg) from exc
    except jwt.InvalidAlgorithmError as exc:
        msg = "Unsupported token algorithm"
        raise AuthenticationError(msg) from exc
    except jwt.InvalidTokenError as exc:
        msg = "Invalid authentication token"
        raise AuthenticationError(msg) from exc

    subject = payload.get("sub")
    if subject is None or not str(subject).strip():
        msg = "Token missing valid subject claim"
        raise AuthenticationError(msg)

    return payload
