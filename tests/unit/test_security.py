"""Unit tests for cryptographic security utilities (bcrypt and JWT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.core.config import get_settings
from app.core.security import (
    AuthenticationError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_dummy_password,
    verify_password,
)


class TestPasswordHashing:
    """Test suite for bcrypt password hashing and verification."""

    def test_hash_creation_succeeds(self):
        hashed = hash_password("MySecurePass123!")
        assert hashed.startswith("$2b$")
        assert len(hashed) > 20

    def test_unique_salts(self):
        hash1 = hash_password("SamePassword123")
        hash2 = hash_password("SamePassword123")
        assert hash1 != hash2, "Each bcrypt hash must generate a unique random salt"

    def test_correct_password_verifies(self):
        password = "CorrectHorseBatteryStaple!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_incorrect_password_rejected(self):
        hashed = hash_password("RealPassword123")
        assert verify_password("WrongPassword123", hashed) is False

    def test_empty_password_raises(self):
        with pytest.raises(ValueError, match="Password cannot be empty"):
            hash_password("")

    def test_oversized_password_raises(self):
        oversized = "A" * 73
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(oversized)

    def test_malformed_hash_handled_safely(self):
        assert verify_password("AnyPassword123", "not_a_valid_bcrypt_hash") is False
        assert verify_password("AnyPassword123", "") is False
        assert verify_password("", "$2b$12$somehash") is False

    def test_dummy_password_verification(self):
        # Must execute without throwing exceptions
        verify_dummy_password()


class TestJWTTokens:
    """Test suite for cryptographically signed JWT token generation and validation."""

    def test_create_and_decode_token(self):
        token = create_access_token(subject="user_12345", expires_delta=timedelta(minutes=30))
        payload = decode_access_token(token)

        assert payload["sub"] == "user_12345"
        assert "exp" in payload
        assert "iat" in payload
        assert "nbf" in payload
        assert payload["exp"] > payload["iat"]

    def test_expired_token_rejected(self):
        expired_token = create_access_token(
            subject="user_expired",
            expires_delta=timedelta(seconds=-10),  # expired 10 seconds ago
        )
        with pytest.raises(AuthenticationError, match="Token has expired"):
            decode_access_token(expired_token)

    def test_tampered_token_rejected(self):
        token = create_access_token(subject="user_tamper")
        # Tamper with the signature bytes
        tampered_token = token[:-5] + "XXXXX"
        with pytest.raises(AuthenticationError, match="Invalid authentication token"):
            decode_access_token(tampered_token)

    def test_token_with_wrong_secret_rejected(self):
        settings = get_settings()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        fake_token = jwt.encode(
            {"sub": "user_fake", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts},
            "completely_different_attacker_secret_key_32_chars!",
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError, match="Invalid authentication token"):
            decode_access_token(fake_token)

    def test_token_with_missing_subject_rejected(self):
        settings = get_settings()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        empty_sub_token = jwt.encode(
            {"sub": "", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError, match="Token missing valid subject claim"):
            decode_access_token(empty_sub_token)

    def test_not_before_violation_rejected(self):
        settings = get_settings()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        future_token = jwt.encode(
            {"sub": "user_future", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts + 1000},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError, match="Token not yet valid"):
            decode_access_token(future_token)

    def test_unsupported_algorithm_rejected(self):
        settings = get_settings()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        # Encode with an algorithm other than server-configured settings.jwt_algorithm
        unsupported_token = jwt.encode(
            {"sub": "user_alg", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts},
            settings.jwt_secret_key,
            algorithm="HS384",
        )
        with pytest.raises(AuthenticationError, match="algorithm"):
            decode_access_token(unsupported_token)

    def test_unsigned_token_rejected(self):
        now_ts = int(datetime.now(timezone.utc).timestamp())
        unsigned_token = jwt.encode(
            {"sub": "user_none", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts},
            key="",
            algorithm="none",
        )
        with pytest.raises(AuthenticationError, match="algorithm"):
            decode_access_token(unsigned_token)

    def test_whitespace_subject_rejected(self):
        settings = get_settings()
        now_ts = int(datetime.now(timezone.utc).timestamp())
        blank_sub_token = jwt.encode(
            {"sub": "   ", "exp": now_ts + 3600, "iat": now_ts, "nbf": now_ts},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        with pytest.raises(AuthenticationError, match="Token missing valid subject claim"):
            decode_access_token(blank_sub_token)

    def test_clock_skew_leeway(self):
        settings = get_settings()
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        # Expired 3 seconds ago
        skewed_token = jwt.encode(
            {"sub": "user_skew", "exp": now_ts - 3, "iat": now_ts - 100, "nbf": now_ts - 100},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
        # Rejects with 0 leeway
        with pytest.raises(AuthenticationError, match="Token has expired"):
            decode_access_token(skewed_token, leeway=0)

        # Accepts with 5 seconds leeway
        payload = decode_access_token(skewed_token, leeway=5)
        assert payload["sub"] == "user_skew"


class TestPasswordBoundaries:
    """Detailed boundary checks for password hashing and verification."""

    def test_exact_72_bytes_succeeds(self):
        pwd_72 = "A" * 72
        hashed = hash_password(pwd_72)
        assert verify_password(pwd_72, hashed) is True

    def test_73_bytes_rejected_in_hashing_and_verification(self):
        pwd_73 = "A" * 73
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(pwd_73)
        # verify_password should return False for oversized input rather than erroring
        assert verify_password(pwd_73, "$2b$12$dummyhashplaceholdervaluehere1234567890") is False

    def test_multibyte_utf8_boundary(self):
        # 4-byte unicode character: 🌟 (U+1F31F)
        # 18 * 4 = 72 bytes (exact boundary)
        star_18 = "🌟" * 18
        assert len(star_18.encode("utf-8")) == 72
        hashed = hash_password(star_18)
        assert verify_password(star_18, hashed) is True

        # 19 * 4 = 76 bytes (> 72 bytes)
        star_19 = "🌟" * 19
        assert len(star_19.encode("utf-8")) == 76
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(star_19)
