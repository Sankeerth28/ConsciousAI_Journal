"""Integration tests for Authentication API, JWT authorization, and production boundaries."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.deps import get_db
from app.core.config import Settings
from app.core.security import create_access_token
from app.main import app
from app.models.user import User
from app.repositories.user import UserRepository


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Provide a TestClient with database session overridden by test session."""

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestUserRegistration:
    """Test suite for user account registration."""

    def test_register_success(self, api_client: TestClient):
        response = api_client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "SecurePassword123!"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "alice@example.com"
        assert "id" in data
        assert data["is_active"] is True
        assert "password" not in data
        assert "hashed_password" not in data

    def test_register_email_normalization(self, api_client: TestClient):
        response = api_client.post(
            "/api/v1/auth/register",
            json={"email": "  Bob.Smith@Domain.COM  ", "password": "Password12345!"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "bob.smith@domain.com"

    def test_register_duplicate_email_rejected(self, api_client: TestClient):
        payload = {"email": "duplicate@example.com", "password": "Password12345!"}
        res1 = api_client.post("/api/v1/auth/register", json=payload)
        assert res1.status_code == 201

        # Attempt duplicate with different casing/whitespace
        duplicate_payload = {"email": " DUPLICATE@example.com ", "password": "AnotherPassword123!"}
        res2 = api_client.post("/api/v1/auth/register", json=duplicate_payload)
        assert res2.status_code == 409
        assert "already exists" in res2.json()["detail"].lower()

    def test_register_validation_errors(self, api_client: TestClient):
        # Invalid email
        res_email = api_client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "ValidPassword123!"},
        )
        assert res_email.status_code == 422

        # Password too short (< 8 chars)
        res_pass = api_client.post(
            "/api/v1/auth/register",
            json={"email": "valid@example.com", "password": "short"},
        )
        assert res_pass.status_code == 422

        # Extra fields forbidden
        res_extra = api_client.post(
            "/api/v1/auth/register",
            json={
                "email": "extra@example.com",
                "password": "ValidPassword123!",
                "is_superuser": True,
            },
        )
        assert res_extra.status_code == 422


class TestUserLogin:
    """Test suite for credential authentication and token issuance."""

    def test_login_success(self, api_client: TestClient):
        # Register user first
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "charlie@example.com", "password": "CharliePassword1!"},
        )

        # Login with valid credentials
        res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "charlie@example.com", "password": "CharliePassword1!"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_wrong_password(self, api_client: TestClient):
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "dave@example.com", "password": "DavePassword1!"},
        )
        res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "dave@example.com", "password": "WrongPassword!"},
        )
        assert res.status_code == 401
        assert "incorrect" in res.json()["detail"].lower()

    def test_login_unknown_email(self, api_client: TestClient):
        res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "AnyPassword123!"},
        )
        assert res.status_code == 401
        assert "incorrect" in res.json()["detail"].lower()

    def test_login_deactivated_account(self, api_client: TestClient, session: Session):
        reg_res = api_client.post(
            "/api/v1/auth/register",
            json={"email": "inactive@example.com", "password": "InactivePass123!"},
        )
        user_id = reg_res.json()["id"]

        # Deactivate user in DB
        repo = UserRepository(session)
        repo.update(user_id, is_active=False)

        res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "inactive@example.com", "password": "InactivePass123!"},
        )
        assert res.status_code == 401
        assert "deactivated" in res.json()["detail"].lower()


class TestCurrentProfileEndpoint:
    """Test suite for /api/v1/auth/me endpoint."""

    def test_me_with_valid_bearer_token(self, api_client: TestClient):
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "emma@example.com", "password": "EmmaPassword1!"},
        )
        login_res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "emma@example.com", "password": "EmmaPassword1!"},
        )
        token = login_res.json()["access_token"]

        res = api_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["email"] == "emma@example.com"
        assert data["is_active"] is True

    def test_me_with_expired_token(self, api_client: TestClient, session: Session):
        repo = UserRepository(session)
        user = User(
            email="expired_user@example.com",
            hashed_password="some_hash",
            is_active=True,
        )
        created = repo.create(user)

        expired_token = create_access_token(
            subject=created.id,
            expires_delta=timedelta(seconds=-30),
        )
        res = api_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert res.status_code == 401
        assert "credentials" in res.json()["detail"].lower()

    def test_me_with_invalid_token_format(self, api_client: TestClient):
        res = api_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid_garbage_token"},
        )
        assert res.status_code == 401


class TestProductionModeAuthorization:
    """Test suite ensuring X-User-ID cannot establish identity in production mode."""

    def test_production_mode_rejects_missing_bearer_token(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )

        # Attempting access with only X-User-ID in production must be rejected
        res = api_client.get(
            "/api/v1/journals",
            headers={"X-User-ID": "spoofed_user"},
        )
        assert res.status_code == 401
        assert "authentication required" in res.json()["detail"].lower()

    def test_production_mode_invalid_bearer_does_not_fallback(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )

        # Sending invalid Bearer with X-User-ID must return 401, not fall back
        res = api_client.get(
            "/api/v1/journals",
            headers={
                "Authorization": "Bearer bad_token",
                "X-User-ID": "spoofed_user",
            },
        )
        assert res.status_code == 401

    def test_production_mode_blocks_weak_secret(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "short")
        with pytest.raises(ValueError, match="at least 32 characters"):
            Settings()

    def test_production_mode_blocks_default_secret(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY",
            "consciousai-default-insecure-dev-secret-change-in-production",
        )
        with pytest.raises(ValueError, match="cannot use known placeholder or default"):
            Settings()


class TestJournalAccessWithJWT:
    """Test suite ensuring authenticated JWT principal enforces journal user isolation."""

    def test_journal_crud_with_bearer_token(self, api_client: TestClient):
        # Register and login User 1
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "user1@example.com", "password": "User1Password123!"},
        )
        token1 = api_client.post(
            "/api/v1/auth/login",
            json={"email": "user1@example.com", "password": "User1Password123!"},
        ).json()["access_token"]

        # Register and login User 2
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "user2@example.com", "password": "User2Password123!"},
        )
        token2 = api_client.post(
            "/api/v1/auth/login",
            json={"email": "user2@example.com", "password": "User2Password123!"},
        ).json()["access_token"]

        # User 1 creates an entry with Bearer token
        create_res = api_client.post(
            "/api/v1/journals",
            headers={"Authorization": f"Bearer {token1}"},
            json={"text": "User 1 personal reflection.", "mood_score": 8.0, "tags": ["private"]},
        )
        assert create_res.status_code == 201
        entry_id = create_res.json()["entry"]["id"]

        # User 1 can retrieve entry
        get_res1 = api_client.get(
            f"/api/v1/journals/{entry_id}",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert get_res1.status_code == 200
        assert get_res1.json()["text"] == "User 1 personal reflection."

        # User 2 cannot retrieve User 1's entry (404)
        get_res2 = api_client.get(
            f"/api/v1/journals/{entry_id}",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert get_res2.status_code == 404

        # User 2 cannot update User 1's entry (404)
        patch_res2 = api_client.patch(
            f"/api/v1/journals/{entry_id}",
            headers={"Authorization": f"Bearer {token2}"},
            json={"mood_score": 2.0},
        )
        assert patch_res2.status_code == 404

        # User 2 cannot delete User 1's entry (404)
        del_res2 = api_client.delete(
            f"/api/v1/journals/{entry_id}",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert del_res2.status_code == 404

        # User 2 cannot attach feedback to User 1's entry (404)
        fb_res2 = api_client.post(
            f"/api/v1/journals/{entry_id}/feedback",
            headers={"Authorization": f"Bearer {token2}"},
            json={"feedback_type": "helpful"},
        )
        assert fb_res2.status_code == 404

        # User 2 export does NOT contain User 1's entry
        export_res2 = api_client.get(
            "/api/v1/journals/export?format=json",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert export_res2.status_code == 200
        assert len(export_res2.json()) == 0

        # Sending both Bearer token for User 1 AND X-User-ID: User 2 -> token wins
        override_attempt = api_client.get(
            f"/api/v1/journals/{entry_id}",
            headers={
                "Authorization": f"Bearer {token1}",
                "X-User-ID": "victim_user_override",
            },
        )
        assert override_attempt.status_code == 200
        assert override_attempt.json()["id"] == entry_id
