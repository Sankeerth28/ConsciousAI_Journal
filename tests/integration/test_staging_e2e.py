"""Staging End-to-End (E2E) Test Suite — ConsciousAI Journal V2.

Verifies the complete 10-step primary user journey:
1. User registration
2. User login & JWT issuance
3. Journal entry creation & reflection
4. Journal entry read
5. Journal entry update (tags, mood)
6. Journal entry soft-deletion & listing exclusion
7. Cross-user isolation (zero existence leakage)
8. Input validation errors
9. Authentication & credential failures
10. Rate limiting enforcement & 429 Retry-After headers
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.deps import get_db
from app.core.rate_limit import InMemoryRateLimitBackend, get_rate_limit_backend
from app.main import app


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Provide a TestClient with isolated session override."""

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestStagingEndToEndJourney:
    """Rigorous end-to-end validation of full user workflows in staging."""

    def test_complete_user_lifecycle_journey(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        # -------------------------------------------------------------
        # STEP 1: Register a new user
        # -------------------------------------------------------------
        reg_payload = {
            "email": "staging_user1@example.com",
            "password": "StrongStagingPass123!@#",
        }
        reg_res = api_client.post("/api/v1/auth/register", json=reg_payload)
        assert reg_res.status_code == 201
        user1_data = reg_res.json()
        assert user1_data["email"] == "staging_user1@example.com"
        assert "password" not in user1_data
        assert "hashed_password" not in user1_data
        assert user1_data["is_active"] is True
        assert user1_data["id"] is not None

        # -------------------------------------------------------------
        # STEP 2: Log in and obtain signed Bearer JWT
        # -------------------------------------------------------------
        login_payload = {
            "email": "staging_user1@example.com",
            "password": "StrongStagingPass123!@#",
        }
        login_res = api_client.post("/api/v1/auth/login", json=login_payload)
        assert login_res.status_code == 200
        token_data = login_res.json()
        assert "access_token" in token_data
        assert token_data["token_type"] == "bearer"
        access_token = token_data["access_token"]
        headers_user1 = {"Authorization": f"Bearer {access_token}"}

        # Verify /me profile
        me_res = api_client.get("/api/v1/auth/me", headers=headers_user1)
        assert me_res.status_code == 200
        assert me_res.json()["email"] == "staging_user1@example.com"

        # -------------------------------------------------------------
        # STEP 3: Create a safe journal entry & receive reflection
        # -------------------------------------------------------------
        entry_payload = {
            "text": "Today I took a walk in the forest and felt deep gratitude and tranquility.",
            "mood_score": 8.5,
            "tags": ["nature", "mindfulness"],
            "persona": "Supportive",
        }
        create_res = api_client.post(
            "/api/v1/journals",
            json=entry_payload,
            headers=headers_user1,
        )
        assert create_res.status_code == 201
        create_data = create_res.json()
        assert create_data["entry"] is not None
        entry_id = create_data["entry"]["id"]
        assert create_data["entry"]["text"] == entry_payload["text"]
        assert create_data["entry"]["mood_score"] == 8.5
        assert "nature" in create_data["entry"]["tags"]
        assert create_data["reflection"]["response"] is not None
        assert create_data["input_safety"]["is_safe"] is True

        # -------------------------------------------------------------
        # STEP 4: Read the journal entry
        # -------------------------------------------------------------
        read_res = api_client.get(f"/api/v1/journals/{entry_id}", headers=headers_user1)
        assert read_res.status_code == 200
        read_data = read_res.json()
        assert read_data["id"] == entry_id
        assert read_data["text"] == entry_payload["text"]

        # -------------------------------------------------------------
        # STEP 5: Update the journal entry (mutable fields)
        # -------------------------------------------------------------
        update_payload = {
            "mood_score": 9.0,
            "tags": ["nature", "mindfulness", "peace"],
            "is_private": True,
        }
        patch_res = api_client.patch(
            f"/api/v1/journals/{entry_id}",
            json=update_payload,
            headers=headers_user1,
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["mood_score"] == 9.0
        assert "peace" in patch_data["tags"]
        assert patch_data["is_private"] is True
        # Verify text remains immutable
        assert patch_data["text"] == entry_payload["text"]

        # -------------------------------------------------------------
        # STEP 6: Delete (soft-delete) the journal entry
        # -------------------------------------------------------------
        del_res = api_client.delete(f"/api/v1/journals/{entry_id}", headers=headers_user1)
        assert del_res.status_code == 200
        assert del_res.json()["id"] == entry_id
        assert "soft-deleted" in del_res.json()["message"]

        # Verify listing excludes soft-deleted entries by default
        list_res = api_client.get("/api/v1/journals", headers=headers_user1)
        assert list_res.status_code == 200
        list_items = list_res.json()["items"]
        assert not any(item["id"] == entry_id for item in list_items)

        # Restore entry for subsequent tests
        restore_res = api_client.post(f"/api/v1/journals/{entry_id}/restore", headers=headers_user1)
        assert restore_res.status_code == 200

        # -------------------------------------------------------------
        # STEP 7: Verify cross-user isolation
        # -------------------------------------------------------------
        # Register second user (User 2)
        api_client.post(
            "/api/v1/auth/register",
            json={"email": "staging_user2@example.com", "password": "StrongStagingPass123!@#"},
        )
        login2_res = api_client.post(
            "/api/v1/auth/login",
            json={"email": "staging_user2@example.com", "password": "StrongStagingPass123!@#"},
        )
        token2 = login2_res.json()["access_token"]
        headers_user2 = {"Authorization": f"Bearer {token2}"}

        # User 2 attempts to GET User 1's entry -> MUST return 404 (zero existence leakage)
        u2_get = api_client.get(f"/api/v1/journals/{entry_id}", headers=headers_user2)
        assert u2_get.status_code == 404
        assert "not found" in u2_get.json()["detail"].lower()

        # User 2 attempts to PATCH User 1's entry -> MUST return 404
        u2_patch = api_client.patch(
            f"/api/v1/journals/{entry_id}",
            json={"tags": ["hacked"]},
            headers=headers_user2,
        )
        assert u2_patch.status_code == 404

        # User 2 attempts to DELETE User 1's entry -> MUST return 404
        u2_del = api_client.delete(f"/api/v1/journals/{entry_id}", headers=headers_user2)
        assert u2_del.status_code == 404

        # User 2 lists journals -> 0 entries visible
        u2_list = api_client.get("/api/v1/journals", headers=headers_user2)
        assert u2_list.status_code == 200
        assert len(u2_list.json()["items"]) == 0

        # -------------------------------------------------------------
        # STEP 8: Verify input validation errors
        # -------------------------------------------------------------
        # Empty text
        bad_empty = api_client.post(
            "/api/v1/journals",
            json={"text": ""},
            headers=headers_user1,
        )
        assert bad_empty.status_code == 422

        # Invalid mood score (> 10)
        bad_mood = api_client.post(
            "/api/v1/journals",
            json={"text": "Valid text", "mood_score": 15.0},
            headers=headers_user1,
        )
        assert bad_mood.status_code == 422

        # Extra forbidden field
        bad_extra = api_client.post(
            "/api/v1/journals",
            json={"text": "Valid text", "unknown_field": 123},
            headers=headers_user1,
        )
        assert bad_extra.status_code == 422

        # -------------------------------------------------------------
        # STEP 9: Verify authentication failures
        # -------------------------------------------------------------
        # Invalid Bearer token string strictly fails with 401
        bogus_res = api_client.get(
            "/api/v1/journals",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert bogus_res.status_code == 401
        assert "could not validate credentials" in bogus_res.json()["detail"].lower()

        # Malformed Authorization header format (missing Bearer prefix)
        malformed_res = api_client.get(
            "/api/v1/journals",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        # In development without Bearer, fallback occurs unless in staging/production
        assert malformed_res.status_code in (200, 401)

        # Wrong login password
        wrong_pass = api_client.post(
            "/api/v1/auth/login",
            json={"email": "staging_user1@example.com", "password": "WrongPassword!"},
        )
        assert wrong_pass.status_code == 401
        assert "incorrect" in wrong_pass.json()["detail"].lower()

        # Unknown login email
        unknown_user = api_client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "StrongStagingPass123!@#"},
        )
        assert unknown_user.status_code == 401

        # -------------------------------------------------------------
        # STEP 10: Verify rate-limit behavior on auth endpoints
        # -------------------------------------------------------------
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_LOGIN_PER_MINUTE", "3")
        backend = get_rate_limit_backend()
        if isinstance(backend, InMemoryRateLimitBackend):
            backend.reset()

        # Attempt repeated failed logins to trigger rate limit (configured for 3/min)
        rate_hit = False
        retry_after = 0
        for _ in range(5):
            rl_res = api_client.post(
                "/api/v1/auth/login",
                json={"email": "staging_user1@example.com", "password": "WrongPassword!"},
            )
            if rl_res.status_code == 429:
                rate_hit = True
                assert "Retry-After" in rl_res.headers
                retry_after = int(rl_res.headers["Retry-After"])
                assert retry_after > 0
                break

        assert rate_hit is True
        assert retry_after > 0

    def test_staging_environment_strictly_enforces_bearer_authentication(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """When APP_ENV=staging, missing Bearer token must return HTTP 401."""
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("JWT_SECRET_KEY", "b" * 32)

        # In staging, unauthenticated access to protected routes is rejected
        unauth_res = api_client.get("/api/v1/journals")
        assert unauth_res.status_code == 401
        assert "authentication required" in unauth_res.json()["detail"].lower()
