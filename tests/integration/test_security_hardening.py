"""Integration tests for Milestone 8 production security hardening and operational readiness."""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

if TYPE_CHECKING:
    from sqlalchemy import Engine

from app.api.deps import get_db
from app.core.config import Settings
from app.core.rate_limit import get_default_rate_limit_backend
from app.main import app


@pytest.fixture(name="client")
def client_fixture(engine: Engine):
    """Provide a TestClient with database session overridden per-request."""

    def _override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    test_client = TestClient(app, raise_server_exceptions=False)
    yield test_client
    app.dependency_overrides.clear()


class TestSecurityHeaders:
    """Test suite ensuring standard security headers are attached to responses."""

    def test_security_headers_present_on_response(self, client: TestClient):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "geolocation=()" in res.headers.get("Permissions-Policy", "")
        assert "default-src 'self'" in res.headers.get("Content-Security-Policy", "")

    def test_hsts_header_in_production(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )
        res = client.get("/health")
        assert res.status_code == 200
        assert "max-age=31536000" in res.headers.get("Strict-Transport-Security", "")

    def test_docs_endpoint_csp_allows_swagger_ui_assets(self, client: TestClient):
        """Verify /docs endpoint allows required Swagger UI CDN and inline assets."""
        res = client.get("/docs")
        assert res.status_code == 200
        csp = res.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "https://cdn.jsdelivr.net" in csp
        assert "https://fastapi.tiangolo.com" in csp
        assert "'unsafe-inline'" in csp
        assert "'unsafe-eval'" not in csp

    def test_redoc_endpoint_csp_allows_redoc_assets(self, client: TestClient):
        """Verify /redoc endpoint allows required ReDoc CDN, Google Fonts, and worker assets."""
        res = client.get("/redoc")
        assert res.status_code == 200
        csp = res.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "https://cdn.jsdelivr.net" in csp
        assert "https://fonts.googleapis.com" in csp
        assert "https://fonts.gstatic.com" in csp
        assert "worker-src 'self' blob:" in csp
        assert "'unsafe-inline'" in csp
        assert "'unsafe-eval'" not in csp

    def test_api_routes_retain_strict_csp(self, client: TestClient):
        """Verify non-documentation API routes maintain strictly locked down CSP."""
        for path in ("/health", "/ready"):
            res = client.get(path)
            assert res.status_code in (200, 503)
            csp = res.headers.get("Content-Security-Policy", "")
            assert csp == "default-src 'self'; frame-ancestors 'none';"
            assert "cdn.jsdelivr.net" not in csp
            assert "'unsafe-inline'" not in csp


class TestRequestBodyLimit:
    """Test suite ensuring request body size limits are enforced."""

    def test_large_request_body_rejected_with_413(self, client: TestClient):
        # Default limit is 1 MB. Send a request claiming Content-Length > 1MB
        oversized_headers = {
            "Content-Length": str(2 * 1024 * 1024),  # 2 MB
            "Content-Type": "application/json",
        }
        res = client.post(
            "/api/v1/journals",
            headers=oversized_headers,
            content=b"{}",
        )
        assert res.status_code == 413
        assert "exceeds maximum allowed limit" in res.json()["detail"]


class TestSafeErrorResponses:
    """Test suite ensuring uncaught exceptions do not leak stack traces in production."""

    def test_production_unhandled_exception_returns_generic_message(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv(
            "JWT_SECRET_KEY", "prod_super_secret_key_that_is_at_least_32_chars_long!"
        )

        # Trigger 500 error by querying non-integer journal ID that causes an error or route mock
        def _failing_route():
            raise RuntimeError("Database connection failure at /secret/internal/path/file.py:42")

        # Temporarily mount a failing route to test global exception handler
        app.add_api_route("/api/test-failure", _failing_route, methods=["GET"])

        res = client.get("/api/test-failure")
        assert res.status_code == 500
        data = res.json()
        assert data["detail"] == "Internal server error."
        assert "/secret/internal" not in str(data)
        assert "Traceback" not in str(data)


class TestRateLimitingIntegration:
    """Test suite for rate limiting enforcement on auth endpoints."""

    def test_login_rate_limiting_enforced(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_LOGIN_PER_MINUTE", "3")
        get_default_rate_limit_backend().reset()

        # 3 requests pass (even if invalid credentials, they hit the endpoint)
        for _ in range(3):
            res = client.post(
                "/api/v1/auth/login",
                json={"email": "ratelimit@example.com", "password": "WrongPassword123!"},
            )
            assert res.status_code == 401

        # 4th request must be rate limited (429)
        res_limited = client.post(
            "/api/v1/auth/login",
            json={"email": "ratelimit@example.com", "password": "WrongPassword123!"},
        )
        assert res_limited.status_code == 429
        assert "Too many requests" in res_limited.json()["detail"]
        assert "Retry-After" in res_limited.headers

    def test_register_rate_limiting_enforced(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        monkeypatch.setenv("RATE_LIMIT_REGISTER_PER_MINUTE", "2")
        get_default_rate_limit_backend().reset()

        # 2 requests pass
        res1 = client.post(
            "/api/v1/auth/register",
            json={"email": "reg1@example.com", "password": "SecurePassword123!"},
        )
        assert res1.status_code == 201

        res2 = client.post(
            "/api/v1/auth/register",
            json={"email": "reg2@example.com", "password": "SecurePassword123!"},
        )
        assert res2.status_code == 201

        # 3rd request is rate limited
        res3 = client.post(
            "/api/v1/auth/register",
            json={"email": "reg3@example.com", "password": "SecurePassword123!"},
        )
        assert res3.status_code == 429
        assert "Retry-After" in res3.headers


class TestCrossUserAuthorizationBoundaries:
    """Comprehensive test suite proving one user cannot access or mutate another user's data."""

    def test_cross_user_isolation_on_all_journal_operations(self, client: TestClient):
        # Register User Alpha
        client.post(
            "/api/v1/auth/register",
            json={"email": "alpha@example.com", "password": "AlphaPassword123!"},
        )
        token_alpha = client.post(
            "/api/v1/auth/login",
            json={"email": "alpha@example.com", "password": "AlphaPassword123!"},
        ).json()["access_token"]

        # Register User Beta
        client.post(
            "/api/v1/auth/register",
            json={"email": "beta@example.com", "password": "BetaPassword123!"},
        )
        token_beta = client.post(
            "/api/v1/auth/login",
            json={"email": "beta@example.com", "password": "BetaPassword123!"},
        ).json()["access_token"]

        headers_alpha = {"Authorization": f"Bearer {token_alpha}"}
        headers_beta = {"Authorization": f"Bearer {token_beta}"}

        # Alpha creates entry
        create_res = client.post(
            "/api/v1/journals",
            headers=headers_alpha,
            json={"text": "Alpha's secret entry.", "mood_score": 9.0},
        )
        assert create_res.status_code == 201
        entry_id = create_res.json()["entry"]["id"]

        # Beta cannot GET Alpha's entry (404)
        get_res = client.get(f"/api/v1/journals/{entry_id}", headers=headers_beta)
        assert get_res.status_code == 404

        # Beta cannot PATCH Alpha's entry (404)
        patch_res = client.patch(
            f"/api/v1/journals/{entry_id}",
            headers=headers_beta,
            json={"mood_score": 1.0},
        )
        assert patch_res.status_code == 404

        # Beta cannot submit feedback on Alpha's entry (404)
        fb_post = client.post(
            f"/api/v1/journals/{entry_id}/feedback",
            headers=headers_beta,
            json={"feedback_type": "helpful"},
        )
        assert fb_post.status_code == 404

        # Beta cannot read feedback for Alpha's entry (404)
        fb_get = client.get(f"/api/v1/journals/{entry_id}/feedback", headers=headers_beta)
        assert fb_get.status_code == 404

        # Beta cannot restore Alpha's entry (404)
        restore_res = client.post(
            f"/api/v1/journals/{entry_id}/restore",
            headers=headers_beta,
        )
        assert restore_res.status_code == 404

        # Beta cannot DELETE Alpha's entry (404)
        del_res = client.delete(f"/api/v1/journals/{entry_id}", headers=headers_beta)
        assert del_res.status_code == 404

        # Alpha's entry is still untouched and retrievable by Alpha
        alpha_verify = client.get(f"/api/v1/journals/{entry_id}", headers=headers_alpha)
        assert alpha_verify.status_code == 200
        assert alpha_verify.json()["text"] == "Alpha's secret entry."


class TestJWTAlgorithmConfusion:
    """Test suite ensuring algorithm manipulation and key confusion are rejected."""

    def test_algorithm_none_rejected(self, client: TestClient):
        # Forge unsigned token with "none" algorithm
        forged_token = jwt.encode(
            {"sub": "victim_user", "exp": 2000000000, "iat": 1700000000, "nbf": 1700000000},
            key="",
            algorithm="none",
        )
        res = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {forged_token}"},
        )
        assert res.status_code == 401

    def test_unsupported_asymmetric_alg_rejected(self, client: TestClient):
        settings = Settings()
        forged_token = jwt.encode(
            {"sub": "victim_user", "exp": 2000000000, "iat": 1700000000, "nbf": 1700000000},
            settings.jwt_secret_key,
            algorithm="HS384",
        )
        res = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {forged_token}"},
        )
        assert res.status_code == 401


class TestConcurrentRegistrationSafety:
    """Test suite verifying database unique constraint handles concurrent race conditions."""

    def test_concurrent_duplicate_registration_returns_409(self, client: TestClient):
        target_email = "concurrent_race@example.com"
        results = []

        def _attempt_registration(idx: int):
            return client.post(
                "/api/v1/auth/register",
                json={"email": target_email, "password": f"Password{idx}12345!"},
            )

        # Run 4 registration requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_attempt_registration, i) for i in range(4)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result().status_code)

        # Exactly 1 request must succeed with 201 Created
        assert results.count(201) == 1
        # All others must be rejected with 409 Conflict
        assert results.count(409) == 3
