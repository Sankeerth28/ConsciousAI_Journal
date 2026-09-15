"""Integration tests for observability, request correlation, and privacy-safe logging."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.deps import get_db
from app.main import app


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Provide a TestClient with database session overridden by test session."""

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestObservabilityMiddleware:
    """Test suite for RequestLoggingMiddleware, X-Request-ID, and log sanitization."""

    def test_request_id_generated_automatically(self, api_client: TestClient):
        """Responses must always contain an X-Request-ID header."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        request_id = response.headers["x-request-id"]
        assert len(request_id) >= 16

    def test_client_supplied_request_id_preserved(self, api_client: TestClient):
        """Custom valid X-Request-ID passed by client should be echoed back."""
        custom_id = "trace-client-id-abc-12345"
        response = api_client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200
        assert response.headers.get("x-request-id") == custom_id

    def test_sensitive_data_excluded_from_access_logs(
        self,
        api_client: TestClient,
        caplog: pytest.LogCaptureFixture,
    ):
        """Access logs must record method, path, and duration without leaking passwords or tokens."""
        import logging

        caplog.set_level(logging.INFO, logger="app.access")

        secret_password = "SuperSecretPassword123!"
        sensitive_journal_text = (
            "Extremely personal deep journal confession that must stay private."
        )

        # Simulate authentication or sensitive post
        response = api_client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": secret_password},
            headers={"X-Request-ID": "audit-test-req-001"},
        )

        assert response.headers.get("x-request-id") == "audit-test-req-001"

        # Check access logs
        access_records = [r for r in caplog.records if r.name == "app.access"]
        assert len(access_records) >= 1

        logged_messages = " ".join(r.getMessage() for r in access_records)

        # Must contain correlation and performance metadata
        assert "audit-test-req-001" in logged_messages
        assert "POST" in logged_messages
        assert "/api/v1/auth/login" in logged_messages
        assert "duration_ms=" in logged_messages

        # Must STRICTLY NOT leak sensitive content
        assert secret_password not in logged_messages
        assert sensitive_journal_text not in logged_messages
        assert "Authorization" not in logged_messages
