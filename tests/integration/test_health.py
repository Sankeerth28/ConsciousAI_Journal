"""Integration tests for health and readiness endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data

    def test_health_response_values(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "ConsciousAI Journal"

    def test_health_content_type(self):
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"

    def test_health_versioned_endpoint(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestReadyEndpoint:
    """Tests for GET /ready."""

    def test_ready_returns_200(self):
        response = client.get("/ready")
        assert response.status_code == 200

    def test_ready_response_structure(self):
        response = client.get("/ready")
        data = response.json()
        assert "status" in data

    def test_ready_response_values(self):
        response = client.get("/ready")
        data = response.json()
        assert data["status"] == "ready"

    def test_ready_versioned_endpoint(self):
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_ready_returns_503_when_database_fails(self, monkeypatch):
        """Readiness check must return 503 when the database is unreachable."""
        import app.database.session

        monkeypatch.setattr(
            app.database.session, "check_database_health", lambda *args, **kwargs: False
        )
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["database"] == "disconnected"

    def test_ready_returns_503_when_configured_redis_fails(self, monkeypatch):
        """Readiness check must return 503 when configured Redis is unreachable."""
        import app.core.rate_limit

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setattr(
            app.core.rate_limit, "check_redis_health", lambda *args, **kwargs: False
        )
        response = client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["redis"] == "disconnected"

    def test_ready_success_when_configured_redis_healthy(self, monkeypatch):
        """Readiness check returns 200 when both DB and configured Redis are reachable."""
        import app.core.rate_limit

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setattr(app.core.rate_limit, "check_redis_health", lambda *args, **kwargs: True)
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"
        assert data["redis"] == "connected"

    def test_health_liveness_ignores_database_failure(self, monkeypatch):
        """Liveness probe /health must return 200 even if database is down."""
        import app.database.session

        monkeypatch.setattr(
            app.database.session, "check_database_health", lambda *args, **kwargs: False
        )
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestNotFound:
    """Tests for undefined routes."""

    def test_undefined_route_returns_404(self):
        response = client.get("/nonexistent")
        assert response.status_code == 404
