"""Unit tests for load testing script and target safety guards."""

from __future__ import annotations

import pytest

from scripts.load_test import run_load_test, validate_target_safety


class TestLoadTestSafetyGuards:
    """Verify safety checks prevent accidental load tests against production/remote hosts."""

    def test_localhost_allowed(self):
        validate_target_safety("http://localhost:8000")
        validate_target_safety("http://127.0.0.1:8000")
        validate_target_safety("http://0.0.0.0:8000")

    def test_remote_host_rejected_without_override(self):
        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            validate_target_safety("https://api.consciousai.com")

        with pytest.raises(ValueError, match="SAFETY VIOLATION"):
            validate_target_safety("http://192.168.1.100:8000")

    def test_remote_host_allowed_with_explicit_override(self):
        # Should not raise
        validate_target_safety("https://api.consciousai.com", allow_remote=True)


class TestLoadTestExecution:
    """Verify load test metric collection and execution."""

    @pytest.mark.anyio
    async def test_run_load_test_against_mock_or_local(self, monkeypatch):
        # Test mock execution with mock response
        class MockResponse:
            status_code = 200

        class MockAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, *args, **kwargs):
                return MockResponse()

        import httpx

        monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)

        summary = await run_load_test(
            target_url="http://127.0.0.1:8000",
            endpoint="/health",
            total_requests=10,
            concurrency=2,
        )

        assert summary["total_requests"] == 10
        assert summary["successful_requests"] == 10
        assert summary["failed_requests"] == 0
        assert "requests_per_second" in summary
        assert "latency_p50_ms" in summary
        assert "latency_p95_ms" in summary
