"""Unit tests for rate limiting and sliding window protection."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from app.core.rate_limit import (
    InMemoryRateLimitBackend,
    RateLimiter,
    get_client_ip,
)


class TestInMemoryRateLimitBackend:
    """Test suite for sliding-window in-memory backend."""

    def test_requests_within_limit_allowed(self):
        backend = InMemoryRateLimitBackend()
        for _ in range(5):
            allowed, retry_after = backend.is_allowed("test:user1", limit=5, window_seconds=60)
            assert allowed is True
            assert retry_after == 0

    def test_request_exceeding_limit_rejected(self):
        backend = InMemoryRateLimitBackend()
        # Consume allowed 3 requests
        for _ in range(3):
            allowed, _ = backend.is_allowed("test:user2", limit=3, window_seconds=60)
            assert allowed is True

        # 4th request must be rejected
        allowed, retry_after = backend.is_allowed("test:user2", limit=3, window_seconds=60)
        assert allowed is False
        assert retry_after > 0
        assert retry_after <= 61

    def test_independent_keys_isolated(self):
        backend = InMemoryRateLimitBackend()
        # Exhaust user A
        for _ in range(2):
            backend.is_allowed("test:user_a", limit=2, window_seconds=60)
        rejected, _ = backend.is_allowed("test:user_a", limit=2, window_seconds=60)
        assert rejected is False

        # User B should still be allowed
        allowed, _ = backend.is_allowed("test:user_b", limit=2, window_seconds=60)
        assert allowed is True

    def test_window_expiration_allows_new_requests(self):
        backend = InMemoryRateLimitBackend()
        # Set limit with a 1-second window
        backend.is_allowed("test:expiring", limit=1, window_seconds=1)
        rejected, _ = backend.is_allowed("test:expiring", limit=1, window_seconds=1)
        assert rejected is False

        # Wait for sliding window to expire
        time.sleep(1.1)
        allowed, _ = backend.is_allowed("test:expiring", limit=1, window_seconds=1)
        assert allowed is True

    def test_backend_reset_clears_records(self):
        backend = InMemoryRateLimitBackend()
        backend.is_allowed("test:reset", limit=1, window_seconds=60)
        rejected, _ = backend.is_allowed("test:reset", limit=1, window_seconds=60)
        assert rejected is False

        backend.reset()
        allowed, _ = backend.is_allowed("test:reset", limit=1, window_seconds=60)
        assert allowed is True


class TestRateLimiterDependency:
    """Test suite for FastAPI RateLimiter route dependency."""

    def test_rate_limiter_raises_429_with_header(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
        backend = InMemoryRateLimitBackend()
        limiter = RateLimiter(key_prefix="auth_test", limit=2, window_seconds=60, backend=backend)

        request = MagicMock(spec=Request)
        request.client = MagicMock(host="192.168.1.50")
        request.headers = {}

        # 2 requests pass
        limiter(request)
        limiter(request)

        # 3rd request raises 429
        with pytest.raises(HTTPException) as exc_info:
            limiter(request)

        assert exc_info.value.status_code == 429
        assert "Too many requests" in exc_info.value.detail
        assert "Retry-After" in exc_info.value.headers

    def test_rate_limiter_disabled_setting_bypasses(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
        backend = InMemoryRateLimitBackend()
        limiter = RateLimiter(key_prefix="auth_test", limit=1, window_seconds=60, backend=backend)

        request = MagicMock(spec=Request)
        request.client = MagicMock(host="192.168.1.51")
        request.headers = {}

        # Even beyond limit, no exception is raised
        limiter(request)
        limiter(request)
        limiter(request)

    def test_get_client_ip_resolution(self):
        # Direct client host
        req1 = MagicMock(spec=Request)
        req1.client = MagicMock(host="10.0.0.1")
        req1.headers = {}
        assert get_client_ip(req1) == "10.0.0.1"

        # Forwarded header fallback
        req2 = MagicMock(spec=Request)
        req2.client = None
        req2.headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18"}
        assert get_client_ip(req2) == "203.0.113.195"


class TestRedisRateLimitBackend:
    """Test suite for Redis rate limiter with atomic operations and graceful failover."""

    def test_redis_allowed_and_atomic_pipeline(self):
        """Verify atomic Redis pipeline operations when under limit."""
        from app.core.rate_limit import RedisRateLimitBackend

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        # First pipe.execute() returns [removed_count, current_count, oldest_entries]
        mock_pipe.execute.side_effect = [
            [0, 2, []],  # Under limit of 5
            [1, True],  # zadd, expire results
        ]

        backend = RedisRateLimitBackend("redis://:secretpass@localhost:6379/0", client=mock_redis)
        allowed, retry_after = backend.is_allowed("rate:test_ip", limit=5, window_seconds=60)

        assert allowed is True
        assert retry_after == 0
        assert mock_pipe.zremrangebyscore.called
        assert mock_pipe.zcard.called
        assert mock_pipe.zadd.called
        assert mock_pipe.expire.called

    def test_redis_rejected_when_limit_exceeded(self):
        """Verify rejection and retry-after calculation when limit is exceeded."""
        from app.core.rate_limit import RedisRateLimitBackend

        now = time.time()
        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        # Exceeded: current count is 5 out of limit 5
        mock_pipe.execute.return_value = [0, 5, [("member1", str(now - 10))]]

        backend = RedisRateLimitBackend("redis://localhost:6379/0", client=mock_redis)
        allowed, retry_after = backend.is_allowed("rate:test_ip", limit=5, window_seconds=60)

        assert allowed is False
        assert retry_after > 0
        assert retry_after <= 60

    def test_redis_failure_falls_back_gracefully(self, caplog: pytest.LogCaptureFixture):
        """When Redis encounters an error, fall back to in-memory without raising 500."""
        from app.core.rate_limit import RedisRateLimitBackend

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_pipe.execute.side_effect = ConnectionError("Redis server connection refused")

        backend = RedisRateLimitBackend(
            "redis://:super_secret_pw@db.internal:6379/0", client=mock_redis
        )

        # Fallback should succeed seamlessly for allowed requests
        allowed, retry_after = backend.is_allowed("rate:test_fallback", limit=3, window_seconds=60)
        assert allowed is True
        assert retry_after == 0

        # Verify warning was logged and credentials are NOT exposed in logs
        assert "super_secret_pw" not in caplog.text
        assert "Falling back gracefully" in caplog.text

    def test_check_redis_health(self):
        """Verify Redis health probe behavior."""
        from app.core.rate_limit import check_redis_health

        assert check_redis_health("") is False
        assert (
            check_redis_health("redis://invalid-non-existent-host:9999/0", timeout_seconds=0.1)
            is False
        )

    def test_get_rate_limit_backend_factory(self, monkeypatch: pytest.MonkeyPatch):
        """Verify factory selects Redis backend when configured and in-memory otherwise."""
        from app.core.rate_limit import (
            InMemoryRateLimitBackend,
            RedisRateLimitBackend,
            get_rate_limit_backend,
        )

        monkeypatch.delenv("REDIS_URL", raising=False)
        backend_local = get_rate_limit_backend()
        assert isinstance(backend_local, InMemoryRateLimitBackend)

        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        backend_redis = get_rate_limit_backend()
        assert isinstance(backend_redis, RedisRateLimitBackend)
        assert backend_redis.redis_url == "redis://localhost:6379/0"


class TestInMemoryBackendPruning:
    """Verify that InMemoryRateLimitBackend bounds memory and prunes stale records."""

    def test_prune_expired_keys(self):
        backend = InMemoryRateLimitBackend(max_keys=5)
        now = time.monotonic()
        # Add 5 old entries
        for i in range(5):
            backend._hits[f"stale_key_{i}"] = [now - 200]

        # Adding 6th entry triggers pruning of stale keys
        backend.is_allowed("fresh_key", limit=10, window_seconds=60)
        assert len(backend._hits) <= 5
        assert "fresh_key" in backend._hits
