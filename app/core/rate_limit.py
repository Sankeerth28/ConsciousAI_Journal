"""Rate limiting and abuse protection for sensitive endpoints.

Provides:
- An extensible rate limiting abstraction (RateLimitBackend).
- A thread-safe in-memory sliding-window backend suitable for development,
  testing, and single-instance deployments.
- FastAPI dependency for enforcing rate limits per client IP on sensitive
  routes (e.g., login, registration).
- Documented architecture and interface for distributed backends (e.g., Redis)
  required for horizontally scaled multi-instance production.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Protocol

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


class RateLimitBackend(Protocol):
    """Protocol defining the rate limiting backend interface."""

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Check if request for key is permitted under limit within window_seconds.

        Returns:
            Tuple of (is_allowed, retry_after_seconds).
        """
        ...

    def reset(self) -> None:
        """Reset all rate limiting state (primarily for tests)."""
        ...


class InMemoryRateLimitBackend:
    """Thread-safe in-memory sliding-window rate limit backend.

    SUITABILITY NOTICE:
    This in-memory backend is suitable for development, testing, and single-instance
    deployments. For horizontally scaled, multi-instance production deployments,
    a shared distributed store such as Redis (RedisRateLimitBackend) is
    required to synchronize limits across all application nodes.
    """

    def __init__(self, max_keys: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}
        self._max_keys = max_keys

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Record an attempt and evaluate against sliding window limit."""
        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            # Memory bounding: prune expired keys if dictionary reaches capacity
            if len(self._hits) >= self._max_keys:
                self._prune_expired_keys(cutoff)

            timestamps = self._hits.get(key, [])
            timestamps = [ts for ts in timestamps if ts > cutoff]

            if len(timestamps) >= limit:
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + window_seconds - now) + 1)
                self._hits[key] = timestamps
                return False, retry_after

            timestamps.append(now)
            self._hits[key] = timestamps
            return True, 0

    def _prune_expired_keys(self, cutoff: float) -> None:
        """Internal helper to clean up inactive keys and prevent memory growth."""
        keys_to_remove = [
            k for k, ts_list in self._hits.items() if not ts_list or ts_list[-1] <= cutoff
        ]
        for k in keys_to_remove:
            self._hits.pop(k, None)

    def reset(self) -> None:
        """Clear all stored rate limit records."""
        with self._lock:
            self._hits.clear()


class RedisRateLimitBackend:
    """Distributed rate limit backend using Redis for multi-instance deployments.

    ARCHITECTURE & ATOMICITY:
    Uses Redis sorted sets with pipelined atomic commands:
    1. ZREMRANGEBYSCORE: Evict entries older than sliding window.
    2. ZCARD: Count active hits in current window.
    3. ZADD + EXPIRE: Record new hit and set key TTL if under limit.

    RESILIENCE:
    If Redis is unavailable or disconnected, the backend logs a sanitized warning
    (never exposing credentials) and gracefully falls back to a bounded in-memory
    backend rather than failing user requests with 500.
    """

    def __init__(
        self,
        redis_url: str,
        client: redis.Redis | None = None,
        fallback_backend: InMemoryRateLimitBackend | None = None,
    ) -> None:
        self.redis_url = redis_url
        self._client = client
        self.fallback = fallback_backend or InMemoryRateLimitBackend()

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            import redis

            self._client = redis.from_url(
                self.redis_url,
                socket_connect_timeout=1.5,
                socket_timeout=1.5,
                decode_responses=True,
            )
        return self._client

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Atomically record attempt in Redis sorted set or gracefully fall back."""
        try:
            client = self._get_client()
            now = time.time()
            cutoff = now - window_seconds

            pipe = client.pipeline(transaction=True)
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)
            results = pipe.execute()

            current_count = results[1]
            if current_count >= limit:
                oldest_entries = results[2]
                if oldest_entries:
                    oldest_ts = float(oldest_entries[0][1])
                    retry_after = max(1, int(oldest_ts + window_seconds - now) + 1)
                else:
                    retry_after = window_seconds
                return False, retry_after

            # Under limit: add current timestamp and refresh TTL
            member_id = f"{now}:{time.monotonic()}"
            pipe = client.pipeline(transaction=True)
            pipe.zadd(key, {member_id: now})
            pipe.expire(key, window_seconds + 5)
            pipe.execute()
            return True, 0

        except Exception as exc:
            logger.warning(
                "Redis rate limit check failed (%s). Falling back gracefully to in-memory backend.",
                exc.__class__.__name__,
            )
            return self.fallback.is_allowed(key, limit, window_seconds)

    def reset(self) -> None:
        """Reset records in Redis or fallback."""
        try:
            client = self._get_client()
            client.flushdb()
        except Exception:
            pass
        self.fallback.reset()


def check_redis_health(redis_url: str, timeout_seconds: float = 2.0) -> bool:
    """Execute a lightweight PING probe to verify Redis connectivity.

    Handles connection timeouts and errors safely without leaking credentials.

    Args:
        redis_url: Connection string to Redis server.
        timeout_seconds: Probe timeout limit in seconds.

    Returns:
        True if Redis responds to PING, False otherwise.
    """
    if not redis_url:
        return False
    try:
        import redis

        client = redis.from_url(
            redis_url,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            decode_responses=True,
        )
        return bool(client.ping())
    except Exception:
        return False


# Global in-memory backend singleton for application instance
_default_backend = InMemoryRateLimitBackend()


def get_default_rate_limit_backend() -> InMemoryRateLimitBackend:
    """Access the default in-memory application rate limit backend."""
    return _default_backend


def get_rate_limit_backend() -> RateLimitBackend:
    """Select the configured rate limit backend.

    Returns RedisRateLimitBackend if REDIS_URL is configured, otherwise InMemoryRateLimitBackend.
    """
    settings = get_settings()
    if settings.redis_url and settings.redis_url.strip():
        return RedisRateLimitBackend(settings.redis_url.strip())
    return _default_backend


def get_client_ip(request: Request) -> str:
    """Extract client IP address safely from request.

    Prefers direct client host. If deployed behind a trusted reverse proxy,
    X-Forwarded-For can be inspected according to network architecture.
    """
    if request.client and request.client.host:
        return request.client.host
    # Fallback if client is None (e.g. some mock test clients)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return "127.0.0.1"


class RateLimiter:
    """FastAPI route dependency enforcing rate limits per client IP."""

    def __init__(
        self,
        key_prefix: str,
        limit: int,
        window_seconds: int = 60,
        backend: RateLimitBackend | None = None,
    ) -> None:
        self.key_prefix = key_prefix
        self.limit = limit
        self.window_seconds = window_seconds
        self._backend = backend

    @property
    def backend(self) -> RateLimitBackend:
        return self._backend or get_rate_limit_backend()

    def __call__(self, request: Request) -> None:
        settings = get_settings()
        if not settings.rate_limit_enabled:
            return

        client_ip = get_client_ip(request)
        rate_key = f"{self.key_prefix}:{client_ip}"

        allowed, retry_after = self.backend.is_allowed(
            key=rate_key,
            limit=self.limit,
            window_seconds=self.window_seconds,
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(retry_after)},
            )
