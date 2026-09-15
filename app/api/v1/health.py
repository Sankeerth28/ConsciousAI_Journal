"""Health and readiness check endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    status: str
    service: str


class ReadyResponse(BaseModel):
    """Response schema for the readiness endpoint."""

    status: str
    database: str | None = None
    redis: str | None = None


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check that the service is alive (liveness probe).

    Returns a simple status confirming the process is running.
    This endpoint does not query external dependencies (PostgreSQL, Redis, AI).
    """
    return HealthResponse(status="ok", service="ConsciousAI Journal")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        200: {"model": ReadyResponse, "description": "All dependencies healthy"},
        503: {"description": "One or more critical dependencies unavailable"},
    },
)
async def readiness_check():
    """Check that application dependencies are healthy and accepting traffic (readiness probe).

    Probes:
    1. Database: verifies engine connectivity via SELECT 1 with 2.0s timeout.
    2. Redis: verifies connectivity via PING if REDIS_URL is configured with 2.0s timeout.

    Returns HTTP 200 when all required dependencies are healthy, or HTTP 503
    with sanitized diagnostic details if any dependency fails.
    """
    settings = get_settings()

    # 1. Database check
    from app.database.session import check_database_health, get_engine

    db_ok = check_database_health(get_engine(), timeout_seconds=2.0)
    db_status = "connected" if db_ok else "disconnected"

    # 2. Redis check
    redis_ok = True
    redis_status = "not_configured"
    if settings.redis_url and settings.redis_url.strip():
        from app.core.rate_limit import check_redis_health

        redis_ok = check_redis_health(settings.redis_url.strip(), timeout_seconds=2.0)
        redis_status = "connected" if redis_ok else "disconnected"

    if db_ok and redis_ok:
        return ReadyResponse(
            status="ready",
            database=db_status,
            redis=redis_status,
        )

    # Sanitized 503 response without leaking connection details or credentials
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "unhealthy",
            "database": db_status,
            "redis": redis_status,
        },
    )
