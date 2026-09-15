"""ConsciousAI Journal V2 — FastAPI Application.

Entry point for the API server. Configures logging, CORS, and registers
all API route modules.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.auth import router as auth_router
from app.api.v1.health import router as health_router
from app.api.v1.journal import router as journal_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import (
    RequestBodyLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — runs on startup and shutdown."""
    settings = get_settings()
    logger.info(
        "Starting %s v%s (env=%s, ai_provider=%s)",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.ai_provider,
    )
    logger.info("Database URL: %s", _mask_database_url(settings.database_url))
    logger.info("CORS origins: %s", settings.cors_origins_list)

    yield

    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    # Configure logging before anything else
    setup_logging(settings.log_level)

    docs_url = "/docs" if settings.enable_api_docs else None
    redoc_url = "/redoc" if settings.enable_api_docs else None
    openapi_url = "/openapi.json" if settings.enable_api_docs else None

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered self-reflection journaling application",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    # --- Security & Observability Middleware ---
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Global Exception Sanitization ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error: %s", exc)
        # In production, never leak internal tracebacks, paths, or details
        detail = "Internal server error." if get_settings().is_production else str(exc)
        return JSONResponse(
            status_code=500,
            content={"detail": detail},
        )

    # --- Routers ---
    app.include_router(health_router)
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1/auth")
    app.include_router(journal_router, prefix="/api/v1/journals")

    return app


def _mask_database_url(url: str) -> str:
    """Mask password in database URL for safe logging."""
    if "@" in url and "://" in url:
        scheme_rest = url.split("://", 1)
        if len(scheme_rest) == 2:
            auth_host = scheme_rest[1].split("@", 1)
            if len(auth_host) == 2:
                user_part = auth_host[0].split(":", 1)[0]
                return f"{scheme_rest[0]}://{user_part}:***@{auth_host[1]}"
    return url


# Create the application instance for uvicorn
app = create_app()
