"""Security middleware for HTTP response headers and request body limits."""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings

logger = logging.getLogger("app.access")

if TYPE_CHECKING:
    from collections.abc import Callable


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing standard secure HTTP headers on all API responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response: Response = await call_next(request)
        settings = get_settings()

        if not settings.security_headers_enabled:
            return response

        headers = response.headers
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        # Route-specific Content-Security-Policy
        path = request.url.path
        if settings.enable_api_docs and path.startswith("/docs"):
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "font-src 'self' data:; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
        elif settings.enable_api_docs and path.startswith("/redoc"):
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com data:; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "connect-src 'self'; "
                "worker-src 'self' blob:; "
                "frame-ancestors 'none';"
            )
        else:
            headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"

        # Enforce HSTS in production environments
        if settings.is_production:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Middleware rejecting requests with Content-Length exceeding configured limits."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        content_length = request.headers.get("Content-Length")

        if content_length:
            try:
                length = int(content_length)
                if length > settings.max_request_body_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request body exceeds maximum allowed limit of {settings.max_request_body_bytes} bytes."
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request ID generation and privacy-safe access logging.

    Observability guarantees:
    - Injects unique X-Request-ID into request state and response headers.
    - Logs HTTP method, URL path, status code, and latency in milliseconds.
    - STRICT PRIVACY: NEVER logs request body, authorization header, token,
      password, journal entry content, or other sensitive details.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id or len(request_id) > 128:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        # Safe structured logging without bodies or auth tokens
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response
