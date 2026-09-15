# ConsciousAI Journal V2 — Production Security & Operational Readiness Guide

## Overview

This guide details the security architecture, configuration safeguards, authentication/authorization mechanisms, rate limiting protocols, and operational readiness requirements implemented in **Milestone 8**.

---

## 1. Environment & Secret Management

All sensitive configuration is loaded through environment variables via Pydantic Settings (`app.core.config.Settings`).

### Required Production Environment Variables
| Variable | Production Requirement | Purpose |
|---|---|---|
| `APP_ENV` | Must be `production` | Enables production security mode and disables insecure fallbacks. |
| `DEBUG` | Must be `false` | Disables debug mode and suppresses stack traces in API responses. |
| `JWT_SECRET_KEY` | At least 32 cryptographically random chars | HMAC signing key. Known placeholders and default dev keys cause immediate startup failure. |
| `JWT_ALGORITHM` | Must be `HS256` | Strictly pins the cryptographic signature algorithm. |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | E.g. `60` (1 hour) | Limits access token lifetime. |
| `CORS_ORIGINS` | Explicit domain list (e.g. `https://consciousai.app`) | Wildcards (`*`) cause startup failure in production. |
| `RATE_LIMIT_ENABLED` | `true` | Enables sliding-window rate limiting on sensitive routes. |
| `RATE_LIMIT_LOGIN_PER_MINUTE` | E.g. `10` | Max login attempts per client IP per minute. |
| `RATE_LIMIT_REGISTER_PER_MINUTE` | E.g. `5` | Max registration attempts per client IP per minute. |
| `SECURITY_HEADERS_ENABLED` | `true` | Injects HSTS, nosniff, frame denial, and CSP headers. |
| `MAX_REQUEST_BODY_BYTES` | E.g. `1048576` (1 MB) | Protects against memory exhaustion and payload bloat. |
| `ENABLE_API_DOCS` | `true` or `false` | Controls public exposure of `/docs`, `/redoc`, and `/openapi.json`. |

### Startup Validation Safeguards
In `APP_ENV=production`, the application performs automated pre-flight checks:
1. Rejects startup if `DEBUG` is `true`.
2. Rejects startup if `JWT_SECRET_KEY` is missing, shorter than 32 characters, or matches any known placeholder.
3. Rejects startup if `JWT_ALGORITHM` is not `HS256`.
4. Rejects startup if `CORS_ORIGINS` contains `*`.

---

## 2. JWT Security & Cryptographic Hardening

- **Signature Algorithm**: Explicitly pinned to `HS256`. Client-manipulated algorithm headers (including `none`, `RS256`, `HS384`, etc.) are unconditionally rejected.
- **Required Claims**: Every access token strictly requires and validates `sub` (subject), `exp` (expiration), `iat` (issued at), and `nbf` (not before).
- **Subject Validation**: Tokens with empty or whitespace-only subject claims are rejected.
- **Clock-Skew Tolerance**: Configurable leeway (default `0s`) prevents premature expiration while avoiding loose validity windows.
- **Error Sanitization**: Token verification errors return safe `HTTP 401 Unauthorized` responses without exposing internal exceptions, stack traces, or signing keys.

---

## 3. Password Security & Timing Protection

- **Hashing**: Salted bcrypt hashing with 12 rounds (`hash_password`). Each hash contains a cryptographically random salt.
- **Length Boundaries**:
  - Minimum: 8 characters (`UserRegister` schema).
  - Maximum: 72 UTF-8 bytes (`hash_password` enforces strict byte counting). Passwords exceeding 72 bytes are safely rejected.
- **Side-Channel Timing Defense**: During login, if an email address is not found in the database, `verify_dummy_password()` executes a real bcrypt verification against a constant dummy hash, equalizing response times and defeating account enumeration.
- **Generic Error Responses**: Both non-existent emails and incorrect passwords return `HTTP 401 Unauthorized` with the uniform message `"Incorrect email or password."`.

---

## 4. Authorization & User Isolation

- **Principal Derivation**: `get_current_user` cryptographically verifies the Bearer token and retrieves the active user from the database.
- **Identity Integrity**: `get_current_user_id` derives identity exclusively from the authenticated JWT principal.
- **Header Spoofing Elimination**:
  - In `APP_ENV=production`, requests without a valid Bearer token strictly fail with `HTTP 401`. The `X-User-ID` header is completely ignored.
  - Supplying an `X-User-ID` header alongside a Bearer token has zero effect; the verified JWT token always takes precedence.
- **Object-Level Isolation**: Every query and mutation in `JournalRepository`, `MemoryRepository`, and `FeedbackRepository` is scoped by `user_id`. Attempting to access or modify another user's entry returns `HTTP 404 Not Found` without disclosing resource existence.

---

## 5. Rate Limiting & Abuse Protection

### Implementation & Architecture
- **Dependency**: `RateLimiter` (`app.core.rate_limit.py`) is attached to `POST /api/v1/auth/login` and `POST /api/v1/auth/register`.
- **Sliding Window**: Requests are tracked with millisecond timestamps per client IP.
- **Rate Limit Response**: When limits are exceeded, the API returns `HTTP 429 Too Many Requests` with a `Retry-After` header indicating seconds until the window opens.

### Multi-Instance Production Requirement
> [!IMPORTANT]
> - **In-Memory Backend**: The built-in `InMemoryRateLimitBackend` tracks limits locally in process memory. It is suitable for local development, testing, and single-instance deployments.
> - **Distributed Production Clusters**: In horizontally scaled multi-instance production, multiple API containers do not share in-memory state. A shared distributed store such as Redis (`RedisRateLimitBackend`) is **required** to synchronize rate limits globally across instances.

---

## 6. HTTP Security Headers & Middleware

All responses from `app.main` pass through security middleware:
- `X-Content-Type-Options: nosniff`: Prevents MIME-type sniffing.
- `X-Frame-Options: DENY`: Prevents clickjacking in iframes.
- `Referrer-Policy: strict-origin-when-cross-origin`: Controls referrer leakage.
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`: Restricts browser features.
- `Content-Security-Policy: default-src 'self'; frame-ancestors 'none';`: Restricts executable sources on API endpoints. Documentation endpoints (`/docs`, `/redoc`) dynamically receive scoped CSP permits for Swagger UI and ReDoc CDN resources (`jsdelivr`, Google Fonts) when `ENABLE_API_DOCS=true`.
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`: Enforced in production to mandate HTTPS.

---

## 7. Global Exception Sanitization

- `app.main` registers a global unhandled exception handler for `Exception`.
- Server-side logging captures the full traceback securely for observability.
- In `APP_ENV=production`, API clients receive only:
  ```json
  {
    "detail": "Internal server error."
  }
  ```
  Internal file paths, database connection strings, and stack traces are never leaked.

---

## 8. Deployment Security Checklist

Before approving production traffic, complete the following operational steps:

- [ ] **Infrastructure Secrets**: Generate a 64+ character random secret and inject it via a secure secrets manager into `JWT_SECRET_KEY`.
- [ ] **Environment Flag**: Set `APP_ENV=production` and `DEBUG=false`.
- [ ] **TLS/HTTPS Termination**: Configure reverse proxy (Nginx/Cloudflare/Caddy) with TLS 1.3 certificates and redirect all HTTP traffic to HTTPS.
- [ ] **CORS Whitelist**: Replace development localhost origins in `CORS_ORIGINS` with exact production client origins.
- [ ] **Database Migration**: Run `alembic upgrade head` on the production database.
- [ ] **Rate Limiting Backend**: If deploying more than one API replica, connect a Redis cluster using `RedisRateLimitBackend`.
- [ ] **API Docs Visibility**: If API documentation should be private, set `ENABLE_API_DOCS=false`.
- [ ] **Database Backups**: Verify automated snapshots and point-in-time recovery on production PostgreSQL.
