# Staging Deployment Guide — ConsciousAI Journal V2

This guide outlines the procedure for deploying and validating ConsciousAI Journal V2 in a staging environment.

## 1. Architecture Overview

The staging environment mirrors production using containerized services orchestrated via `docker-compose.staging.yml`:
- **Web Application (`web`)**: FastAPI application running under `uvicorn` as non-root user `appuser` (UID 10001).
- **Relational Database (`db`)**: PostgreSQL 16 Alpine container with persistent volume and connection pooling.
- **Cache & Rate Limiting (`redis`)**: Redis 7 Alpine container for distributed sliding-window rate limiting.

## 2. Prerequisites

- Docker Engine 24+ and Docker Compose v2+
- A secure environment file created from `.env.staging.example`
- Domain or ingress proxy configured with TLS termination (for staging testing)

## 3. Environment Configuration

1. Copy the staging environment template:
   ```bash
   cp .env.staging.example .env.staging
   ```
2. Generate cryptographically strong random secrets:
   ```bash
   # Generate 32-byte hex secret for JWT signing
   openssl rand -hex 32
   ```
3. Populate `.env.staging` with unique passwords and secret keys. Ensure `DEBUG=false` and `APP_ENV=staging`.

> [!IMPORTANT]
> Never commit `.env.staging` or real secrets into version control. Ensure `.gitignore` and `.dockerignore` exclude all `.env*` files except `.env.example`.

## 4. Deployment Steps

1. **Build and start the container stack**:
   ```bash
   docker compose -f docker-compose.staging.yml --env-file .env.staging up -d --build
   ```

2. **Verify container health status**:
   ```bash
   docker compose -f docker-compose.staging.yml ps
   ```
   All containers (`web`, `db`, `redis`) should transition to `healthy`.

3. **Apply database migrations**:
   ```bash
   docker compose -f docker-compose.staging.yml exec web alembic upgrade head
   ```

4. **Verify migration version**:
   ```bash
   docker compose -f docker-compose.staging.yml exec web alembic current
   ```

## 5. Health & Readiness Verification

Test both health endpoints from the host:

- **Liveness Probe** (`/health`):
  ```bash
  curl -i http://localhost:8000/health
  ```
  *Expected Response*: HTTP 200 `{"status":"ok","service":"ConsciousAI Journal"}`

- **Readiness Probe** (`/ready`):
  ```bash
  curl -i http://localhost:8000/ready
  ```
  *Expected Response*: HTTP 200 `{"status":"ready","database":"connected","redis":"connected"}`

## 6. Smoke Testing & Safe Load Validation

Execute the safe load benchmark to verify latency and stability:
```bash
python scripts/load_test.py --url http://127.0.0.1:8000 --endpoint /health -n 200 -c 10
```

Verify that requests achieve sub-50ms p95 latency with 0 HTTP 5xx errors.
