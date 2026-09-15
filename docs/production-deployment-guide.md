# Production Deployment Guide — ConsciousAI Journal V2

This document details the production deployment architecture, security hardening standards, and operational guidelines for ConsciousAI Journal V2.

## 1. Production Topology & Architecture

```
[ Clients / Mobile / Web ]
           │ (HTTPS / TLS 1.3)
           ▼
[ Reverse Proxy / Load Balancer (ALB / Nginx / Cloudflare) ]
           │ (HTTP / Proxy Headers)
           ▼
[ Application Cluster (FastAPI / Uvicorn) - Non-Root UID 10001 ]
     │                              │
     ▼                              ▼
[ Managed PostgreSQL 16 ]     [ Managed Redis 7 ]
  - Connection Pooling          - Distributed Rate Limiting
  - Automatic Backups           - Read/Write Atomic Pipelines
  - Point-in-Time Recovery
```

## 2. Production Hardening Requirements

The application enforces strict runtime validation when `APP_ENV=production`:
1. **`DEBUG=false`**: Server will refuse to start if `DEBUG=true`.
2. **`JWT_SECRET_KEY`**: Must be cryptographically secure and at least 32 characters in length. Default development and placeholder secrets are rejected at startup.
3. **CORS Whitelist**: Wildcard origins (`*`) are prohibited. Only explicit HTTPS domains may be specified.
4. **Non-Root Container**: Containers must execute as UID/GID 10001 (`appuser:appgroup`).
5. **Database Connection Pool**:
   - `DB_POOL_SIZE`: 10–20 per worker.
   - `DB_MAX_OVERFLOW`: 20.
   - `DB_POOL_TIMEOUT`: 30 seconds.
   - `DB_POOL_RECYCLE`: 1800 seconds.
   - `pool_pre_ping=True`: Active connection verification before query execution.

## 3. Secret Management

All sensitive values must be injected via secure environment variables or secret managers (e.g., AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager):
- `DATABASE_URL`: `postgresql://<user>:<password>@<db-host>:5432/<dbname>?sslmode=require`
- `REDIS_URL`: `rediss://:<password>@<redis-host>:6379/0` (TLS enabled)
- `JWT_SECRET_KEY`: High-entropy 256-bit secret.

> [!CAUTION]
> Never place secrets in Dockerfiles, Git repositories, compose files, or build arguments.

## 4. Zero-Downtime Deployment Procedure

1. **Pre-Deployment Database Migration**:
   - Run Alembic migrations prior to updating web replicas:
     ```bash
     alembic upgrade head
     ```
   - Ensure all schema changes maintain backward compatibility with currently running application instances.

2. **Rolling Container Update**:
   - Update container image tag.
   - Orchestrator probes `/health` (liveness) and `/ready` (readiness).
   - Traffic routes only to new replicas reporting HTTP 200 on `/ready`.
   - Old replicas are gracefully drained using `GRACEFUL_TIMEOUT=30`.

3. **Post-Deployment Verification**:
   - Check readiness: `curl -f https://api.yourdomain.com/ready`
   - Verify logs for errors using monitoring dashboards.
