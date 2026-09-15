# Release Readiness Checklist — ConsciousAI Journal V2

Use this checklist prior to promoting any release of ConsciousAI Journal V2 to Staging or Production.

## Phase 1: Pre-Release Quality & Security Gates

- [ ] **Automated Test Suite**: All unit and integration tests pass without failure (`uv run pytest`).
- [ ] **Linting & Formatting**: Ruff checks pass cleanly (`uv run ruff check .` and `uv run ruff format --check .`).
- [ ] **Import Safety**: Heavy ML packages (`torch`, `faiss`) are not eagerly imported at startup (`uv run python scripts/check_import_safety.py`).
- [ ] **Database Migrations**: Alembic schema migrations are current and clean (`uv run alembic current`).
- [ ] **Secrets Audit**:
  - [ ] No real `.env` or credential files are tracked in Git (`git status`).
  - [ ] Default / placeholder secrets are strictly rejected in staging and production.
  - [ ] `DEBUG=true` is rejected in staging and production configurations.
- [ ] **Container Security**:
  - [ ] Multi-stage Dockerfile builds successfully (`docker build -t consciousai:test .`).
  - [ ] Container executes under non-root UID 10001 (`appuser:appgroup`).
  - [ ] Container defines a valid `HEALTHCHECK` targeting `/health`.
  - [ ] No private keys, `.env` files, or secrets exist inside the image layers.

## Phase 2: Staging Deployment Verification

- [ ] **Stack Deployment**: `docker-compose.staging.yml` starts up cleanly with healthy `web`, `db`, and `redis` containers.
- [ ] **Migration Execution**: Database migrations apply successfully on PostgreSQL 16 (`alembic upgrade head`).
- [ ] **Health Probes**:
  - [ ] Liveness probe `/health` returns HTTP 200.
  - [ ] Readiness probe `/ready` returns HTTP 200 with `database: connected` and `redis: connected`.
- [ ] **Fault-Tolerance Drill**:
  - [ ] Readiness probe `/ready` returns HTTP 503 when the database is stopped.
  - [ ] Rate limiting falls back gracefully to in-memory tracking if Redis is disconnected.
- [ ] **Load & Latency Validation**: Run `scripts/load_test.py` against staging and confirm sub-50ms p95 latency.

## Phase 3: Production Deployment & Cutover

- [ ] **Pre-Deployment Backup**: Full snapshot of PostgreSQL database taken and verified.
- [ ] **Zero-Downtime Rollout**: Apply migrations, deploy new container replicas, and verify traffic shifts only to healthy instances.
- [ ] **Post-Deployment Smoke Tests**:
  - [ ] Verify `/health` and `/ready` return HTTP 200.
  - [ ] Verify user registration, authentication, and journal entry creation.
  - [ ] Verify privacy-safe access logging (no sensitive content logged).

## Phase 4: Post-Release Monitoring

- [ ] Monitor HTTP 5xx error rates (target < 0.1%).
- [ ] Monitor database connection pool saturation.
- [ ] Monitor Redis memory and key eviction.
- [ ] Tag release in Git repository (e.g. `v2.0.0a1`).
