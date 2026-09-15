# ConsciousAI Journal V2 — Roadmap

**Date:** 2026-09-14  
**Status:** Milestone 1 Complete (Audit)

---

## Vision

Transform the ConsciousAI Journal from a single-notebook prototype into a production-grade, maintainable AI journaling application with a modern architecture, proper data layer, comprehensive testing, and a professional user interface.

---

## Target Architecture

```
conscious-ai-journal/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── journal.py         # Journal CRUD endpoints
│   │       ├── analytics.py       # Analytics endpoints
│   │       ├── memories.py        # Memory management endpoints
│   │       ├── settings.py        # User settings endpoints
│   │       └── health.py          # Health/readiness endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic settings from env vars
│   │   ├── logging.py             # Structured logging setup
│   │   └── security.py            # Input sanitization, safety checks
│   ├── models/
│   │   ├── __init__.py
│   │   └── database.py            # SQLAlchemy/SQLModel ORM models
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── journal.py             # Pydantic request/response schemas
│   │   ├── analytics.py
│   │   ├── memory.py
│   │   └── settings.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   ├── journal.py             # Journal CRUD operations
│   │   ├── memory.py              # Memory CRUD operations
│   │   ├── analytics.py           # Analytics queries
│   │   └── settings.py            # Settings CRUD
│   ├── services/
│   │   ├── __init__.py
│   │   ├── journal.py             # Journal business logic
│   │   ├── analytics.py           # Analytics computation
│   │   └── export.py              # CSV/data export
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Abstract provider interfaces
│   │   │   ├── huggingface.py     # HuggingFace implementation
│   │   │   └── mock.py            # Mock provider for tests
│   │   ├── emotion_classifier.py  # Emotion detection service
│   │   ├── value_classifier.py    # Value detection service
│   │   ├── embeddings.py          # Embedding generation
│   │   ├── reflection_engine.py   # LLM response generation
│   │   └── memory_engine.py       # Vector store + memory logic
│   └── database/
│       ├── __init__.py
│       ├── session.py             # Database session management
│       └── init_db.py             # Database initialization
├── frontend/                      # React + Vite + TypeScript
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── types/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── notebooks/
│   └── legacy/
│       └── Concious_updated.ipynb  # Original notebook (preserved)
├── docs/
│   ├── legacy-audit.md            # This audit
│   └── v2-roadmap.md              # This roadmap
├── scripts/
│   └── import_legacy_journal.py   # CSV → SQLite migration
├── migrations/                    # Alembic migrations
├── data/                          # Local data directory (gitignored)
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── Makefile
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend framework | FastAPI | Async, typed, auto-docs, modern Python |
| Data validation | Pydantic v2 | Type safety, serialization |
| ORM | SQLModel (SQLAlchemy) | Type-safe models, migration support |
| Database (dev) | SQLite | Zero-config local development |
| Database (prod) | PostgreSQL | Production-ready, same SQL dialect via SQLAlchemy |
| Migrations | Alembic | Schema versioning |
| AI/ML | Transformers, sentence-transformers | Preserve existing model choices |
| Vector store | FAISS (abstracted) | Local dev; swappable for Pinecone/Weaviate later |
| Frontend | React + Vite + TypeScript | Modern, fast, type-safe |
| Testing | Pytest | Industry standard |
| Linting | Ruff | Fast, comprehensive |
| Containerization | Docker + docker-compose | Reproducible environments |
| CI | GitHub Actions | Automated checks |

---

## Milestones

### Milestone 1 ✅ — Repository Audit
- [x] Inspect all repository files
- [x] Document current architecture
- [x] Identify bugs, security risks, performance issues
- [x] Create `docs/legacy-audit.md`
- [x] Create `docs/v2-roadmap.md`
- [x] Preserve original notebook under `notebooks/legacy/`
- [x] Identify top-priority problems

### Milestone 2 ✅ — Project Foundation
- [x] Create `pyproject.toml` with dependencies
- [x] Create `.gitignore`
- [x] Create `.env.example`
- [x] Create `app/core/config.py` (Pydantic settings)
- [x] Create `app/core/logging.py` (structured logging)
- [x] Create `app/main.py` (FastAPI app with health endpoint)
- [x] Create `Makefile` with common commands
- [x] Create basic `tests/` structure
- [x] Verify: `GET /health` returns 200

### Milestone 3 ✅ — Database & Repository Layer
- [x] Create SQLModel database models (JournalEntry, Memory, Feedback, UserSettings)
- [x] Create Alembic migration setup with SQLite batch support
- [x] Create initial migration (001_initial_schema.py)
- [x] Create repository layer (JournalRepository, MemoryRepository, FeedbackRepository, UserSettingsRepository)
- [x] Create `scripts/import_legacy_journal.py` with dry-run, SHA-256 fingerprinting, and timezone support
- [x] Verify: Database initializes, Alembic migrations pass, CSV importer passes, 74 tests passing

### Milestone 4 ✅ — AI Provider Abstraction
- [x] Create provider interfaces (LLMProvider, EmbeddingProvider, EmotionClassifier, ValueClassifier)
- [x] Create deterministic MockProviders for zero-dependency tests
- [x] Create lazy-loading HuggingFace providers (LLM, Embeddings, Emotion, Value) with CPU/CUDA detection
- [x] Extract and modernize EmotionClassificationService (non-clinical framing, fallbacks)
- [x] Extract and modernize ValueClassificationService (thematic analysis, fallbacks)
- [x] Extract EmbeddingService (384-dim dense vectors)
- [x] Create conservative SafetyInterceptor (crisis 988 lifeline, medical boundary protections)
- [x] Create ReflectionEngine (4 personas, prompt guidelines, emotion-mapped resilient static fallbacks)
- [x] Document AI architecture in `docs/ai-architecture.md`
- [x] Verify: 146 unit tests passing, zero-startup overhead, zero weight download during import or mock tests

### Milestone 5 ✅ — Safe Semantic Memory Retrieval & Vector Storage
- [x] Create vector store abstraction (`VectorStore` Protocol in `app/ai/interfaces.py`)
- [x] Create schemas: `VectorRecord`, `VectorSearchResult`, `MemorySearchResult`, `MemoryRetrievalResult`
- [x] Create `LocalVectorStore` with normalized cosine similarity `[0.0, 1.0]`, pure math fallback, and thread safety
- [x] Create deterministic `MockVectorStore` for offline tests
- [x] Create dedicated `MemoryEmbedding` SQLModel and Alembic migration `002_memory_embeddings.py` (with soft delete and user_id)
- [x] Update `MemoryRepository` with soft delete, restore, indexing eligibility, and embedding persistence
- [x] Implement `MemoryRetrievalService` enforcing database source-of-truth revalidation, user isolation, and eviction
- [x] Implement prompt injection defense containment delimiters for historical memory context
- [x] Integrate memory retrieval into `JournalReflectionPipeline` strictly after input safety interception
- [x] Add comprehensive test suite (+40 tests, 223 total passing) including privacy logging and zero unsafe downstream calls
- [x] Document AI memory architecture and Mermaid diagrams in `docs/ai-architecture.md`
- [x] *Explicit Future Work*: Automatic memory promotion, autonomous weight learning, and knowledge graphs remain future enhancements (Milestone 8+).

### Milestone 6 — Journal API
- [x] Create journal CRUD endpoints (`POST /api/v1/journals`, `GET /api/v1/journals`, `GET /api/v1/journals/{id}`, `PATCH /api/v1/journals/{id}`, `DELETE /api/v1/journals/{id}`, `POST /api/v1/journals/{id}/restore`)
- [x] Create feedback endpoints (`POST /api/v1/journals/{id}/feedback`, `GET /api/v1/journals/{id}/feedback`)
- [x] Create export endpoint (`GET /api/v1/journals/export` supporting JSON and CSV formats)
- [x] Add pagination, deterministic ordering (`created_at DESC`, `id DESC`), and multi-dimensional filtering (emotion, value_theme, tag, is_private, date range, search)
- [x] Add Pydantic v2 schemas and strict field validation rejecting immutable fields
- [x] Enforce user boundary across all queries, updates, deletes, restores, feedback, and exports
- [x] Ensure crisis-safe input pipeline order (zero persistence, zero embeddings, zero downstream calls for unsafe input)
- [x] Verify: All API endpoints pass integration tests (260 tests passed)

### Milestone 7 — Production Authentication & Authorization
- [x] Implement cryptographically signed JWT access tokens (HS256) with `sub`, `exp`, `iat`, `nbf` claims
- [x] Implement secure salted password hashing using bcrypt with 12 rounds
- [x] Create `User` SQLModel table with unique index on normalized email (`004_users_table`)
- [x] Create `UserRepository` with email normalization (lowercased, trimmed) and safe account updates
- [x] Create authentication endpoints: `POST /api/v1/auth/register`, `POST /api/v1/auth/login`, `GET /api/v1/auth/me`
- [x] Introduce `get_current_user` and update `get_current_user_id` to derive identity from verified JWT principal
- [x] Enforce production safeguards: reject missing/invalid Bearer tokens and header spoofing in `APP_ENV=production`
- [x] Block startup in production if `JWT_SECRET_KEY` uses weak or default development secrets
- [x] Preserve zero-friction local-first ergonomics with gated development fallback
- [x] Verify: All authentication and authorization unit and integration tests pass (289 tests passed)

### Milestone 8 — Production Security Hardening & Operational Readiness ✅
- [x] Security headers middleware (CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy)
- [x] Strict CORS origin validation rejecting wildcard origins in production
- [x] Request payload size limiting (1MB max body size guard)
- [x] In-memory rate limiting with Redis-ready interface and sliding window
- [x] Production credential validation and weak secret blocking
- [x] Constant-time dummy verification for unknown users
- [x] Verify: 320 tests passed, 3 skipped, zero security lints

### Milestone 9 — Deployment, Observability, Performance & Release Readiness ✅
- [x] Dockerfile multi-stage build with non-root execution (UID/GID 10001)
- [x] Docker Compose orchestration with PostgreSQL 16, Redis 7, and healthy dependencies
- [x] Shared Redis rate-limiting backend with atomic multi-exec pipelines and bounded fallback
- [x] Liveness (`/health`) and Readiness (`/ready`) probes with dependency connectivity checks
- [x] Structured JSON/key-value access logging with Request IDs and privacy sanitization
- [x] Database connection pooling (QueuePool, pool_pre_ping, connection recycling)
- [x] Local load testing benchmark (100 reqs, concurrency 10, 19.5 RPS, p50 191.9ms)
- [x] Remote load-testing safety guard preventing accidental targeting of production
- [x] Verify: 357 tests passed, 3 skipped, clean container build

### Milestone 10 — Staging Validation & Production Go/No-Go ✅
- [x] Staging configuration verified (staging/production bearer token enforcement, security headers, CORS)
- [x] Clean database migration test from scratch to head (`004_users_table`)
- [x] Step-by-step Alembic rollback verification (`004` -> `003` -> `base`) and re-upgrade
- [x] Database outage probe failure (503 Service Unavailable) and recovery (200 OK)
- [x] Database backup, disaster wipe, and data restoration drill verified
- [x] 10-step staging user journey E2E test suite implemented and passing
- [x] Docker container staging runtime verified: non-root UID 10001, `/app/data` volume permissions, periodic healthchecks
- [x] Graceful shutdown on SIGTERM and state-preserving restart verified
- [x] Zero secret leakage in Docker image layers, logs, or git commits
- [x] Comprehensive Go/No-Go Production Readiness Report compiled
- [x] Verify: 362 tests passed, 3 skipped, lint and format clean

### Milestone 11 — Production Infrastructure Activation & Final Go/No-Go 🟡
- [x] Staging multi-service cluster validated with PostgreSQL 16, Redis 7, and FastAPI
- [x] Cross-database DDL fix: PostgreSQL boolean defaults converted to `sa.false()` / `sa.true()`
- [x] Live Redis outage and recovery drill verified (HTTP 503 during outage, bounded in-memory fallback, HTTP 200 recovery)
- [x] Live PostgreSQL backup and restore drill verified via `pg_dump` and `pg_restore` (100% data integrity post-disaster)
- [x] Application rollback drill verified (failed deployment blocked by config validation, rolled back to v1.0.0-staging)
- [x] Dependency vulnerability audit passed (`uvx pip-audit`: 0 CVEs detected across 35 packages)
- [x] Realistic staging load benchmark passed (200 requests, concurrency 20, 68.7 RPS, 0 errors)
- [x] Manual authorization and cross-user isolation review passed (strict 404 on cross-user read/patch/delete, 401 on missing/spoofed tokens)
- [x] Zero sensitive data in logs or container layers audited and verified
- [ ] Managed Cloud PostgreSQL 16 (AWS RDS / GCP Cloud SQL) provisioning with PITR and automated multi-AZ snapshots (Awaiting cloud infrastructure activation)
- [ ] Managed Cloud Redis 7 (AWS ElastiCache / GCP Memorystore) with in-transit TLS encryption (Awaiting cloud infrastructure activation)
- [ ] Public TLS 1.3 certificate termination and HTTPS edge routing at ingress/reverse-proxy (Awaiting cloud ingress provisioning)
- [ ] Production KMS secret injection for `JWT_SECRET_KEY`, `DATABASE_URL`, `REDIS_URL` (Awaiting cloud secrets manager configuration)
- [ ] Centralized log ingestion pipeline (Datadog/CloudWatch/GCP Logging) with 30-day retention and PagerDuty alerts (Awaiting cloud observability setup)

---

## Final Production Recommendation

- **Milestone 11 Status**: CONDITIONAL GO
- **Application Validation**: 100% COMPLETE & VERIFIED (362 tests passed, 0 failures, zero vulnerabilities)
- **Container Packaging**: VERIFIED (non-root UID/GID 10001, multi-stage, zero secret layers)
- **Database & DDL**: VERIFIED (clean migrations to `004_users_table` on PostgreSQL 16)
- **Condition for Live DNS Activation**: Provisioning of managed cloud PostgreSQL 16, Redis 7, KMS secrets, and ingress TLS 1.3 certificate termination. Live DNS traffic switchover remains BLOCKED until cloud infrastructure provisioning is completed.
