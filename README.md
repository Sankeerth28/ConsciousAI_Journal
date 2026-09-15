## 🧠 ConsciousAI Journal V2

An AI-powered self-reflection journaling application that helps you explore your thoughts and feelings through intelligent, empathetic analysis.

> **Project Status:** Milestone 9 Complete ✅  
> Deployment, Observability, Performance & Release Readiness: Distributed atomic Redis rate limiting with bounded failover, independent liveness (`/health`) and dependency readiness (`/ready`) probes, privacy-safe observability with `X-Request-ID` correlation, non-root multi-stage Docker container (UID 10001), Docker Compose staging environment, CI/CD pipeline, and comprehensive operational deployment guides.  
> **Final Verdict:** READY FOR STAGING

---

## ✨ Features & Capabilities

- **AI-Powered Reflections** — Contextual, empathetic responses to journal entries (Supportive, Coach, Therapist-Style Reflection, Neutral)
- **Approved Semantic Memory Retrieval** — Retrieves only explicitly approved, non-deleted memories; strictly revalidated against database before injection
- **Prompt Injection Defense** — Past memories are framed with strict containment delimiters as historical data, never executed as instructions
- **Strict Safety Order** — Validation → Crisis/Medical Interception → Emotion/Value Classification → Memory Retrieval → Reflection Generation → Output Safety Validation
- **Region-Aware Crisis Boundaries** — Configurable crisis resources (US, CA, UK, AU, IN, Global) with emergency numbers and trusted-person guidance
- **Medical & Clinical Boundary Guard** — Clear non-clinical framing; refuses medical diagnoses and prescriptions
- **Post-Generation Output Safety** — Rejects diagnosis claims, medication instructions, therapist claims, and dependency-forming language
- **Emotion & Value Analysis** — Detection of emotional tone (8 categories) and core values (7 categories) with bounded confidence scores [0.0, 1.0]
- **Resilient Fallbacks** — Emotion-mapped static fallbacks guarantee the user never encounters raw errors
- **Zero-Overhead Offline Testing** — Deterministic mock providers and LocalVectorStore enable fast, offline testing; real HF smoke tests explicitly opt-in
- **Relational Data Layer** — SQLModel (SQLite/PostgreSQL) with Alembic migrations and repository pattern (002_memory_embeddings)
- **Legacy CSV Migration** — Resilient SHA-256 fingerprint deduplication migrating V1 entries into V2
- **Privacy-First & Zero Leakage** — Raw journal text, prompts, memories, reflections, and API tokens are never logged or leaked

---

## 🏗️ Architecture

```
app/                    FastAPI backend
├── api/v1/             Versioned API endpoints (health, ready)
├── core/               Configuration (Pydantic Settings), logging
├── models/             SQLModel database models (JournalEntry, Memory, MemoryEmbedding, Feedback, UserSettings)
├── schemas/            Pydantic request/response schemas
├── repositories/       Data access repositories with soft-delete, approval, and embedding persistence
├── services/           Business logic, MemoryRetrievalService & CSV migration utilities
├── ai/                 AI services & provider abstraction
│   ├── interfaces.py   Protocols (LLMProvider, EmbeddingProvider, EmotionClassifier, ValueClassifier, VectorStore)
│   ├── schemas.py      Pydantic structured output and vector schemas
│   ├── safety.py       SafetyInterceptor (crisis & medical boundary protection)
│   ├── reflection_engine.py  Persona-based reflective response engine & static fallbacks
│   ├── emotion_classifier.py EmotionClassificationService (non-clinical framing)
│   ├── value_classifier.py   ValueClassificationService (thematic analysis)
│   ├── embeddings.py         EmbeddingService (384-dim dense vectors)
│   └── providers/
│       ├── mock.py           Deterministic offline mock providers & MockVectorStore
│       ├── vector_store.py   LocalVectorStore (NumPy cosine similarity + pure math fallback)
│       └── huggingface.py    Lazy-loaded HuggingFace pipeline providers (CPU/CUDA)
└── database/           Engine, WAL sessions, table initialization

alembic/                Database schema migrations (001_initial_schema, 002_memory_embeddings)
data/                   Local data directory (gitignored SQLite database)
tests/                  Pytest test suite (223 passing unit and integration tests)
docs/                   Project documentation, legacy audit, memory architecture, and AI architecture
scripts/                Legacy CSV migration and utility tools
notebooks/legacy/       Preserved original notebook prototype
```

---

## 📋 Prerequisites

- **Python 3.10–3.12**
- **pip** (recent version)
- No GPU required for development (uses mock AI provider and local SQLite)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sankeerth28/ConsciousAI_Journal.git
cd ConsciousAI_Journal
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Activate (Windows PowerShell):
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux):
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env if needed (defaults work for local development)
```

### 5. Initialize / Migrate the Database

Run Alembic migrations to create the database schema:

```bash
python -m alembic upgrade head
```

### 6. Run the API Server

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at **http://localhost:8000**.

### 7. Verify Health Endpoints

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"ok","service":"ConsciousAI Journal"}

# Readiness check
curl http://localhost:8000/ready
# → {"status":"ready"}

# API docs (auto-generated Swagger UI)
# Open http://localhost:8000/docs in your browser
```

---

## 🗄️ Database & Alembic Migrations

ConsciousAI Journal V2 uses **SQLModel** (backed by SQLAlchemy) with SQLite locally and full PostgreSQL compatibility.

### Database Models

| Table | Description | Key Fields |
|-------|-------------|------------|
| `journal_entries` | Core journal entries | `id`, `text`, `mood_score`, `top_emotion`, `top_value`, `detected_emotions` (JSON), `detected_values` (JSON), `tags` (JSON), `ai_response`, `feedback` (legacy), `legacy_source_hash`, `is_private`, `is_deleted`, `created_at`, `updated_at` |
| `memories` | Extracted memory insights | `id`, `content`, `source_entry_id` (FK), `memory_type`, `importance`, `is_approved` (defaults to `False`), `created_at`, `updated_at` |
| `feedbacks` | Canonical user feedback | `id`, `journal_entry_id` (FK), `feedback_type`, `comment`, `created_at` |
| `user_settings` | User preferences | `id`, `persona`, `memory_enabled`, `analytics_enabled`, `preferred_response_length`, `created_at`, `updated_at` |

### Migration Commands

```bash
# Upgrade database to latest revision:
python -m alembic upgrade head

# Rollback one migration:
python -m alembic downgrade -1

# Create a new migration revision:
python -m alembic revision --autogenerate -m "description_of_changes"
```

---

## 📥 Legacy CSV Migration

To migrate data from the legacy V1 prototype's `journal_log.csv`:

```bash
# 1. Preview migration without modifying the database (Dry-Run):
python scripts/import_legacy_journal.py --file data/journal_log.csv --dry-run

# 2. Run the actual migration:
python scripts/import_legacy_journal.py --file data/journal_log.csv

# 3. Specify a custom legacy timezone for naive timestamps (e.g. UTC, America/New_York):
python scripts/import_legacy_journal.py --file data/journal_log.csv --legacy-timezone America/New_York
```

### Migration Highlights & Safety Guards:
- **Deduplication via SHA-256**: Creates a fingerprint (`legacy_source_hash`) from normalized text and timestamp. Two entries with identical timestamps but distinct text are preserved; identical entries are safely skipped.
- **Feedback as Single Source of Truth**: Legacy feedback strings are canonicalized and stored in the `feedbacks` table.
- **Memory Privacy**: Journal entries never automatically become permanent memories (`Memory.is_approved` defaults to `False`).
- **Privacy Safe Logging**: Journal text is never logged or leaked to console outputs during import.

---

## 🧪 Running Tests

```bash
# Run the full test suite (289 unit and integration tests)
python -m pytest tests/ -v
```

---

## 🔍 Linting & Formatting

```bash
# Check lint rules
python -m ruff check app/ tests/ scripts/

# Format check
python -m ruff format --check app/ tests/ scripts/

# Automatically fix lint and format issues
python -m ruff check --fix app/ tests/ scripts/ && python -m ruff format app/ tests/ scripts/
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | ConsciousAI Journal | Application display name |
| `APP_ENV` | development | Environment (`development` / `production`) |
| `DEBUG` | true | Enable debug mode |
| `DATABASE_URL` | sqlite:///./data/consciousai.db | Database connection string |
| `LOG_LEVEL` | INFO | Logging level |
| `AI_PROVIDER` | mock | AI provider (mock/huggingface) |
| `LLM_MODEL_NAME` | google/flan-t5-large | LLM model identifier |
| `EMBEDDING_MODEL_NAME` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `EMOTION_MODEL_NAME` | facebook/bart-large-mnli | Emotion classifier model |
| `VALUE_MODEL_NAME` | facebook/bart-large-mnli | Value classifier model |
| `JWT_SECRET_KEY` | *(dev default)* | Cryptographic HMAC secret key (>=32 chars required in prod) |
| `JWT_ALGORITHM` | HS256 | JWT signing algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | 1440 | Access token validity duration in minutes |
| `HF_TOKEN` | *(empty)* | HuggingFace API token |
| `CORS_ORIGINS` | http://localhost:5173,http://localhost:3000 | Allowed CORS origins |

---

## 🗺️ Roadmap

See [`docs/v2-roadmap.md`](docs/v2-roadmap.md) for the full roadmap.

| Milestone | Status | Details |
|-----------|--------|---------|
| 1. Repository Audit | ✅ Complete | Audit of legacy notebook, documentation, inventory |
| 2. Project Foundation | ✅ Complete | FastAPI skeleton, Pydantic settings, logging, Makefile |
| 3. Database & Repository Layer | ✅ Complete | SQLModel models, Alembic, repositories, CSV migration |
| 4. AI Provider Abstraction | ✅ Complete | Provider interfaces, Mock/HuggingFace, Safety, Reflection |
| 5. Memory System | ✅ Complete | Vector store abstraction, local vector store, approval gate, user isolation |
| 6. Journal API | ✅ Complete | REST endpoints for journal CRUD, feedback, search, filters, export, user boundary |
| 7. Production Authentication | ✅ Complete | Signed JWT access tokens (HS256), salted bcrypt hashing, production spoofing rejection |
| 8. Security Hardening & Readiness | ✅ Complete | Strict config validation, rate limiting, security headers, exception sanitization |
| 9. Deployment, Observability & Readiness | ✅ Complete | Multi-stage Docker, UID 10001, Redis rate limit & failover, /health & /ready probes, CI/CD, guides |
| 10. Modern Frontend | 🔲 Planned | React + Vite + TypeScript dashboard & editor |
| 11. End-to-End Release Validation | 🔲 Planned | Full integration E2E flows & final production promotion |

---

## ⚠️ Current Limitations & Deployment Guidelines

- **Staging Readiness** — Staging containerization, database pooling, Redis rate limiting, health probes, and CI/CD pipelines are verified and complete.
- **Production Status** — The application architecture is `READY FOR STAGING`. Production promotion requires running against managed cloud infrastructure (PostgreSQL 16, Redis 7, TLS termination, KMS/Vault secrets) and completing the release checklist in [`docs/release-checklist.md`](docs/release-checklist.md).
- **Redis Fallback** — In-memory rate limiting gracefully handles Redis unavailability or disconnected states without failing user requests with 500. Memory is bounded to prevent resource exhaustion.
- **Privacy Assurance** — Access logs Correlate requests via `X-Request-ID` and record latency/status while strictly omitting passwords, tokens, journal bodies, and secrets.
- **No Frontend UI yet** — Modern React frontend will be built in the next milestone.

---

## 🔒 Privacy & Safety

- All data is stored locally by default (SQLite WAL mode)
- No journal content is sent to external services when using the mock provider
- SHA-256 fingerprinting prevents data corruption and duplication
- The AI is **not a therapist** — it provides reflective prompts, not medical advice

---

## 📜 Legacy

The original notebook prototype is preserved at [`notebooks/legacy/Concious_updated.ipynb`](notebooks/legacy/Concious_updated.ipynb).

See [`docs/legacy-audit.md`](docs/legacy-audit.md) for a detailed analysis of the original codebase.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
