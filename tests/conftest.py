"""Shared pytest fixtures for ConsciousAI Journal tests."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy import Engine

from app.models.feedback import Feedback  # noqa: F401
from app.models.journal import JournalEntry  # noqa: F401
from app.models.memory import Memory, MemoryEmbedding  # noqa: F401
from app.models.settings import UserSettings  # noqa: F401


def pytest_configure(config: pytest.Config) -> None:
    """Set default test environment variables before any modules are collected."""
    os.environ["APP_ENV"] = "development"
    os.environ["DEBUG"] = "true"
    os.environ["JWT_SECRET_KEY"] = "consciousai-default-insecure-dev-secret-change-in-production"
    os.environ["JWT_ALGORITHM"] = "HS256"
    os.environ["CORS_ORIGINS"] = "http://localhost:5173,http://localhost:3000"
    os.environ["RATE_LIMIT_ENABLED"] = "false"
    os.environ["REDIS_URL"] = ""


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate test runs from developer's local .env file settings."""
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv(
        "JWT_SECRET_KEY",
        "consciousai-default-insecure-dev-secret-change-in-production",
    )
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("REDIS_URL", "")
    from app.core.rate_limit import get_default_rate_limit_backend

    get_default_rate_limit_backend().reset()


@pytest.fixture(name="engine")
def engine_fixture() -> Generator[Engine, None, None]:
    """Create an isolated in-memory SQLite engine with foreign keys enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(name="session")
def session_fixture(engine: Engine) -> Generator[Session, None, None]:
    """Provide a clean database session per test function."""
    with Session(engine) as session:
        yield session
