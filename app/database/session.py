"""Database engine and session management.

Provides a SQLModel/SQLAlchemy engine and session factory configured
from the application settings. Supports SQLite for local development
and PostgreSQL for production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from sqlalchemy import event, text
from sqlmodel import Session, create_engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy import Engine

    from app.core.config import Settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine configured from application settings."""
    from app.core.config import get_settings

    return create_db_engine(get_settings())


def create_db_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine from application settings.

    Applies SQLite-specific pragmas (WAL mode, foreign keys) when the
    database URL targets SQLite, and robust connection pooling with pre-ping
    when targeting production PostgreSQL.

    Args:
        settings: Application settings containing DATABASE_URL and pool configs.

    Returns:
        A SQLAlchemy Engine instance.
    """
    is_sqlite = settings.database_url.startswith("sqlite")
    engine_kwargs: dict[str, Any] = {
        "echo": settings.debug and settings.is_development,
    }

    if is_sqlite:
        # SQLite requires check_same_thread=False for async/threaded usage
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Production PostgreSQL connection pooling
        engine_kwargs["pool_size"] = settings.db_pool_size
        engine_kwargs["max_overflow"] = settings.db_max_overflow
        engine_kwargs["pool_timeout"] = settings.db_pool_timeout
        engine_kwargs["pool_recycle"] = settings.db_pool_recycle
        engine_kwargs["pool_pre_ping"] = True

    engine = create_engine(
        settings.database_url,
        **engine_kwargs,
    )

    # Enable WAL mode and foreign keys for SQLite
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def check_database_health(engine: Engine, timeout_seconds: float = 3.0) -> bool:
    """Execute a lightweight SELECT 1 probe to verify database connectivity.

    Handles connection timeouts and errors safely without leaking credentials.

    Args:
        engine: SQLAlchemy Engine instance.
        timeout_seconds: Probe timeout limit in seconds.

    Returns:
        True if database responds to SELECT 1, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execution_options(timeout=timeout_seconds).execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_session(engine: Engine) -> Session:
    """Create a new database session.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        A new SQLModel Session.
    """
    return Session(engine)


def get_db_session(engine: Engine) -> Generator[Session, None, None]:
    """Dependency / context generator for explicit database sessions.

    Ensures the session is cleanly closed after use.
    """
    with Session(engine) as session:
        yield session
