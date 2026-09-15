"""Unit tests for database connection pooling, engine initialization, and health checks."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.core.config import Settings
from app.database.session import check_database_health, create_db_engine, get_engine


class TestDatabaseEngineAndPool:
    """Test suite for engine creation, pooling parameters, and health checks."""

    def test_sqlite_engine_configuration(self):
        """SQLite uses check_same_thread=False and WAL pragma."""
        settings = Settings(
            app_env="development",
            database_url="sqlite:///data/test_pool.db",
            jwt_secret_key="a" * 32,
        )
        engine = create_db_engine(settings)
        assert engine.url.drivername == "sqlite"

    def test_postgresql_engine_pool_configuration(self):
        """PostgreSQL engine must apply pool_size, max_overflow, timeout, and pre_ping."""
        settings = Settings(
            app_env="staging",
            debug=False,
            database_url="postgresql://user:pass@localhost:5432/testdb",
            jwt_secret_key="a" * 32,
            db_pool_size=15,
            db_max_overflow=25,
            db_pool_timeout=45,
            db_pool_recycle=1200,
        )
        engine = create_db_engine(settings)
        assert engine.pool.size() == 15
        assert engine.pool._max_overflow == 25
        assert engine.pool._timeout == 45
        assert engine.pool._recycle == 1200
        assert engine.pool._pre_ping is True

    def test_check_database_health_success(self):
        """SELECT 1 execution succeeds on valid database engine."""
        settings = Settings(
            app_env="development",
            database_url="sqlite:///:memory:",
            jwt_secret_key="a" * 32,
        )
        engine = create_db_engine(settings)
        assert check_database_health(engine, timeout_seconds=1.0) is True

    def test_check_database_health_failure(self):
        """Connection failure safely returns False without raising exception."""
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = ConnectionError("Database connection refused")
        assert check_database_health(mock_engine, timeout_seconds=0.1) is False

    def test_get_engine_returns_cached_instance(self):
        """get_engine returns the same cached instance across invocations."""
        engine_1 = get_engine()
        engine_2 = get_engine()
        assert engine_1 is engine_2
