"""Alembic environment configuration."""

from __future__ import annotations

import logging
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel

from alembic import context
from app.core.config import get_settings

# Import all models to register them on SQLModel.metadata
from app.models.feedback import Feedback  # noqa: F401
from app.models.journal import JournalEntry  # noqa: F401
from app.models.memory import Memory  # noqa: F401
from app.models.settings import UserSettings  # noqa: F401
from app.models.user import User  # noqa: F401

config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

logger = logging.getLogger("alembic.env")

# Target metadata for autogenerate
target_metadata = SQLModel.metadata


def get_url() -> str:
    """Retrieve database URL from config or application settings."""
    import os

    # 1. Custom URL set explicitly on alembic config (e.g. in test suites)
    cfg_url = config.get_main_option("sqlalchemy.url")
    if cfg_url and cfg_url != "sqlite:///./data/consciousai.db":
        return cfg_url

    # 2. Explicit environment variable override
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    # 3. Application settings from .env
    settings = get_settings()
    if settings.database_url:
        return settings.database_url

    if cfg_url:
        return cfg_url
    return "sqlite:///./data/consciousai.db"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    is_sqlite = configuration["sqlalchemy.url"].startswith("sqlite")
    connect_args = {}
    if is_sqlite:
        connect_args["check_same_thread"] = False

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        if is_sqlite:
            connection.execute(
                SQLModel.metadata.tables.get("dummy", None) or "PRAGMA foreign_keys=ON"
            ) if False else None

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # Critical for SQLite ALTER / constraint operations
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
