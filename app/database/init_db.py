"""Database initialization.

Creates tables and runs startup database setup.
In production, Alembic migrations should be used instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlmodel import SQLModel, select

# Import all models so they are registered in SQLModel.metadata
from app.models.feedback import Feedback  # noqa: F401
from app.models.journal import JournalEntry  # noqa: F401
from app.models.memory import Memory  # noqa: F401
from app.models.settings import UserSettings

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)


def init_db(engine: Engine) -> None:
    """Initialize database tables and default configuration.

    Args:
        engine: SQLAlchemy Engine instance.
    """
    logger.info("Initializing database tables...")
    SQLModel.metadata.create_all(engine)
    logger.info("Database tables initialized.")

    # Seed default user settings if not present
    from app.database.session import get_session

    with get_session(engine) as session:
        existing = session.exec(select(UserSettings).where(UserSettings.id == 1)).first()
        if not existing:
            default_settings = UserSettings(id=1)
            session.add(default_settings)
            session.commit()
            logger.info("Seeded default UserSettings (id=1).")
