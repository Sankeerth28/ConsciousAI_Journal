"""Integration tests for database configuration, pragmas, and Alembic migrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlmodel import Session, select

if TYPE_CHECKING:
    from pathlib import Path

from alembic import command
from app.core.config import Settings
from app.database.session import create_db_engine
from app.models.feedback import Feedback
from app.models.journal import JournalEntry


class TestDatabasePragmas:
    """Tests for SQLite-specific configuration pragmas."""

    def test_wal_mode_and_foreign_keys_enabled(self, tmp_path: Path):
        db_path = tmp_path / "pragma_test.db"
        settings = Settings(database_url=f"sqlite:///{db_path}")
        engine = create_db_engine(settings)

        # Trigger connection by executing dummy statement
        with engine.connect() as conn:
            journal_mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
            foreign_keys = conn.exec_driver_sql("PRAGMA foreign_keys").scalar()

            assert str(journal_mode).upper() == "WAL"
            assert foreign_keys == 1

        engine.dispose()

    def test_foreign_key_constraint_enforced_by_sqlite(self, tmp_path: Path):
        db_path = tmp_path / "fk_test.db"
        settings = Settings(database_url=f"sqlite:///{db_path}")
        engine = create_db_engine(settings)

        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            # Attempt inserting feedback pointing to nonexistent journal entry
            bad_feedback = Feedback(journal_entry_id=99999, feedback_type="Insightful")
            session.add(bad_feedback)
            with pytest.raises(Exception) as exc_info:
                session.commit()
            assert "FOREIGN KEY constraint failed" in str(
                exc_info.value
            ) or "IntegrityError" in str(type(exc_info.value).__name__)

        engine.dispose()


class TestAlembicMigrations:
    """Tests for Alembic migration commands and schema correctness."""

    def test_fresh_database_upgrade_head_and_idempotence(self, tmp_path: Path):
        db_path = tmp_path / "alembic_test.db"
        db_url = f"sqlite:///{db_path}"

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # 1. First upgrade to head on fresh database
        command.upgrade(alembic_cfg, "head")

        # Verify tables exist
        engine = create_engine(db_url)
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())

        expected_tables = {
            "journal_entries",
            "memories",
            "feedbacks",
            "user_settings",
            "memory_embeddings",
            "alembic_version",
        }
        assert expected_tables.issubset(tables)

        # Check indexes on journal_entries
        indexes = {ix["name"] for ix in inspector.get_indexes("journal_entries")}
        assert "ix_journal_entries_created_at" in indexes
        assert "ix_journal_entries_top_emotion" in indexes
        assert "ix_journal_entries_legacy_source_hash" in indexes

        # Check indexes on memories
        memory_indexes = {ix["name"] for ix in inspector.get_indexes("memories")}
        assert "ix_memories_is_deleted" in memory_indexes
        assert "ix_memories_is_approved_is_deleted" in memory_indexes
        assert "ix_memories_user_id" in memory_indexes

        # Check indexes on memory_embeddings
        emb_indexes = {ix["name"] for ix in inspector.get_indexes("memory_embeddings")}
        assert "ix_memory_embeddings_memory_id" in emb_indexes
        assert "ix_memory_embeddings_dimension" in emb_indexes
        assert "ix_memory_embeddings_model_name" in emb_indexes

        # 2. Second upgrade to head (idempotence verification)
        command.upgrade(alembic_cfg, "head")

        # 3. Application can write and read from migrated database
        with Session(engine) as session:
            entry = JournalEntry(text="Entry created on migrated DB")
            session.add(entry)
            session.commit()
            session.refresh(entry)

            fetched = session.exec(select(JournalEntry).where(JournalEntry.id == entry.id)).first()
            assert fetched is not None
            assert fetched.text == "Entry created on migrated DB"

        engine.dispose()

    def test_downgrade_and_reupgrade_cycle(self, tmp_path: Path):
        db_path = tmp_path / "alembic_cycle_test.db"
        db_url = f"sqlite:///{db_path}"

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # 1. Upgrade to head
        command.upgrade(alembic_cfg, "head")
        engine = create_engine(db_url)
        inspector = inspect(engine)
        assert "memory_embeddings" in inspector.get_table_names()

        # 2. Downgrade to 001_initial_schema
        command.downgrade(alembic_cfg, "001_initial_schema")
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "memory_embeddings" not in tables
        assert "memories" in tables

        # 3. Downgrade to base
        command.downgrade(alembic_cfg, "base")
        inspector = inspect(engine)
        assert "memories" not in inspector.get_table_names()

        # 4. Re-upgrade to head
        command.upgrade(alembic_cfg, "head")
        inspector = inspect(engine)
        assert "memory_embeddings" in inspector.get_table_names()
        assert "memories" in inspector.get_table_names()
        engine.dispose()

    def test_migration_against_populated_database(self, tmp_path: Path):
        db_path = tmp_path / "alembic_populated_test.db"
        db_url = f"sqlite:///{db_path}"

        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # 1. Start at 001_initial_schema
        command.upgrade(alembic_cfg, "001_initial_schema")
        settings = Settings(database_url=db_url)
        engine = create_db_engine(settings)

        # 2. Insert data at 001_initial_schema
        with engine.begin() as conn:
            from sqlalchemy import text

            conn.execute(
                text(
                    "INSERT INTO journal_entries (id, text, detected_emotions, detected_values, tags, created_at, updated_at) "
                    "VALUES (1, 'Pre-existing journal text', '[]', '[]', '[]', '2026-09-14 12:00:00', '2026-09-14 12:00:00')"
                )
            )
            # Approved memory created at 001
            conn.execute(
                text(
                    "INSERT INTO memories (id, content, source_entry_id, memory_type, importance, is_approved, created_at, updated_at) "
                    "VALUES (1, 'Pre-existing approved memory insight', 1, 'reflection', 0.8, 1, '2026-09-14 12:00:00', '2026-09-14 12:00:00')"
                )
            )
            # Unapproved memory created at 001
            conn.execute(
                text(
                    "INSERT INTO memories (id, content, source_entry_id, memory_type, importance, is_approved, created_at, updated_at) "
                    "VALUES (2, 'Pre-existing unapproved memory insight', 1, 'reflection', 0.5, 0, '2026-09-14 12:00:00', '2026-09-14 12:00:00')"
                )
            )

        # 3. Run migration 002 on populated database
        command.upgrade(alembic_cfg, "head")

        # 4. Verify existing data preserved and new columns have defaults
        from datetime import datetime, timezone

        from app.models.memory import Memory, MemoryEmbedding

        with Session(engine) as session:
            # Check approved memory
            mem1 = session.get(Memory, 1)
            assert mem1 is not None
            assert mem1.content == "Pre-existing approved memory insight"
            assert mem1.is_approved is True
            assert mem1.is_deleted is False
            assert mem1.deleted_at is None
            assert mem1.user_id is None

            # Check unapproved memory
            mem2 = session.get(Memory, 2)
            assert mem2 is not None
            assert mem2.content == "Pre-existing unapproved memory insight"
            assert mem2.is_approved is False
            assert mem2.is_deleted is False

            # Add a deleted memory under 002 schema
            mem3 = Memory(
                content="Soft-deleted memory insight",
                source_entry_id=1,
                memory_type="reflection",
                is_approved=True,
                is_deleted=True,
                deleted_at=datetime.now(timezone.utc),
            )
            session.add(mem3)
            session.commit()
            session.refresh(mem3)
            assert mem3.id is not None
            assert mem3.is_deleted is True
            mem3_id = mem3.id

            # Add embedding to approved memory (mem1)
            emb1 = MemoryEmbedding(
                memory_id=mem1.id,
                embedding_json=[0.1, 0.2, 0.3],
                dimension=3,
                provider="mock",
                model_name="test-model",
            )
            session.add(emb1)
            session.commit()

            session.refresh(mem1)
            assert mem1.embedding is not None
            assert mem1.embedding.dimension == 3
            assert mem1.embedding.embedding_json == [0.1, 0.2, 0.3]

            # Verify foreign key constraint failure for non-existent memory
            bad_emb = MemoryEmbedding(
                memory_id=99999,
                embedding_json=[0.1, 0.2, 0.3],
                dimension=3,
                provider="mock",
                model_name="test-model",
            )
            session.add(bad_emb)
            with pytest.raises(Exception) as exc_info:
                session.commit()
            assert "FOREIGN KEY constraint failed" in str(
                exc_info.value
            ) or "IntegrityError" in str(type(exc_info.value).__name__)
            session.rollback()

            # Verify cascade deletion: deleting memory 1 cascades to delete emb1
            session.delete(mem1)
            session.commit()

            assert session.get(Memory, 1) is None
            assert session.get(MemoryEmbedding, mem1.id) is None

        # 5. Verify repeated upgrade on populated database is idempotent
        command.upgrade(alembic_cfg, "head")

        # Verify records still exist after repeated upgrade
        with Session(engine) as session:
            assert session.get(Memory, 2) is not None
            assert session.get(Memory, mem3_id) is not None

        engine.dispose()
