"""Integration tests for clean database initialization, migration rollback, recovery, and backup drills."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlmodel import Session, select

from alembic import command
from app.main import app
from app.models.journal import JournalEntry
from app.models.user import User

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestCleanMigrationAndRollback:
    """Validate Alembic migrations execute cleanly from scratch and support reversible rollback."""

    def test_clean_migration_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = Path(tmpdir) / "clean_test.db"
            db_url = f"sqlite:///{db_file.as_posix()}"

            # Configure Alembic for temporary clean database
            alembic_cfg = Config(str(ROOT_DIR / "alembic.ini"))
            alembic_cfg.set_main_option("script_location", str(ROOT_DIR / "alembic"))
            alembic_cfg.set_main_option("sqlalchemy.url", db_url)

            # 1. Upgrade from scratch to head
            command.upgrade(alembic_cfg, "head")

            engine = create_engine(db_url)
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            # Confirm all schema tables exist at head (004)
            assert "users" in tables
            assert "journal_entries" in tables
            assert "memories" in tables
            assert "memory_embeddings" in tables
            assert "feedbacks" in tables
            assert "user_settings" in tables
            engine.dispose()

            # 2. Test Step-by-Step Rollback (Downgrade)
            # Downgrade 004 -> 003 (removes users table)
            command.downgrade(alembic_cfg, "003")
            eng_003 = create_engine(db_url)
            inspector = inspect(eng_003)
            assert "users" not in inspector.get_table_names()
            assert "journal_entries" in inspector.get_table_names()
            eng_003.dispose()

            # Downgrade to base (clean schema rollback)
            command.downgrade(alembic_cfg, "base")
            eng_base = create_engine(db_url)
            inspector = inspect(eng_base)
            assert "journal_entries" not in inspector.get_table_names()
            assert "memories" not in inspector.get_table_names()
            eng_base.dispose()

            # 3. Re-upgrade back from base to head
            command.upgrade(alembic_cfg, "head")
            eng_head = create_engine(db_url)
            inspector = inspect(eng_head)
            assert "users" in inspector.get_table_names()
            assert "journal_entries" in inspector.get_table_names()
            eng_head.dispose()


class TestDatabaseOutageAndRecovery:
    """Verify application behavior during database outages and subsequent recovery."""

    def test_readiness_probe_fails_during_db_outage_and_recovers(self, monkeypatch):
        from app.database import session as db_session

        client = TestClient(app)

        # Baseline: DB healthy -> /ready returns 200
        res_baseline = client.get("/ready")
        assert res_baseline.status_code == 200
        assert res_baseline.json()["database"] == "connected"

        # Outage: Database fails connectivity probe
        monkeypatch.setattr(
            db_session,
            "check_database_health",
            lambda *args, **kwargs: False,
        )
        res_outage = client.get("/ready")
        assert res_outage.status_code == 503
        assert res_outage.json()["database"] == "disconnected"
        assert res_outage.json()["status"] == "unhealthy"

        # Liveness probe remains alive during DB outage
        res_live = client.get("/health")
        assert res_live.status_code == 200

        # Recovery: Database becomes reachable again
        monkeypatch.setattr(
            db_session,
            "check_database_health",
            lambda *args, **kwargs: True,
        )
        res_recovered = client.get("/ready")
        assert res_recovered.status_code == 200
        assert res_recovered.json()["database"] == "connected"


class TestBackupAndRestoreDrill:
    """Simulate disaster recovery: backup generation, database destruction, and data restoration."""

    def test_backup_and_restore_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            primary_db_path = Path(tmpdir) / "primary.db"
            backup_db_path = Path(tmpdir) / "primary.db.bak"
            db_url = f"sqlite:///{primary_db_path.as_posix()}"

            # Setup primary database with schema
            alembic_cfg = Config(str(ROOT_DIR / "alembic.ini"))
            alembic_cfg.set_main_option("script_location", str(ROOT_DIR / "alembic"))
            alembic_cfg.set_main_option("sqlalchemy.url", db_url)
            command.upgrade(alembic_cfg, "head")

            engine = create_engine(db_url)
            with Session(engine) as session:
                user = User(
                    id="u_backup_test",
                    email="drill@example.com",
                    hashed_password="hash",
                    is_active=True,
                )
                entry = JournalEntry(
                    user_id="u_backup_test",
                    text="Critical journal record before disaster drill.",
                )
                session.add(user)
                session.add(entry)
                session.commit()

            engine.dispose()

            # 1. Execute Backup (Snapshot)
            shutil.copy2(primary_db_path, backup_db_path)
            assert backup_db_path.is_file()
            assert backup_db_path.stat().st_size > 0

            # 2. Simulate Disaster (Corrupt / delete primary database)
            primary_db_path.unlink()
            assert not primary_db_path.exists()

            # 3. Restore from Backup
            shutil.copy2(backup_db_path, primary_db_path)
            assert primary_db_path.is_file()

            # 4. Verify Restored Data Integrity
            restored_engine = create_engine(db_url)
            with Session(restored_engine) as session:
                restored_user = session.exec(select(User).where(User.id == "u_backup_test")).first()
                restored_entry = session.exec(
                    select(JournalEntry).where(JournalEntry.user_id == "u_backup_test")
                ).first()

                assert restored_user is not None
                assert restored_user.email == "drill@example.com"
                assert restored_entry is not None
                assert restored_entry.text == "Critical journal record before disaster drill."

            restored_engine.dispose()
