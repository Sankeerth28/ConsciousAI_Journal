"""Unit tests for legacy CSV migration utilities and CLI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlmodel import Session, create_engine

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

from app.models.memory import Memory
from app.repositories.feedback import FeedbackRepository
from app.repositories.journal import JournalRepository
from app.services.csv_import import (
    compute_legacy_hash,
    normalize_legacy_feedback,
    normalize_tags_and_labels,
    parse_legacy_timestamp,
)
from scripts.import_legacy_journal import run_csv_import


class TestCSVUtilities:
    """Tests for CSV parsing and fingerprinting functions."""

    def test_legacy_hash_deterministic(self):
        h1 = compute_legacy_hash("Feeling anxious about tomorrow", "2025-02-14T10:00:00")
        h2 = compute_legacy_hash("Feeling   anxious about tomorrow  ", "2025-02-14T10:00:00")
        assert h1 == h2

    def test_same_timestamp_different_text_produces_different_hashes(self):
        ts = "2025-02-14T10:00:00"
        h1 = compute_legacy_hash("Text A", ts)
        h2 = compute_legacy_hash("Text B", ts)
        assert h1 != h2

    def test_parse_legacy_timestamp_iso(self):
        dt, err = parse_legacy_timestamp("2025-02-14T10:30:00Z")
        assert err is None
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 2
        assert dt.hour == 10

    def test_parse_legacy_timestamp_naive_with_legacy_timezone(self):
        # Naive string "2025-06-01 12:00:00" interpreted in America/New_York (UTC-4 in summer)
        dt, err = parse_legacy_timestamp(
            "2025-06-01 12:00:00", legacy_timezone_name="America/New_York"
        )
        assert err is None
        assert dt is not None
        # 12:00 in EDT is 16:00 UTC
        assert dt.hour == 16

    def test_parse_legacy_timestamp_invalid(self):
        dt, err = parse_legacy_timestamp("not-a-valid-date")
        assert dt is None
        assert "Unrecognized date format" in str(err)

    def test_normalize_legacy_feedback(self):
        assert normalize_legacy_feedback("insightful") == "Insightful"
        assert normalize_legacy_feedback("  HELPFUL ") == "Helpful"
        assert normalize_legacy_feedback("") is None
        assert normalize_legacy_feedback(None) is None

    def test_normalize_tags_and_labels(self):
        assert normalize_tags_and_labels("calm, happy, calm,  ") == ["calm", "happy"]
        assert normalize_tags_and_labels(["work", " work ", "career"]) == ["work", "career"]


class TestCSVImportExecution:
    """Tests for run_csv_import CLI logic against isolated SQLite databases."""

    def test_csv_import_success(self, tmp_path: Path):
        csv_file = tmp_path / "journal.csv"
        csv_file.write_text(
            "timestamp,text,emotion,value_theme,ai_response,feedback\n"
            "2025-01-01T10:00:00,First entry,happy,growth,Great insight!,Insightful\n"
            "2025-01-02T10:00:00,Second entry,calm,peace,Stay calm.,Helpful\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"

        # Initialize schema in DB
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        exit_code, summary = run_csv_import(
            file_path=str(csv_file),
            dry_run=False,
            database_url=db_url,
        )

        assert exit_code == 0
        assert summary.total_rows == 2
        assert summary.imported == 2
        assert summary.duplicates == 0
        assert summary.invalid == 0
        assert summary.feedback_created == 2

        with Session(engine) as session:
            j_repo = JournalRepository(session)
            fb_repo = FeedbackRepository(session)
            entries = j_repo.list()
            assert len(entries) == 2
            assert len(fb_repo.list()) == 2

    def test_same_timestamp_different_text_creates_two_entries(self, tmp_path: Path):
        csv_file = tmp_path / "journal.csv"
        ts = "2025-01-01T10:00:00"
        csv_file.write_text(
            f"timestamp,text,emotion,value_theme,ai_response,feedback\n"
            f"{ts},First entry at 10am,happy,growth,,\n"
            f"{ts},Second completely different entry at same timestamp,sad,hope,,\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        exit_code, summary = run_csv_import(
            file_path=str(csv_file),
            dry_run=False,
            database_url=db_url,
        )

        assert exit_code == 0
        assert summary.imported == 2
        with Session(engine) as session:
            entries = JournalRepository(session).list()
            assert len(entries) == 2

    def test_duplicate_import_protection(self, tmp_path: Path):
        csv_file = tmp_path / "journal.csv"
        csv_file.write_text(
            "timestamp,text,emotion,value_theme,ai_response,feedback\n"
            "2025-01-01T10:00:00,Duplicate candidate,happy,growth,,\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        # First run
        run_csv_import(str(csv_file), dry_run=False, database_url=db_url)

        # Second run with same file
        exit_code, summary = run_csv_import(str(csv_file), dry_run=False, database_url=db_url)
        assert exit_code == 0
        assert summary.imported == 0
        assert summary.duplicates == 1

        with Session(engine) as session:
            entries = JournalRepository(session).list()
            assert len(entries) == 1

    def test_dry_run_mode_creates_no_records(self, tmp_path: Path):
        csv_file = tmp_path / "journal.csv"
        csv_file.write_text(
            "timestamp,text,emotion,value_theme,ai_response,feedback\n"
            "2025-01-01T10:00:00,Dry run entry,happy,growth,,\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        exit_code, summary = run_csv_import(str(csv_file), dry_run=True, database_url=db_url)
        assert exit_code == 0
        assert summary.imported == 1  # Simulated imported count

        with Session(engine) as session:
            entries = JournalRepository(session).list()
            assert len(entries) == 0  # No database changes

    def test_migration_never_creates_memories(self, tmp_path: Path):
        csv_file = tmp_path / "journal.csv"
        csv_file.write_text(
            "timestamp,text,emotion,value_theme,ai_response,feedback\n"
            "2025-01-01T10:00:00,Journal entry,happy,growth,,\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        run_csv_import(str(csv_file), dry_run=False, database_url=db_url)

        with Session(engine) as session:
            from sqlmodel import select

            memories = list(session.exec(select(Memory)).all())
            assert len(memories) == 0

    def test_missing_file_returns_error(self, tmp_path: Path):
        exit_code, _ = run_csv_import(str(tmp_path / "nonexistent.csv"))
        assert exit_code == 1

    def test_zero_byte_csv_returns_error(self, tmp_path: Path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("", encoding="utf-8")
        exit_code, _ = run_csv_import(str(csv_file))
        assert exit_code == 1

    def test_header_only_csv_returns_zero_rows(self, tmp_path: Path):
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("timestamp,text,emotion\n", encoding="utf-8")
        exit_code, summary = run_csv_import(str(csv_file))
        assert exit_code == 0
        assert summary.total_rows == 0

    def test_missing_text_column_returns_error(self, tmp_path: Path):
        csv_file = tmp_path / "no_text.csv"
        csv_file.write_text("timestamp,emotion\n2025-01-01,happy\n", encoding="utf-8")
        exit_code, _ = run_csv_import(str(csv_file))
        assert exit_code == 1

    def test_no_journal_text_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.DEBUG)
        secret_text = "SECRET_SUPER_CONFIDENTIAL_JOURNAL_CONTENT"
        csv_file = tmp_path / "journal.csv"
        csv_file.write_text(
            f"timestamp,text\n2025-01-01T10:00:00,{secret_text}\n",
            encoding="utf-8",
        )
        db_file = tmp_path / "test.db"
        db_url = f"sqlite:///{db_file}"
        engine = create_engine(db_url)
        from sqlmodel import SQLModel

        SQLModel.metadata.create_all(engine)

        run_csv_import(str(csv_file), database_url=db_url, verbose=True)

        for record in caplog.records:
            assert secret_text not in record.message
