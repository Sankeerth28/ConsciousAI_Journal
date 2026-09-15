#!/usr/bin/env python3
"""Import legacy journal CSV data into ConsciousAI Journal V2 database.

Usage:
    python scripts/import_legacy_journal.py --file data/journal_log.csv
    python scripts/import_legacy_journal.py --file data/journal_log.csv --dry-run
    python scripts/import_legacy_journal.py --file data/journal_log.csv --legacy-timezone UTC
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

from sqlmodel import Session, create_engine

# Ensure app package is importable when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.models.feedback import Feedback
from app.models.journal import JournalEntry
from app.repositories.feedback import FeedbackRepository
from app.repositories.journal import JournalRepository
from app.services.csv_import import (
    compute_legacy_hash,
    normalize_legacy_feedback,
    normalize_tags_and_labels,
    parse_legacy_timestamp,
)

logger = logging.getLogger("legacy_import")


def setup_cli_logging(verbose: bool = False) -> None:
    """Configure terminal logging."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)


class ImportSummary:
    """Tracks statistics for the CSV import process."""

    def __init__(self) -> None:
        self.total_rows: int = 0
        self.imported: int = 0
        self.duplicates: int = 0
        self.invalid: int = 0
        self.feedback_created: int = 0
        self.reasons: list[str] = []

    def record_invalid(self, row_idx: int, reason: str) -> None:
        self.invalid += 1
        msg = f"Row {row_idx}: {reason}"
        self.reasons.append(msg)
        logger.warning(msg)

    def print_report(self, dry_run: bool) -> None:
        mode = "DRY RUN (no database changes)" if dry_run else "LIVE MIGRATION"
        print("\n" + "=" * 60)
        print(f" legacy CSV Import Summary — {mode}")
        print("=" * 60)
        print(f" Total rows evaluated : {self.total_rows}")
        print(f" Successfully imported: {self.imported}")
        print(f" Duplicates skipped   : {self.duplicates}")
        print(f" Invalid rows rejected: {self.invalid}")
        print(f" Feedback records linked: {self.feedback_created}")
        print("=" * 60)

        if self.reasons:
            print("\nRejection details (sample up to 10):")
            for r in self.reasons[:10]:
                print(f" - {r}")
            if len(self.reasons) > 10:
                print(f" ... and {len(self.reasons) - 10} more.")
        print()


def run_csv_import(
    file_path: str,
    dry_run: bool = False,
    database_url: str | None = None,
    legacy_timezone: str = "UTC",
    verbose: bool = False,
) -> tuple[int, ImportSummary]:
    """Execute the CSV import process.

    Returns:
        Tuple of (exit_code, summary).
    """
    summary = ImportSummary()

    # 1. Validate file existence
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return 1, summary

    if not os.path.isfile(file_path):
        logger.error("Path is not a regular file: %s", file_path)
        return 1, summary

    # 2. Check for zero-byte file
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error("File is empty (0 bytes): %s", file_path)
        return 1, summary

    logger.info("Opening CSV file: %s (%d bytes)", file_path, file_size)
    logger.info("Using legacy timezone assumption: %s for naive timestamps", legacy_timezone)

    # 3. Read and validate CSV headers
    try:
        with open(file_path, encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.error("CSV file contains no headers: %s", file_path)
                return 1, summary

            headers = [h.strip() for h in reader.fieldnames if h]

            # Require at least 'text' or legacy text equivalent
            text_col = next((h for h in headers if h.lower() in ("text", "entry", "journal")), None)
            if not text_col:
                logger.error("Required 'text' column not found. Available columns: %s", headers)
                return 1, summary

            # Detect other standard legacy columns
            timestamp_col = next(
                (h for h in headers if h.lower() in ("timestamp", "created_at", "date")), None
            )
            emotion_col = next(
                (h for h in headers if h.lower() in ("emotion", "detected_emotion")), None
            )
            value_col = next(
                (h for h in headers if h.lower() in ("value_theme", "value", "theme")), None
            )
            response_col = next(
                (h for h in headers if h.lower() in ("ai_response", "response")), None
            )
            feedback_col = next(
                (h for h in headers if h.lower() in ("feedback", "feedback_type")), None
            )

            rows = list(reader)
    except Exception as exc:
        logger.error("Failed to read CSV file: %s", exc)
        return 1, summary

    if not rows:
        logger.warning("CSV file contains only headers (0 data rows).")
        return 0, summary

    summary.total_rows = len(rows)
    logger.info("Parsed %d data rows from CSV", summary.total_rows)

    # 4. Connect to database
    settings = get_settings()
    db_url = database_url or settings.database_url
    logger.info("Database target: %s", db_url)

    engine = create_engine(db_url)
    seen_hashes_in_batch: set[str] = set()

    with Session(engine) as session:
        journal_repo = JournalRepository(session)
        feedback_repo = FeedbackRepository(session)

        for idx, row in enumerate(rows, start=1):
            raw_text = row.get(text_col, "")
            if not raw_text or not raw_text.strip():
                summary.record_invalid(idx, "Empty journal text")
                continue

            clean_text = raw_text.strip()

            # Parse timestamp
            raw_ts = row.get(timestamp_col, "") if timestamp_col else ""
            if raw_ts:
                parsed_ts, ts_err = parse_legacy_timestamp(
                    raw_ts, legacy_timezone_name=legacy_timezone
                )
                if ts_err or not parsed_ts:
                    summary.record_invalid(idx, f"Invalid timestamp '{raw_ts}': {ts_err}")
                    continue
            else:
                from app.models.base import utcnow

                parsed_ts = utcnow()

            # Compute SHA-256 fingerprint for safe deduplication
            source_hash = compute_legacy_hash(clean_text, raw_ts)

            # Check for duplicate within the current file
            if source_hash in seen_hashes_in_batch:
                summary.duplicates += 1
                if verbose:
                    logger.debug("Row %d: Duplicate within current CSV file (skipped)", idx)
                continue

            seen_hashes_in_batch.add(source_hash)

            # Check for duplicate in database
            existing_entry = journal_repo.get_by_legacy_hash(source_hash)
            if existing_entry is not None:
                summary.duplicates += 1
                if verbose:
                    logger.debug(
                        "Row %d: Already exists in database (hash=%s...)", idx, source_hash[:8]
                    )
                continue

            # Parse optional fields
            raw_emotion = row.get(emotion_col, "").strip() if emotion_col else None
            raw_value = row.get(value_col, "").strip() if value_col else None
            raw_response = row.get(response_col, "").strip() if response_col else None
            raw_feedback = row.get(feedback_col, "").strip() if feedback_col else None

            emotions = normalize_tags_and_labels(raw_emotion) if raw_emotion else []
            values = normalize_tags_and_labels(raw_value) if raw_value else []

            top_emotion = emotions[0] if emotions else None
            top_value = values[0] if values else None

            entry = JournalEntry(
                text=clean_text,
                top_emotion=top_emotion,
                top_value=top_value,
                detected_emotions=emotions,
                detected_values=values,
                tags=[],
                ai_response=raw_response or None,
                feedback=raw_feedback or None,  # Legacy field
                legacy_source_hash=source_hash,
                created_at=parsed_ts,
                updated_at=parsed_ts,
            )

            if not dry_run:
                saved_entry = journal_repo.create(entry)

                # Canonical feedback creation if valid legacy feedback exists
                if raw_feedback:
                    canonical_type = normalize_legacy_feedback(raw_feedback)
                    if canonical_type:
                        feedback_record = Feedback(
                            journal_entry_id=saved_entry.id,
                            feedback_type=canonical_type,
                            comment=None,
                            created_at=parsed_ts,
                        )
                        feedback_repo.create(feedback_record)
                        summary.feedback_created += 1

            summary.imported += 1

    summary.print_report(dry_run)
    return 0, summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Import legacy CSV journal data into ConsciousAI Journal V2 database."
    )
    parser.add_argument(
        "--file",
        "-f",
        default="data/journal_log.csv",
        help="Path to the legacy CSV file (default: data/journal_log.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the import and display results without committing to database",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Custom SQLAlchemy database URL (defaults to DATABASE_URL from settings)",
    )
    parser.add_argument(
        "--legacy-timezone",
        default="UTC",
        help="Timezone for naive timestamps in the CSV (default: UTC)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed diagnostic logging",
    )

    args = parser.parse_args()
    setup_cli_logging(args.verbose)

    exit_code, _ = run_csv_import(
        file_path=args.file,
        dry_run=args.dry_run,
        database_url=args.database_url,
        legacy_timezone=args.legacy_timezone,
        verbose=args.verbose,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
