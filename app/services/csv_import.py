"""Utilities for parsing, validating, and fingerprinting legacy CSV journal data."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def compute_legacy_hash(text: str, timestamp_str: str, source_id: str = "") -> str:
    """Compute a deterministic SHA-256 fingerprint for a legacy journal entry.

    Normalizes whitespace and text to ensure two imports of the same content
    produce identical hashes regardless of minor whitespace artifacts.

    Args:
        text: Raw journal entry text.
        timestamp_str: Original timestamp string from the CSV.
        source_id: Optional source file or record identifier.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    normalized_text = " ".join((text or "").split()).strip()
    normalized_ts = (timestamp_str or "").strip()
    payload = f"{normalized_ts}|{normalized_text}|{source_id.strip()}".encode()
    return hashlib.sha256(payload).hexdigest()


COMMON_DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
]


def parse_legacy_timestamp(
    timestamp_str: str,
    legacy_timezone_name: str = "UTC",
) -> tuple[datetime | None, str | None]:
    """Parse a legacy timestamp string into a timezone-aware UTC datetime.

    Args:
        timestamp_str: The timestamp string from the CSV.
        legacy_timezone_name: Assumed IANA timezone for naive timestamps (default UTC).

    Returns:
        Tuple of (parsed_utc_datetime, error_message).
    """
    if not timestamp_str or not timestamp_str.strip():
        return None, "Timestamp string is empty"

    raw = timestamp_str.strip()

    # Resolve configured legacy timezone
    try:
        assumed_tz = ZoneInfo(legacy_timezone_name)
    except ZoneInfoNotFoundError:
        assumed_tz = timezone.utc

    # 1. Try standard fromisoformat first (handles Z, offsets, standard ISO)
    iso_candidate = raw
    if iso_candidate.endswith("Z") or iso_candidate.endswith("z"):
        iso_candidate = iso_candidate[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=assumed_tz)
        return dt.astimezone(timezone.utc), None
    except ValueError:
        pass

    # 2. Try common strptime formats
    for fmt in COMMON_DATETIME_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=assumed_tz)
            return dt.astimezone(timezone.utc), None
        except ValueError:
            continue

    return None, f"Unrecognized date format '{raw}'"


SUPPORTED_FEEDBACK_TYPES = {
    "insightful": "Insightful",
    "helpful": "Helpful",
    "irrelevant": "Irrelevant",
    "not helpful": "Not Helpful",
    "offensive": "Offensive",
    "other": "Other",
}


def normalize_legacy_feedback(feedback_str: str | None) -> str | None:
    """Map legacy feedback strings safely to canonical feedback types.

    Returns canonical string or None if blank / unmapped.
    """
    if not feedback_str:
        return None
    cleaned = feedback_str.strip().lower()
    return SUPPORTED_FEEDBACK_TYPES.get(
        cleaned, feedback_str.strip() if feedback_str.strip() else None
    )


def normalize_tags_and_labels(val: Any) -> list[str]:
    """Extract list of clean strings from legacy single or comma-separated value."""
    if val is None:
        return []
    items = val if isinstance(val, (list, tuple, set)) else str(val).split(",")

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
