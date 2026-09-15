"""Journal repository for database access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func
from sqlmodel import col, select

if TYPE_CHECKING:
    from datetime import datetime

from app.models.base import utcnow
from app.models.journal import JournalEntry
from app.repositories.base import BaseRepository

IMMUTABLE_FIELDS = {"id", "created_at", "legacy_source_hash", "user_id"}


class JournalRepository(BaseRepository[JournalEntry]):
    """Repository handling persistence operations for JournalEntry records."""

    def create(self, entry: JournalEntry) -> JournalEntry:
        """Persist a new journal entry.

        Args:
            entry: JournalEntry instance to create.

        Returns:
            The persisted JournalEntry with populated ID and timestamps.
        """
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry

    def get_by_id(
        self,
        entry_id: int,
        include_deleted: bool = False,
        user_id: str | None = None,
    ) -> JournalEntry | None:
        """Fetch a journal entry by primary key with optional user isolation.

        Args:
            entry_id: Primary key ID.
            include_deleted: If True, returns even soft-deleted entries.
            user_id: Optional user identifier for tenant isolation.

        Returns:
            JournalEntry if found and authorized, None otherwise.
        """
        statement = select(JournalEntry).where(JournalEntry.id == entry_id)
        if not include_deleted:
            statement = statement.where(JournalEntry.is_deleted.is_(False))
        if user_id is not None:
            statement = statement.where(JournalEntry.user_id == user_id)
        return self.session.exec(statement).first()

    def get_by_legacy_hash(self, hash_str: str) -> JournalEntry | None:
        """Retrieve an entry by its legacy source hash (fingerprint).

        Args:
            hash_str: SHA-256 fingerprint string.

        Returns:
            Matching JournalEntry or None.
        """
        if not hash_str:
            return None
        statement = select(JournalEntry).where(JournalEntry.legacy_source_hash == hash_str)
        return self.session.exec(statement).first()

    def get_by_timestamp(self, timestamp: datetime) -> list[JournalEntry]:
        """Diagnostic lookup for entries matching an exact timestamp.

        NOTE: This is diagnostic only. Deduplication must use legacy_source_hash.

        Args:
            timestamp: The datetime to query.

        Returns:
            List of matching JournalEntry records.
        """
        statement = select(JournalEntry).where(JournalEntry.created_at == timestamp)
        return list(self.session.exec(statement).all())

    def list(
        self,
        skip: int = 0,
        limit: int = 50,
        emotion: str | None = None,
        value_theme: str | None = None,
        tag: str | None = None,
        is_private: bool | None = None,
        include_deleted: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        search: str | None = None,
        user_id: str | None = None,
    ) -> list[JournalEntry]:
        """Query journal entries with filtering, user isolation, and deterministic pagination.

        Ordering is strictly deterministic: created_at DESC, id DESC.

        Args:
            skip: Offset number of records.
            limit: Maximum records to return.
            emotion: Filter by top_emotion.
            value_theme: Filter by top_value.
            tag: Filter by tag (contains check).
            is_private: Filter by privacy flag.
            include_deleted: Whether to include soft-deleted records.
            start_date: Earliest created_at (inclusive).
            end_date: Latest created_at (inclusive).
            search: Case-insensitive substring match in text.
            user_id: Optional user identifier for tenant isolation.

        Returns:
            Ordered list of matching JournalEntry records.
        """
        statement = select(JournalEntry)

        if not include_deleted:
            statement = statement.where(JournalEntry.is_deleted.is_(False))

        if user_id is not None:
            statement = statement.where(JournalEntry.user_id == user_id)

        if emotion:
            statement = statement.where(JournalEntry.top_emotion == emotion)

        if value_theme:
            statement = statement.where(JournalEntry.top_value == value_theme)

        if is_private is not None:
            statement = statement.where(JournalEntry.is_private.is_(is_private))

        if start_date:
            statement = statement.where(JournalEntry.created_at >= start_date)

        if end_date:
            statement = statement.where(JournalEntry.created_at <= end_date)

        if search:
            statement = statement.where(col(JournalEntry.text).contains(search))

        # Deterministic ordering: newest first, tie-break by ID
        statement = statement.order_by(JournalEntry.created_at.desc(), JournalEntry.id.desc())
        statement = statement.offset(skip).limit(limit)

        results = list(self.session.exec(statement).all())

        # Tag filter in Python to support all SQL dialects seamlessly
        if tag:
            results = [e for e in results if tag in e.tags]

        return results

    def count(
        self,
        emotion: str | None = None,
        value_theme: str | None = None,
        tag: str | None = None,
        is_private: bool | None = None,
        include_deleted: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        search: str | None = None,
        user_id: str | None = None,
    ) -> int:
        """Count entries matching filters."""
        if tag:
            # When tag filter is used, count matches from list query
            return len(
                self.list(
                    skip=0,
                    limit=100000,
                    emotion=emotion,
                    value_theme=value_theme,
                    tag=tag,
                    is_private=is_private,
                    include_deleted=include_deleted,
                    start_date=start_date,
                    end_date=end_date,
                    search=search,
                    user_id=user_id,
                )
            )

        statement = select(func.count(JournalEntry.id))

        if not include_deleted:
            statement = statement.where(JournalEntry.is_deleted.is_(False))

        if user_id is not None:
            statement = statement.where(JournalEntry.user_id == user_id)

        if emotion:
            statement = statement.where(JournalEntry.top_emotion == emotion)

        if value_theme:
            statement = statement.where(JournalEntry.top_value == value_theme)

        if is_private is not None:
            statement = statement.where(JournalEntry.is_private.is_(is_private))

        if start_date:
            statement = statement.where(JournalEntry.created_at >= start_date)

        if end_date:
            statement = statement.where(JournalEntry.created_at <= end_date)

        if search:
            statement = statement.where(col(JournalEntry.text).contains(search))

        return self.session.exec(statement).one() or 0

    def update(
        self,
        entry_id: int,
        user_id: str | None = None,
        **updates: Any,
    ) -> JournalEntry | None:
        """Update fields on a journal entry with safety validation.

        Args:
            entry_id: ID of the entry to update.
            user_id: Optional user identifier for tenant isolation.
            **updates: Keyword arguments of fields to update.

        Raises:
            ValueError: If attempting to modify immutable/audit fields or unknown fields.

        Returns:
            Updated JournalEntry or None if not found/unauthorized.
        """
        entry = self.get_by_id(entry_id, include_deleted=True, user_id=user_id)
        if not entry:
            return None

        for field, value in updates.items():
            if field in IMMUTABLE_FIELDS:
                msg = f"Cannot update immutable field '{field}'"
                raise ValueError(msg)
            if not hasattr(entry, field):
                msg = f"Unknown field '{field}' on JournalEntry"
                raise ValueError(msg)
            setattr(entry, field, value)

        entry.updated_at = utcnow()
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry

    def soft_delete(self, entry_id: int, user_id: str | None = None) -> bool:
        """Mark an entry as deleted without removing from database.

        Args:
            entry_id: Primary key ID.
            user_id: Optional user identifier for tenant isolation.

        Returns:
            True if entry was found, authorized, and marked deleted; False otherwise.
        """
        entry = self.get_by_id(entry_id, include_deleted=False, user_id=user_id)
        if not entry:
            return False

        entry.is_deleted = True
        entry.deleted_at = utcnow()
        entry.updated_at = utcnow()
        self.session.add(entry)
        self.session.commit()
        return True

    def restore(self, entry_id: int, user_id: str | None = None) -> JournalEntry | None:
        """Restore a soft-deleted entry.

        Args:
            entry_id: Primary key ID.
            user_id: Optional user identifier for tenant isolation.

        Returns:
            Restored JournalEntry or None if not found/unauthorized.
        """
        entry = self.get_by_id(entry_id, include_deleted=True, user_id=user_id)
        if not entry or not entry.is_deleted:
            return None

        entry.is_deleted = False
        entry.deleted_at = None
        entry.updated_at = utcnow()
        self.session.add(entry)
        self.session.commit()
        self.session.refresh(entry)
        return entry

    def hard_delete(self, entry_id: int, user_id: str | None = None) -> bool:
        """Permanently delete an entry and all related child records.

        Args:
            entry_id: Primary key ID.
            user_id: Optional user identifier for tenant isolation.

        Returns:
            True if entry existed, was authorized, and was removed; False otherwise.
        """
        entry = self.get_by_id(entry_id, include_deleted=True, user_id=user_id)
        if not entry:
            return False

        self.session.delete(entry)
        self.session.commit()
        return True
