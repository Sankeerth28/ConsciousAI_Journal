"""Unit tests for repository data access layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sqlmodel import Session

from app.models.feedback import Feedback
from app.models.journal import JournalEntry
from app.models.memory import Memory
from app.repositories.feedback import FeedbackRepository
from app.repositories.journal import JournalRepository
from app.repositories.memory import MemoryRepository
from app.repositories.settings import UserSettingsRepository


class TestJournalRepository:
    """Tests for JournalRepository operations."""

    def test_create_and_get_by_id(self, session: Session):
        repo = JournalRepository(session)
        entry = JournalEntry(text="Today was a breakthrough day.", top_emotion="hopeful")
        created = repo.create(entry)

        assert created.id is not None
        fetched = repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.text == "Today was a breakthrough day."
        assert fetched.top_emotion == "hopeful"

    def test_get_by_legacy_hash(self, session: Session):
        repo = JournalRepository(session)
        entry = JournalEntry(text="Entry with fingerprint", legacy_source_hash="sha256_hash_abc123")
        repo.create(entry)

        found = repo.get_by_legacy_hash("sha256_hash_abc123")
        assert found is not None
        assert found.text == "Entry with fingerprint"

        assert repo.get_by_legacy_hash("nonexistent") is None

    def test_diagnostic_get_by_timestamp(self, session: Session):
        repo = JournalRepository(session)
        ts = datetime(2025, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
        e1 = JournalEntry(text="Entry 1", created_at=ts)
        e2 = JournalEntry(text="Entry 2", created_at=ts)
        repo.create(e1)
        repo.create(e2)

        matches = repo.get_by_timestamp(ts)
        assert len(matches) == 2

    def test_deterministic_pagination_ordering(self, session: Session):
        repo = JournalRepository(session)
        base = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        # Create entries with staggered timestamps
        e1 = repo.create(JournalEntry(text="First", created_at=base))
        e2 = repo.create(JournalEntry(text="Second", created_at=base + timedelta(hours=1)))
        e3 = repo.create(JournalEntry(text="Third", created_at=base + timedelta(hours=2)))

        ordered = repo.list()
        # Newest first
        assert [e.id for e in ordered] == [e3.id, e2.id, e1.id]

    def test_filtering_by_emotion_and_tag(self, session: Session):
        repo = JournalRepository(session)
        repo.create(JournalEntry(text="Happy entry", top_emotion="happy", tags=["joy"]))
        repo.create(JournalEntry(text="Calm entry", top_emotion="calm", tags=["meditation"]))
        repo.create(JournalEntry(text="Another happy", top_emotion="happy", tags=["reflection"]))

        happy_entries = repo.list(emotion="happy")
        assert len(happy_entries) == 2

        joy_entries = repo.list(tag="joy")
        assert len(joy_entries) == 1
        assert joy_entries[0].text == "Happy entry"

    def test_search_filter(self, session: Session):
        repo = JournalRepository(session)
        repo.create(JournalEntry(text="Meeting about the new project roadmap"))
        repo.create(JournalEntry(text="Went for a quiet morning walk"))

        results = repo.list(search="roadmap")
        assert len(results) == 1
        assert "roadmap" in results[0].text

    def test_soft_delete_and_restore(self, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="To be deleted"))

        # Soft delete
        assert repo.soft_delete(entry.id) is True

        # Excluded by default
        assert repo.get_by_id(entry.id) is None
        assert repo.list() == []

        # Available when include_deleted=True
        deleted_entry = repo.get_by_id(entry.id, include_deleted=True)
        assert deleted_entry is not None
        assert deleted_entry.is_deleted is True
        assert deleted_entry.deleted_at is not None

        # Restore
        restored = repo.restore(entry.id)
        assert restored is not None
        assert restored.is_deleted is False
        assert repo.get_by_id(entry.id) is not None

    def test_hard_delete(self, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Hard delete me"))

        assert repo.hard_delete(entry.id) is True
        assert repo.get_by_id(entry.id, include_deleted=True) is None

    def test_update_validates_immutable_and_unknown_fields(self, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Original text"))

        # Cannot modify immutable fields
        with pytest.raises(ValueError, match="Cannot update immutable field 'created_at'"):
            repo.update(entry.id, created_at=datetime.now(timezone.utc))

        with pytest.raises(ValueError, match="Cannot update immutable field 'id'"):
            repo.update(entry.id, id=999)

        with pytest.raises(ValueError, match="Unknown field 'nonexistent_col'"):
            repo.update(entry.id, nonexistent_col="bad")

        # Valid update works
        updated = repo.update(entry.id, text="Updated text", mood_score=8.5)
        assert updated is not None
        assert updated.text == "Updated text"
        assert updated.mood_score == 8.5


class TestMemoryRepository:
    """Tests for MemoryRepository operations."""

    def test_create_and_approve_memory(self, session: Session):
        repo = MemoryRepository(session)
        mem = repo.create(Memory(content="Prefers working in silence"))

        # Default is unapproved (Correction 3)
        assert mem.is_approved is False

        # Approve
        approved = repo.approve(mem.id)
        assert approved is not None
        assert approved.is_approved is True

    def test_list_filtering_by_approval_and_type(self, session: Session):
        repo = MemoryRepository(session)
        m1 = repo.create(Memory(content="Unapproved fact", memory_type="fact", is_approved=False))
        m2 = repo.create(
            Memory(content="Approved reflection", memory_type="reflection", is_approved=True)
        )

        unapproved = repo.list(is_approved=False)
        assert len(unapproved) == 1
        assert unapproved[0].id == m1.id

        approved_reflections = repo.list(is_approved=True, memory_type="reflection")
        assert len(approved_reflections) == 1
        assert approved_reflections[0].id == m2.id


class TestFeedbackRepository:
    """Tests for canonical FeedbackRepository operations."""

    def test_create_and_list_by_entry(self, session: Session):
        j_repo = JournalRepository(session)
        fb_repo = FeedbackRepository(session)

        entry = j_repo.create(JournalEntry(text="Entry for feedback"))
        fb = fb_repo.create(
            Feedback(journal_entry_id=entry.id, feedback_type="Insightful", comment="Spot on")
        )

        assert fb.id is not None
        feedback_list = fb_repo.list_by_entry(entry.id)
        assert len(feedback_list) == 1
        assert feedback_list[0].feedback_type == "Insightful"


class TestUserSettingsRepository:
    """Tests for UserSettingsRepository operations."""

    def test_get_or_create_and_update(self, session: Session):
        repo = UserSettingsRepository(session)
        settings = repo.get_or_create(1)
        assert settings.persona == "Supportive"

        updated = repo.update(1, persona="Coach", preferred_response_length="long")
        assert updated.persona == "Coach"
        assert updated.preferred_response_length == "long"


class TestCascadeBehavior:
    """Test foreign key relationships and cascade rules."""

    def test_journal_entry_deletion_cascades_feedback(self, session: Session):
        j_repo = JournalRepository(session)
        fb_repo = FeedbackRepository(session)
        mem_repo = MemoryRepository(session)

        entry = j_repo.create(JournalEntry(text="Entry to cascade"))
        fb = fb_repo.create(Feedback(journal_entry_id=entry.id, feedback_type="Helpful"))
        mem = mem_repo.create(Memory(content="Memory linked to entry", source_entry_id=entry.id))

        # Delete entry
        j_repo.hard_delete(entry.id)

        # Feedback should be cascaded and removed
        assert fb_repo.get_by_id(fb.id) is None

        # Memory source_entry_id should be set null or entry detached
        refreshed_mem = mem_repo.get_by_id(mem.id)
        assert refreshed_mem is not None
        assert refreshed_mem.source_entry_id is None
