"""Unit tests for SQLModel database models."""

from __future__ import annotations

from datetime import timezone

from app.models.base import utcnow
from app.models.feedback import Feedback
from app.models.journal import JournalEntry
from app.models.memory import Memory
from app.models.settings import UserSettings


class TestBaseModels:
    """Tests for base utilities and timestamps."""

    def test_utcnow_is_timezone_aware(self):
        now = utcnow()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc


class TestJournalEntryModel:
    """Tests for JournalEntry model and normalization."""

    def test_journal_entry_defaults(self):
        entry = JournalEntry(text="I am reflecting on today.")
        assert entry.text == "I am reflecting on today."
        assert entry.mood_score is None
        assert entry.top_emotion is None
        assert entry.top_value is None
        assert entry.detected_emotions == []
        assert entry.detected_values == []
        assert entry.tags == []
        assert entry.ai_response is None
        assert entry.feedback is None
        assert entry.legacy_source_hash is None
        assert entry.is_private is False
        assert entry.is_deleted is False
        assert entry.deleted_at is None
        assert entry.created_at.tzinfo is not None

    def test_string_list_normalization(self):
        entry = JournalEntry(
            text="Reflecting",
            detected_emotions=["  calm  ", "happy", "calm", "", "  "],
            detected_values=["honesty", "honesty", "growth"],
            tags=[" daily ", "thoughts", "daily"],
        )
        assert entry.detected_emotions == ["calm", "happy"]
        assert entry.detected_values == ["honesty", "growth"]
        assert entry.tags == ["daily", "thoughts"]

    def test_string_list_from_comma_separated_string(self):
        entry = JournalEntry(
            text="Testing comma string",
            detected_emotions="happy, calm, happy,  peaceful",  # type: ignore[arg-type]
        )
        assert entry.detected_emotions == ["happy", "calm", "peaceful"]

    def test_none_list_normalizes_to_empty_list(self):
        entry = JournalEntry(
            text="Testing None lists",
            detected_emotions=None,  # type: ignore[arg-type]
            tags=None,  # type: ignore[arg-type]
        )
        assert entry.detected_emotions == []
        assert entry.tags == []


class TestMemoryModel:
    """Tests for Memory model (Correction 3: default is_approved=False)."""

    def test_memory_defaults_to_unapproved(self):
        mem = Memory(content="User feels anxious before presentations")
        assert mem.content == "User feels anxious before presentations"
        assert mem.is_approved is False
        assert mem.memory_type == "reflection"
        assert mem.importance == 0.5
        assert mem.source_entry_id is None
        assert mem.created_at.tzinfo is not None


class TestFeedbackModel:
    """Tests for Feedback model (Correction 2: canonical feedback)."""

    def test_feedback_attributes(self):
        fb = Feedback(journal_entry_id=1, feedback_type="Insightful", comment="Very helpful")
        assert fb.journal_entry_id == 1
        assert fb.feedback_type == "Insightful"
        assert fb.comment == "Very helpful"
        assert fb.created_at.tzinfo is not None


class TestUserSettingsModel:
    """Tests for UserSettings model."""

    def test_user_settings_defaults(self):
        settings = UserSettings()
        assert settings.id == 1
        assert settings.persona == "Supportive"
        assert settings.memory_enabled is True
        assert settings.analytics_enabled is True
        assert settings.preferred_response_length == "medium"
