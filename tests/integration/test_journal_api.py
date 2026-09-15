"""Comprehensive integration tests for Journal REST API, feedback, export, and user isolation."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.ai.pipeline import JournalReflectionPipeline
from app.ai.providers.mock import (
    MockEmotionClassifier,
    MockLLMProvider,
    MockValueClassifier,
)
from app.ai.providers.vector_store import LocalVectorStore
from app.api.deps import get_db, get_pipeline
from app.main import app
from app.models.journal import JournalEntry
from app.repositories.journal import JournalRepository
from app.services.memory_service import MemoryRetrievalService
from tests.unit.test_memory_service import DeterministicEmbeddingProvider


@pytest.fixture(name="api_client")
def api_client_fixture(session: Session):
    """Provide a TestClient with database session overridden by test session."""

    def _override_get_db():
        yield session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestJournalCreationAndSafety:
    """Tests covering journal entry creation, metadata return, and safety interception."""

    def test_create_journal_entry_happy_path(self, api_client: TestClient, session: Session):
        """Safe journal input is persisted, classified, reflected, and returned with HTTP 201."""
        payload = {
            "text": "Today I went for a morning run through the woods and felt deeply alive.",
            "mood_score": 8.5,
            "tags": ["morning", "running", "nature"],
            "persona": "Supportive",
            "region": "GLOBAL",
            "is_private": False,
        }

        response = api_client.post(
            "/api/v1/journals",
            json=payload,
            headers={"X-User-ID": "alice"},
        )

        assert response.status_code == 201
        data = response.json()

        # 1. Entry persisted
        entry = data["entry"]
        assert entry is not None
        assert entry["id"] > 0
        assert entry["user_id"] == "alice"
        assert entry["text"] == payload["text"]
        assert entry["mood_score"] == 8.5
        assert set(entry["tags"]) == {"morning", "running", "nature"}
        assert entry["top_emotion"] is not None
        assert entry["top_value"] is not None
        assert entry["ai_response"] is not None
        assert entry["is_private"] is False

        # 2. Reflection and safety metadata returned
        reflection = data["reflection"]
        assert reflection["safety_flag"] is False
        assert reflection["persona"] == "Supportive"
        assert len(reflection["response"]) > 0

        input_safety = data["input_safety"]
        assert input_safety["is_safe"] is True

        # 3. Database verification
        db_entry = session.get(JournalEntry, entry["id"])
        assert db_entry is not None
        assert db_entry.user_id == "alice"
        assert db_entry.text == payload["text"]

    def test_unsafe_input_causes_zero_persistence_and_zero_downstream_calls(
        self,
        api_client: TestClient,
        session: Session,
    ):
        """CRITICAL NON-NEGOTIABLE SAFETY GUARANTEE:

        Unsafe input must halt at Step 2 of the pipeline:
        - HTTP 200 returned with crisis envelope (so client can show help)
        - entry is None in response
        - ZERO journal entries persisted to database
        - ZERO calls to emotion, value, or reflection models
        - Crisis helpline resources returned
        """
        spy_emotion = MagicMock(spec=MockEmotionClassifier)
        spy_value = MagicMock(spec=MockValueClassifier)
        spy_reflection_llm = MagicMock(spec=MockLLMProvider)

        custom_pipeline = JournalReflectionPipeline(
            emotion_service=spy_emotion,
            value_service=spy_value,
        )

        app.dependency_overrides[get_pipeline] = lambda: custom_pipeline

        unsafe_text = "I feel so hopeless and I want to commit suicide right now."
        payload = {
            "text": unsafe_text,
            "persona": "Supportive",
            "region": "US",
        }

        response = api_client.post(
            "/api/v1/journals",
            json=payload,
            headers={"X-User-ID": "alice"},
        )

        assert response.status_code == 200
        data = response.json()

        # Entry must NOT be created
        assert data["entry"] is None

        # Safety flags triggered
        assert data["input_safety"]["is_safe"] is False
        assert data["input_safety"]["reason"] == "crisis_detected"
        assert "988" in data["input_safety"]["crisis_response"]
        assert data["reflection"]["safety_flag"] is True

        # Downstream services were NOT called
        assert spy_emotion.classify.call_count == 0
        assert spy_value.classify.call_count == 0
        assert spy_reflection_llm.generate.call_count == 0

        # Database is completely clean
        db_records = session.exec(select(JournalEntry)).all()
        assert len(db_records) == 0

    def test_create_journal_entry_validation_errors(self, api_client: TestClient):
        """Payload validation correctly rejects empty text, oversized text, and forbidden fields."""
        # Empty text
        r_empty = api_client.post("/api/v1/journals", json={"text": ""})
        assert r_empty.status_code == 422

        # Invalid mood score (> 10)
        r_mood = api_client.post(
            "/api/v1/journals",
            json={"text": "A good day", "mood_score": 15.0},
        )
        assert r_mood.status_code == 422

        # Extra forbidden field (extra='forbid')
        r_extra = api_client.post(
            "/api/v1/journals",
            json={"text": "A good day", "malicious_field": "hacked"},
        )
        assert r_extra.status_code == 422


class TestUserIsolationAcrossAllEndpoints:
    """Rigorous tests proving that user data is isolated across all operations."""

    def test_user_isolation_for_read_update_delete_restore(
        self,
        api_client: TestClient,
        session: Session,
    ):
        """User Bob must NEVER access, update, delete, or restore Alice's journal entry."""
        repo = JournalRepository(session)
        alice_entry = repo.create(
            JournalEntry(
                text="Alice's private secret entry",
                user_id="alice",
                is_private=True,
            )
        )

        # 1. Bob cannot read Alice's entry (404)
        r_get = api_client.get(
            f"/api/v1/journals/{alice_entry.id}",
            headers={"X-User-ID": "bob"},
        )
        assert r_get.status_code == 404

        # 2. Bob cannot update Alice's entry (404)
        r_patch = api_client.patch(
            f"/api/v1/journals/{alice_entry.id}",
            json={"mood_score": 5.0},
            headers={"X-User-ID": "bob"},
        )
        assert r_patch.status_code == 404

        # 3. Bob cannot delete Alice's entry (404)
        r_delete = api_client.delete(
            f"/api/v1/journals/{alice_entry.id}",
            headers={"X-User-ID": "bob"},
        )
        assert r_delete.status_code == 404

        # 4. Bob cannot restore Alice's entry
        r_restore = api_client.post(
            f"/api/v1/journals/{alice_entry.id}/restore",
            headers={"X-User-ID": "bob"},
        )
        assert r_restore.status_code == 404

        # 5. Bob's list does NOT include Alice's entry
        r_list = api_client.get(
            "/api/v1/journals",
            headers={"X-User-ID": "bob"},
        )
        assert r_list.status_code == 200
        assert r_list.json()["total"] == 0
        assert len(r_list.json()["items"]) == 0

        # 6. Alice CAN read her own entry
        r_alice_get = api_client.get(
            f"/api/v1/journals/{alice_entry.id}",
            headers={"X-User-ID": "alice"},
        )
        assert r_alice_get.status_code == 200
        assert r_alice_get.json()["text"] == "Alice's private secret entry"


class TestJournalListingAndFiltering:
    """Tests for pagination, deterministic ordering, and attribute filters."""

    @pytest.fixture(autouse=True)
    def seed_entries(self, session: Session):
        repo = JournalRepository(session)
        # Create chronological entries for user 'charlie'
        repo.create(
            JournalEntry(
                text="Morning coffee and journaling",
                top_emotion="calm",
                top_value="mindfulness",
                tags=["morning", "routine"],
                is_private=False,
                user_id="charlie",
                created_at=datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc),
            )
        )
        repo.create(
            JournalEntry(
                text="Afternoon work breakthrough on algorithm",
                top_emotion="happy",
                top_value="growth",
                tags=["work", "coding"],
                is_private=True,
                user_id="charlie",
                created_at=datetime(2026, 9, 2, 14, 0, tzinfo=timezone.utc),
            )
        )
        repo.create(
            JournalEntry(
                text="Evening meditation under starry sky",
                top_emotion="calm",
                top_value="mindfulness",
                tags=["night", "meditation"],
                is_private=False,
                user_id="charlie",
                created_at=datetime(2026, 9, 3, 21, 0, tzinfo=timezone.utc),
            )
        )

    def test_deterministic_ordering_newest_first(self, api_client: TestClient):
        response = api_client.get(
            "/api/v1/journals",
            headers={"X-User-ID": "charlie"},
        )
        assert response.status_code == 200
        items = response.json()["items"]
        assert len(items) == 3
        # Newest (Sept 3) first, then Sept 2, then Sept 1
        assert "Evening meditation" in items[0]["text"]
        assert "Afternoon work" in items[1]["text"]
        assert "Morning coffee" in items[2]["text"]

    def test_pagination_skip_limit_has_more(self, api_client: TestClient):
        # Page 1 (limit 2)
        r1 = api_client.get(
            "/api/v1/journals?skip=0&limit=2",
            headers={"X-User-ID": "charlie"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["total"] == 3
        assert len(d1["items"]) == 2
        assert d1["has_more"] is True

        # Page 2 (skip 2, limit 2)
        r2 = api_client.get(
            "/api/v1/journals?skip=2&limit=2",
            headers={"X-User-ID": "charlie"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert len(d2["items"]) == 1
        assert d2["has_more"] is False

    def test_filter_by_emotion_and_value(self, api_client: TestClient):
        r_calm = api_client.get(
            "/api/v1/journals?emotion=calm",
            headers={"X-User-ID": "charlie"},
        )
        assert r_calm.status_code == 200
        assert r_calm.json()["total"] == 2

        r_growth = api_client.get(
            "/api/v1/journals?value_theme=growth",
            headers={"X-User-ID": "charlie"},
        )
        assert r_growth.status_code == 200
        assert r_growth.json()["total"] == 1
        assert "breakthrough" in r_growth.json()["items"][0]["text"]

    def test_filter_by_tag(self, api_client: TestClient):
        r_tag = api_client.get(
            "/api/v1/journals?tag=coding",
            headers={"X-User-ID": "charlie"},
        )
        assert r_tag.status_code == 200
        assert r_tag.json()["total"] == 1
        assert "algorithm" in r_tag.json()["items"][0]["text"]

    def test_filter_by_privacy(self, api_client: TestClient):
        r_priv = api_client.get(
            "/api/v1/journals?is_private=true",
            headers={"X-User-ID": "charlie"},
        )
        assert r_priv.status_code == 200
        assert r_priv.json()["total"] == 1

        r_pub = api_client.get(
            "/api/v1/journals?is_private=false",
            headers={"X-User-ID": "charlie"},
        )
        assert r_pub.status_code == 200
        assert r_pub.json()["total"] == 2

    def test_filter_by_search_query(self, api_client: TestClient):
        r_search = api_client.get(
            "/api/v1/journals?search=meditation",
            headers={"X-User-ID": "charlie"},
        )
        assert r_search.status_code == 200
        assert r_search.json()["total"] == 1
        assert "Evening meditation" in r_search.json()["items"][0]["text"]

    def test_filter_by_date_range(self, api_client: TestClient):
        r_date = api_client.get(
            "/api/v1/journals?start_date=2026-09-02T00:00:00Z&end_date=2026-09-02T23:59:59Z",
            headers={"X-User-ID": "charlie"},
        )
        assert r_date.status_code == 200
        assert r_date.json()["total"] == 1
        assert "Afternoon work" in r_date.json()["items"][0]["text"]


class TestJournalUpdateDeleteRestore:
    """Tests for updating mutable fields, rejecting immutable fields, and soft-delete/restore."""

    def test_partial_update_mutable_fields(self, api_client: TestClient, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(
            JournalEntry(
                text="Original entry",
                user_id="dave",
                mood_score=6.0,
                tags=["draft"],
            )
        )

        r_patch = api_client.patch(
            f"/api/v1/journals/{entry.id}",
            json={"mood_score": 7.5, "tags": ["final", "polished"], "is_private": True},
            headers={"X-User-ID": "dave"},
        )
        assert r_patch.status_code == 200
        data = r_patch.json()
        assert data["mood_score"] == 7.5
        assert set(data["tags"]) == {"final", "polished"}
        assert data["is_private"] is True
        assert data["text"] == "Original entry"

    def test_update_rejects_immutable_fields(self, api_client: TestClient, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Base text", user_id="dave"))

        # Extra forbidden field on schema
        r_user_id = api_client.patch(
            f"/api/v1/journals/{entry.id}",
            json={"user_id": "attacker"},
            headers={"X-User-ID": "dave"},
        )
        assert r_user_id.status_code == 422

        r_text = api_client.patch(
            f"/api/v1/journals/{entry.id}",
            json={"text": "tampered text"},
            headers={"X-User-ID": "dave"},
        )
        assert r_text.status_code == 422

    def test_soft_delete_and_restore(self, api_client: TestClient, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Entry to delete", user_id="emma"))

        # 1. Soft-delete
        r_del = api_client.delete(
            f"/api/v1/journals/{entry.id}",
            headers={"X-User-ID": "emma"},
        )
        assert r_del.status_code == 200
        assert r_del.json()["permanent"] is False

        # 2. Excluded from normal list
        r_list = api_client.get("/api/v1/journals", headers={"X-User-ID": "emma"})
        assert r_list.json()["total"] == 0

        # 3. Direct GET returns 404
        r_get = api_client.get(f"/api/v1/journals/{entry.id}", headers={"X-User-ID": "emma"})
        assert r_get.status_code == 404

        # 4. Restore
        r_restore = api_client.post(
            f"/api/v1/journals/{entry.id}/restore",
            headers={"X-User-ID": "emma"},
        )
        assert r_restore.status_code == 200
        assert r_restore.json()["id"] == entry.id

        # 5. Once restored, appears back in list
        r_list_after = api_client.get("/api/v1/journals", headers={"X-User-ID": "emma"})
        assert r_list_after.json()["total"] == 1

    def test_permanent_hard_delete(self, api_client: TestClient, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Permanently delete me", user_id="frank"))

        r_del = api_client.delete(
            f"/api/v1/journals/{entry.id}?permanent=true",
            headers={"X-User-ID": "frank"},
        )
        assert r_del.status_code == 200
        assert r_del.json()["permanent"] is True

        # Completely removed from DB
        assert session.get(JournalEntry, entry.id) is None


class TestFeedbackEndpoints:
    """Tests for attaching and listing feedback for journal entries."""

    def test_create_and_list_feedback(self, api_client: TestClient, session: Session):
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Reflecting on feedback", user_id="grace"))

        # 1. Attach feedback
        r_post = api_client.post(
            f"/api/v1/journals/{entry.id}/feedback",
            json={"feedback_type": "Insightful", "comment": "This question really made me pause."},
            headers={"X-User-ID": "grace"},
        )
        assert r_post.status_code == 201
        data = r_post.json()
        assert data["feedback_type"] == "Insightful"
        assert data["comment"] == "This question really made me pause."
        assert data["journal_entry_id"] == entry.id

        # 2. List feedback for entry
        r_get = api_client.get(
            f"/api/v1/journals/{entry.id}/feedback",
            headers={"X-User-ID": "grace"},
        )
        assert r_get.status_code == 200
        feedbacks = r_get.json()
        assert len(feedbacks) == 1
        assert feedbacks[0]["feedback_type"] == "Insightful"

    def test_feedback_user_isolation(self, api_client: TestClient, session: Session):
        """User cannot attach or view feedback on another user's journal entry."""
        repo = JournalRepository(session)
        entry = repo.create(JournalEntry(text="Grace's private reflection", user_id="grace"))

        # Attacker 'heidi' tries to post feedback
        r_post = api_client.post(
            f"/api/v1/journals/{entry.id}/feedback",
            json={"feedback_type": "Irrelevant"},
            headers={"X-User-ID": "heidi"},
        )
        assert r_post.status_code == 404

        # Heidi tries to view feedback
        r_get = api_client.get(
            f"/api/v1/journals/{entry.id}/feedback",
            headers={"X-User-ID": "heidi"},
        )
        assert r_get.status_code == 404


class TestExportEndpoints:
    """Tests for CSV and JSON data export."""

    @pytest.fixture(autouse=True)
    def seed_export_data(self, session: Session):
        repo = JournalRepository(session)
        repo.create(
            JournalEntry(
                text="Export entry one",
                user_id="export_user",
                mood_score=9.0,
                tags=["joy", "celebration"],
                top_emotion="happy",
                top_value="community",
            )
        )
        repo.create(
            JournalEntry(
                text="Export entry two",
                user_id="export_user",
                mood_score=6.5,
                tags=["routine"],
                top_emotion="calm",
                top_value="growth",
            )
        )
        # Soft-deleted entry (should NOT be exported)
        repo.create(
            JournalEntry(
                text="Deleted entry",
                user_id="export_user",
                is_deleted=True,
            )
        )
        # Another user's entry (should NOT be exported)
        repo.create(
            JournalEntry(
                text="Other user entry",
                user_id="other_user",
            )
        )

    def test_export_json(self, api_client: TestClient):
        response = api_client.get(
            "/api/v1/journals/export?format=json",
            headers={"X-User-ID": "export_user"},
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert len(data) == 2
        texts = [d["text"] for d in data]
        assert "Export entry one" in texts
        assert "Export entry two" in texts
        assert "Deleted entry" not in texts
        assert "Other user entry" not in texts

    def test_export_csv(self, api_client: TestClient):
        response = api_client.get(
            "/api/v1/journals/export?format=csv",
            headers={"X-User-ID": "export_user"},
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]

        csv_text = response.text
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["text"] == "Export entry two"  # deterministic newest first
        assert rows[1]["text"] == "Export entry one"
        assert "Other user entry" not in csv_text
        assert "Deleted entry" not in csv_text

    def test_export_with_tag_filter(self, api_client: TestClient):
        response = api_client.get(
            "/api/v1/journals/export?format=json&tag=celebration",
            headers={"X-User-ID": "export_user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["text"] == "Export entry one"


class TestEdgeCasesAndRegression:
    """Tests covering boundary conditions, missing records, transaction rollback, and M5 regression."""

    def test_missing_entry_returns_404(self, api_client: TestClient):
        r_get = api_client.get("/api/v1/journals/99999", headers={"X-User-ID": "anyone"})
        assert r_get.status_code == 404
        assert r_get.json()["detail"] == "Journal entry not found"

        r_patch = api_client.patch(
            "/api/v1/journals/99999", json={"is_private": True}, headers={"X-User-ID": "anyone"}
        )
        assert r_patch.status_code == 404

        r_delete = api_client.delete("/api/v1/journals/99999", headers={"X-User-ID": "anyone"})
        assert r_delete.status_code == 404

        r_restore = api_client.post(
            "/api/v1/journals/99999/restore", headers={"X-User-ID": "anyone"}
        )
        assert r_restore.status_code == 404

    def test_pagination_boundary_validations(self, api_client: TestClient):
        # Negative skip -> 422
        r_neg_skip = api_client.get("/api/v1/journals?skip=-1", headers={"X-User-ID": "u"})
        assert r_neg_skip.status_code == 422

        # Limit 0 -> 422
        r_zero_limit = api_client.get("/api/v1/journals?limit=0", headers={"X-User-ID": "u"})
        assert r_zero_limit.status_code == 422

        # Limit > 100 -> 422
        r_over_limit = api_client.get("/api/v1/journals?limit=101", headers={"X-User-ID": "u"})
        assert r_over_limit.status_code == 422

    def test_empty_result_set(self, api_client: TestClient):
        response = api_client.get(
            "/api/v1/journals?search=NONEXISTENT_PHRASE_XYZ",
            headers={"X-User-ID": "empty_user"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["has_more"] is False

    def test_transaction_rollback_on_persistence_failure(
        self,
        api_client: TestClient,
        session: Session,
    ):
        """If database persistence fails unexpectedly, rollback occurs and 500 is returned."""
        from app.api.deps import get_journal_repo

        mock_repo = MagicMock(spec=JournalRepository)
        mock_repo.create.side_effect = RuntimeError("Database write error")

        app.dependency_overrides[get_journal_repo] = lambda: mock_repo

        response = api_client.post(
            "/api/v1/journals",
            json={"text": "Safe entry that encounters a db crash"},
            headers={"X-User-ID": "user_crash"},
        )
        assert response.status_code == 500
        assert "Failed to persist journal entry" in response.json()["detail"]

    def test_milestone_5_regression_memory_retrieval_integration(
        self,
        api_client: TestClient,
        session: Session,
    ):
        """Milestone 5 safety regression:

        - Safe input retrieves approved memory for current user and incorporates it into reflection.
        - Cross-user approved memory is NOT retrieved.
        """
        from app.models.memory import Memory
        from app.repositories.memory import MemoryRepository

        store = LocalVectorStore(dimension=4)
        emb_provider = DeterministicEmbeddingProvider(dimension=4)
        mem_service = MemoryRetrievalService(store, emb_provider, expected_dimension=4)

        # Alice's approved memory
        repo = MemoryRepository(session)
        alice_mem = repo.create(
            Memory(
                content="Valued my daily walk in the park",
                is_approved=True,
                user_id="alice",
            )
        )
        mem_service.index_approved_memory(session, alice_mem.id)  # type: ignore[arg-type]

        # Bob's approved memory
        bob_mem = repo.create(
            Memory(
                content="Bob's secret meditation practice",
                is_approved=True,
                user_id="bob",
            )
        )
        mem_service.index_approved_memory(session, bob_mem.id)  # type: ignore[arg-type]

        # Configure pipeline with this memory service
        pipeline = JournalReflectionPipeline(memory_service=mem_service)
        app.dependency_overrides[get_pipeline] = lambda: pipeline

        # Alice posts a journal entry
        response = api_client.post(
            "/api/v1/journals",
            json={"text": "Taking time to walk and breathe today."},
            headers={"X-User-ID": "alice"},
        )
        assert response.status_code == 201
        data = response.json()

        # Alice's reflection incorporates context, but Bob's memory is NOT present
        created_entry = session.get(JournalEntry, data["entry"]["id"])
        assert created_entry is not None
        assert created_entry.ai_response is not None
        assert "Bob's secret meditation" not in created_entry.ai_response
