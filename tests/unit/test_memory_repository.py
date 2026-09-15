"""Unit tests for MemoryRepository, soft-deletion, approval, and embedding lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models.memory import Memory
from app.repositories.memory import MemoryRepository

if TYPE_CHECKING:
    from sqlmodel import Session


class TestMemoryRepository:
    """Comprehensive tests for MemoryRepository persistence and rules."""

    def test_memory_creation_defaults_unapproved_and_not_deleted(self, session: Session):
        repo = MemoryRepository(session)
        memory = Memory(content="Realized importance of patience.")
        created = repo.create(memory)

        assert created.id is not None
        assert created.is_approved is False
        assert created.is_deleted is False
        assert created.deleted_at is None
        assert created.user_id is None

    def test_approve_memory(self, session: Session):
        repo = MemoryRepository(session)
        created = repo.create(Memory(content="Deep insight"))
        assert created.is_approved is False

        approved = repo.approve(created.id)  # type: ignore[arg-type]
        assert approved is not None
        assert approved.is_approved is True

    def test_soft_delete_and_restore(self, session: Session):
        repo = MemoryRepository(session)
        created = repo.create(Memory(content="Temporary reflection", is_approved=True))

        # Soft delete
        success = repo.soft_delete(created.id)  # type: ignore[arg-type]
        assert success is True

        # Default get_by_id excludes soft-deleted
        assert repo.get_by_id(created.id) is None  # type: ignore[arg-type]

        # include_deleted=True returns it
        deleted_mem = repo.get_by_id(created.id, include_deleted=True)  # type: ignore[arg-type]
        assert deleted_mem is not None
        assert deleted_mem.is_deleted is True
        assert deleted_mem.deleted_at is not None

        # Restore
        restored = repo.restore(created.id)  # type: ignore[arg-type]
        assert restored is True

        active_mem = repo.get_by_id(created.id)  # type: ignore[arg-type]
        assert active_mem is not None
        assert active_mem.is_deleted is False
        assert active_mem.deleted_at is None

    def test_list_excludes_deleted_by_default(self, session: Session):
        repo = MemoryRepository(session)
        m1 = repo.create(Memory(content="Active 1", is_approved=True))
        m2 = repo.create(Memory(content="Active 2", is_approved=True))
        m3 = repo.create(Memory(content="Deleted 3", is_approved=True))
        repo.soft_delete(m3.id)  # type: ignore[arg-type]

        # Default list
        active_list = repo.list()
        active_ids = [m.id for m in active_list]
        assert m1.id in active_ids
        assert m2.id in active_ids
        assert m3.id not in active_ids

        # With include_deleted=True
        all_list = repo.list(include_deleted=True)
        all_ids = [m.id for m in all_list]
        assert m3.id in all_ids

    def test_get_approved_for_indexing_filtering(self, session: Session):
        repo = MemoryRepository(session)

        # 1. Unapproved memory -> excluded
        m_unapproved = repo.create(Memory(content="Unapproved", is_approved=False, user_id="u1"))

        # 2. Approved memory -> included
        m_approved = repo.create(Memory(content="Approved", is_approved=True, user_id="u1"))

        # 3. Soft-deleted approved memory -> excluded
        m_deleted = repo.create(Memory(content="Deleted", is_approved=True, user_id="u1"))
        repo.soft_delete(m_deleted.id)  # type: ignore[arg-type]

        # 4. Empty content memory -> excluded
        m_empty = repo.create(Memory(content="", is_approved=True, user_id="u1"))

        # 5. Different user approved memory
        m_u2 = repo.create(Memory(content="User 2 note", is_approved=True, user_id="u2"))

        # Fetch for user u1
        eligible_u1 = repo.get_approved_for_indexing(user_id="u1")
        eligible_u1_ids = [m.id for m in eligible_u1]

        assert m_approved.id in eligible_u1_ids
        assert m_unapproved.id not in eligible_u1_ids
        assert m_deleted.id not in eligible_u1_ids
        assert m_empty.id not in eligible_u1_ids
        assert m_u2.id not in eligible_u1_ids

        # Fetch without user_id filter -> includes both u1 and u2 approved
        eligible_all = repo.get_approved_for_indexing()
        eligible_all_ids = [m.id for m in eligible_all]
        assert m_approved.id in eligible_all_ids
        assert m_u2.id in eligible_all_ids

    def test_embedding_lifecycle_upsert_and_cascade(self, session: Session):
        repo = MemoryRepository(session)
        memory = repo.create(Memory(content="Insight about boundaries", is_approved=True))
        mem_id = memory.id
        assert mem_id is not None

        # Initially no embedding
        assert repo.get_embedding(mem_id) is None

        # Upsert embedding
        emb1 = repo.upsert_embedding(
            memory_id=mem_id,
            embedding=[0.1, 0.2, 0.3],
            dimension=3,
            provider="local",
            model_name="test-model",
            version="1.0",
        )
        assert emb1.id is not None
        assert emb1.dimension == 3
        assert emb1.embedding_json == [0.1, 0.2, 0.3]

        # Update embedding for the same memory (replacement behavior)
        emb2 = repo.upsert_embedding(
            memory_id=mem_id,
            embedding=[0.4, 0.5, 0.6],
            dimension=3,
            provider="local",
            model_name="test-model-v2",
            version="2.0",
        )
        assert emb2.id == emb1.id
        assert emb2.embedding_json == [0.4, 0.5, 0.6]
        assert emb2.model_name == "test-model-v2"

        # Hard delete memory cascades to embedding
        repo.delete(mem_id, soft=False)
        assert repo.get_by_id(mem_id, include_deleted=True) is None
        assert repo.get_embedding(mem_id) is None

    def test_restoring_memory_does_not_bypass_approval(self, session: Session):
        """Restoring a deleted memory must never accidentally mark it approved."""
        repo = MemoryRepository(session)
        created = repo.create(Memory(content="Unapproved memory", is_approved=False))
        assert created.is_approved is False

        # Soft delete
        repo.soft_delete(created.id)  # type: ignore[arg-type]
        deleted_mem = repo.get_by_id(created.id, include_deleted=True)  # type: ignore[arg-type]
        assert deleted_mem is not None
        assert deleted_mem.is_deleted is True
        assert deleted_mem.is_approved is False

        # Restore
        repo.restore(created.id)  # type: ignore[arg-type]
        restored_mem = repo.get_by_id(created.id)  # type: ignore[arg-type]
        assert restored_mem is not None
        assert restored_mem.is_deleted is False
        assert restored_mem.is_approved is False  # Must remain False!
