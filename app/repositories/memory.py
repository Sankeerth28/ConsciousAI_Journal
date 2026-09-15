"""Memory repository for database access."""

from __future__ import annotations

from typing import Any

from sqlmodel import select

from app.models.base import utcnow
from app.models.memory import Memory, MemoryEmbedding
from app.repositories.base import BaseRepository

IMMUTABLE_FIELDS = {"id", "created_at"}


class MemoryRepository(BaseRepository[Memory]):
    """Repository handling persistence operations for Memory and MemoryEmbedding records."""

    def create(self, memory: Memory) -> Memory:
        """Persist a new memory.

        Note: New memories default to is_approved=False and is_deleted=False.
        """
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        return memory

    def get_by_id(self, memory_id: int, include_deleted: bool = False) -> Memory | None:
        """Fetch memory by primary key, excluding soft-deleted ones by default."""
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return None
        if memory.is_deleted and not include_deleted:
            return None
        return memory

    def list(
        self,
        source_entry_id: int | None = None,
        memory_type: str | None = None,
        is_approved: bool | None = None,
        include_deleted: bool = False,
        user_id: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Memory]:
        """Query memories with filtering and deterministic pagination."""
        statement = select(Memory)

        if not include_deleted:
            statement = statement.where(Memory.is_deleted.is_(False))

        if user_id is not None:
            statement = statement.where(Memory.user_id == user_id)

        if source_entry_id is not None:
            statement = statement.where(Memory.source_entry_id == source_entry_id)

        if memory_type is not None:
            statement = statement.where(Memory.memory_type == memory_type)

        if is_approved is not None:
            statement = statement.where(Memory.is_approved.is_(is_approved))

        statement = statement.order_by(Memory.created_at.desc(), Memory.id.desc())
        statement = statement.offset(skip).limit(limit)

        return list(self.session.exec(statement).all())

    def update(self, memory_id: int, **updates: Any) -> Memory | None:
        """Update memory fields with validation."""
        memory = self.get_by_id(memory_id, include_deleted=True)
        if not memory:
            return None

        for field, value in updates.items():
            if field in IMMUTABLE_FIELDS:
                msg = f"Cannot update immutable field '{field}'"
                raise ValueError(msg)
            if not hasattr(memory, field):
                msg = f"Unknown field '{field}' on Memory"
                raise ValueError(msg)
            setattr(memory, field, value)

        memory.updated_at = utcnow()
        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)
        return memory

    def approve(self, memory_id: int) -> Memory | None:
        """Mark a memory as approved by the user."""
        return self.update(memory_id, is_approved=True)

    def soft_delete(self, memory_id: int) -> bool:
        """Mark a memory as soft-deleted."""
        memory = self.session.get(Memory, memory_id)
        if not memory:
            return False

        memory.is_deleted = True
        memory.deleted_at = utcnow()
        memory.updated_at = utcnow()
        self.session.add(memory)
        self.session.commit()
        return True

    def restore(self, memory_id: int) -> bool:
        """Restore a soft-deleted memory."""
        memory = self.session.get(Memory, memory_id)
        if not memory:
            return False

        memory.is_deleted = False
        memory.deleted_at = None
        memory.updated_at = utcnow()
        self.session.add(memory)
        self.session.commit()
        return True

    def delete(self, memory_id: int, soft: bool = True) -> bool:
        """Delete a memory record (soft-delete by default, hard-delete if soft=False)."""
        if soft:
            return self.soft_delete(memory_id)

        memory = self.session.get(Memory, memory_id)
        if not memory:
            return False

        self.session.delete(memory)
        self.session.commit()
        return True

    def get_approved_for_indexing(self, user_id: str | None = None) -> list[Memory]:
        """Fetch all eligible memories for indexing.

        Eligibility rules:
        - is_approved == True
        - is_deleted == False
        - content is not empty
        - user_id matches if specified
        """
        statement = select(Memory).where(
            Memory.is_approved.is_(True),
            Memory.is_deleted.is_(False),
            Memory.content != "",
        )
        if user_id is not None:
            statement = statement.where(Memory.user_id == user_id)

        statement = statement.order_by(Memory.id.asc())
        return list(self.session.exec(statement).all())

    def upsert_embedding(
        self,
        memory_id: int,
        embedding: list[float],
        dimension: int,
        provider: str,
        model_name: str,
        version: str = "1.0",
    ) -> MemoryEmbedding:
        """Persist or replace the dense vector embedding for a memory."""
        statement = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
        existing = self.session.exec(statement).first()

        if existing:
            existing.embedding_json = embedding
            existing.dimension = dimension
            existing.provider = provider
            existing.model_name = model_name
            existing.version = version
            existing.updated_at = utcnow()
            self.session.add(existing)
            self.session.commit()
            self.session.refresh(existing)
            return existing

        new_embedding = MemoryEmbedding(
            memory_id=memory_id,
            embedding_json=embedding,
            dimension=dimension,
            provider=provider,
            model_name=model_name,
            version=version,
        )
        self.session.add(new_embedding)
        self.session.commit()
        self.session.refresh(new_embedding)
        return new_embedding

    def get_embedding(self, memory_id: int) -> MemoryEmbedding | None:
        """Fetch the embedding for a given memory ID."""
        statement = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
        return self.session.exec(statement).first()

    def delete_embedding(self, memory_id: int) -> bool:
        """Remove embedding record for a memory if present."""
        statement = select(MemoryEmbedding).where(MemoryEmbedding.memory_id == memory_id)
        existing = self.session.exec(statement).first()
        if not existing:
            return False

        self.session.delete(existing)
        self.session.commit()
        return True
