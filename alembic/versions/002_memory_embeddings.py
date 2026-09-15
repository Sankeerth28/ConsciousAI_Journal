"""Memory embeddings and soft delete support.

Revision ID: 002_memory_embeddings
Revises: 001_initial_schema
Create Date: 2026-09-14 21:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002_memory_embeddings"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Update memories table with soft delete and ownership fields
    with op.batch_alter_table("memories") as batch_op:
        batch_op.add_column(
            sa.Column("is_deleted", sa.Boolean(), server_default=sa.false(), nullable=False)
        )
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("user_id", sa.String(length=100), nullable=True))

    op.create_index("ix_memories_is_deleted", "memories", ["is_deleted"])
    op.create_index("ix_memories_user_id", "memories", ["user_id"])
    op.create_index(
        "ix_memories_is_approved_is_deleted",
        "memories",
        ["is_approved", "is_deleted"],
    )

    # 2. Create memory_embeddings table
    op.create_table(
        "memory_embeddings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("memory_id", sa.Integer(), nullable=False),
        sa.Column("embedding_json", sa.JSON(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=20), server_default="1.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_memory_embeddings_memory_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("memory_id", name="uq_memory_embeddings_memory_id"),
    )
    op.create_index("ix_memory_embeddings_memory_id", "memory_embeddings", ["memory_id"])
    op.create_index("ix_memory_embeddings_dimension", "memory_embeddings", ["dimension"])
    op.create_index("ix_memory_embeddings_model_name", "memory_embeddings", ["model_name"])


def downgrade() -> None:
    op.drop_table("memory_embeddings")

    op.drop_index("ix_memories_is_approved_is_deleted", table_name="memories")
    op.drop_index("ix_memories_user_id", table_name="memories")
    op.drop_index("ix_memories_is_deleted", table_name="memories")

    with op.batch_alter_table("memories") as batch_op:
        batch_op.drop_column("user_id")
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("is_deleted")
