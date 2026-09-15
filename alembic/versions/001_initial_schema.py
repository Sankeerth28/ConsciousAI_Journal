"""Initial database schema.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-14 17:30:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- journal_entries ---
    op.create_table(
        "journal_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("mood_score", sa.Float(), nullable=True),
        sa.Column("top_emotion", sa.String(length=50), nullable=True),
        sa.Column("top_value", sa.String(length=50), nullable=True),
        sa.Column("detected_emotions", sa.JSON(), nullable=False),
        sa.Column("detected_values", sa.JSON(), nullable=False),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("ai_response", sa.Text(), nullable=True),
        sa.Column("feedback", sa.String(length=255), nullable=True),
        sa.Column("legacy_source_hash", sa.String(length=64), nullable=True),
        sa.Column("is_private", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("legacy_source_hash", name="uq_journal_entries_legacy_source_hash"),
    )
    op.create_index("ix_journal_entries_top_emotion", "journal_entries", ["top_emotion"])
    op.create_index("ix_journal_entries_top_value", "journal_entries", ["top_value"])
    op.create_index(
        "ix_journal_entries_legacy_source_hash", "journal_entries", ["legacy_source_hash"]
    )
    op.create_index("ix_journal_entries_is_private", "journal_entries", ["is_private"])
    op.create_index("ix_journal_entries_is_deleted", "journal_entries", ["is_deleted"])
    op.create_index("ix_journal_entries_created_at", "journal_entries", ["created_at"])

    # --- memories ---
    op.create_table(
        "memories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_entry_id", sa.Integer(), nullable=True),
        sa.Column("memory_type", sa.String(length=50), server_default="reflection", nullable=False),
        sa.Column("importance", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("is_approved", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_entry_id"],
            ["journal_entries.id"],
            name="fk_memories_source_entry_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memories_source_entry_id", "memories", ["source_entry_id"])
    op.create_index("ix_memories_memory_type", "memories", ["memory_type"])
    op.create_index("ix_memories_is_approved", "memories", ["is_approved"])
    op.create_index("ix_memories_created_at", "memories", ["created_at"])

    # --- feedbacks ---
    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("journal_entry_id", sa.Integer(), nullable=False),
        sa.Column("feedback_type", sa.String(length=50), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["journal_entry_id"],
            ["journal_entries.id"],
            name="fk_feedbacks_journal_entry_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedbacks_journal_entry_id", "feedbacks", ["journal_entry_id"])
    op.create_index("ix_feedbacks_feedback_type", "feedbacks", ["feedback_type"])
    op.create_index("ix_feedbacks_created_at", "feedbacks", ["created_at"])

    # --- user_settings ---
    op.create_table(
        "user_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("persona", sa.String(length=50), server_default="Supportive", nullable=False),
        sa.Column("memory_enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("analytics_enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column(
            "preferred_response_length",
            sa.String(length=50),
            server_default="medium",
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("user_settings")
    op.drop_table("feedbacks")
    op.drop_table("memories")
    op.drop_table("journal_entries")
