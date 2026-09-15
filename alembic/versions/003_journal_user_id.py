"""Add user_id to journal entries.

Revision ID: 003_journal_user_id
Revises: 002_memory_embeddings
Create Date: 2026-09-15 06:15:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "003_journal_user_id"
down_revision = "002_memory_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("journal_entries") as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.String(length=100), nullable=True))
        batch_op.create_index("ix_journal_entries_user_id", ["user_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("journal_entries") as batch_op:
        batch_op.drop_index("ix_journal_entries_user_id")
        batch_op.drop_column("user_id")
