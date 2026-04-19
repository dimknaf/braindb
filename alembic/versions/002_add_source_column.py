"""add source column for provenance tracking

Revision ID: 002
Revises: 001
Create Date: 2026-04-08
"""
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE entities ADD COLUMN source VARCHAR(50) DEFAULT NULL")
    op.execute("CREATE INDEX entities_source_idx ON entities(source)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS entities_source_idx")
    op.execute("ALTER TABLE entities DROP COLUMN IF EXISTS source")
