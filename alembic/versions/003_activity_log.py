"""activity log

Revision ID: 003
Revises: 002
Create Date: 2026-04-08
"""
from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE activity_log (
            id              BIGSERIAL PRIMARY KEY,
            timestamp       TIMESTAMPTZ DEFAULT now() NOT NULL,
            operation       VARCHAR(30) NOT NULL,
            entity_type     VARCHAR(20),
            entity_id       UUID,
            details         JSONB DEFAULT '{}'::jsonb,
            context_note    TEXT
        )
    """)
    op.execute("CREATE INDEX activity_log_timestamp_idx ON activity_log(timestamp DESC)")
    op.execute("CREATE INDEX activity_log_operation_idx ON activity_log(operation)")
    op.execute("CREATE INDEX activity_log_entity_id_idx ON activity_log(entity_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS activity_log_entity_id_idx")
    op.execute("DROP INDEX IF EXISTS activity_log_operation_idx")
    op.execute("DROP INDEX IF EXISTS activity_log_timestamp_idx")
    op.execute("DROP TABLE IF EXISTS activity_log")
