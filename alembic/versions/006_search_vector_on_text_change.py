"""search_vector trigger — fire only when the indexed text changes

Revision ID: 006
Revises: 005
Create Date: 2026-10-12

The trigger that rebuilds entities.search_vector fired on every UPDATE,
including the access counting every recall does (access_count,
accessed_at), so each read re-tokenized the full body of every entity it
returned. It now fires only when title, content or keywords are written.
Same function, same stored values; no data changes.
"""
from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS entities_search_vector_update ON entities")
    op.execute("""
        CREATE TRIGGER entities_search_vector_update
        BEFORE INSERT OR UPDATE OF title, content, keywords ON entities
        FOR EACH ROW EXECUTE FUNCTION update_search_vector()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS entities_search_vector_update ON entities")
    op.execute("""
        CREATE TRIGGER entities_search_vector_update
        BEFORE INSERT OR UPDATE ON entities
        FOR EACH ROW EXECUTE FUNCTION update_search_vector()
    """)
