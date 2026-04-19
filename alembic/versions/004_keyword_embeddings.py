"""keyword embeddings — keywords as entities with vector search

Revision ID: 004
Revises: 003
Create Date: 2026-04-12
"""
from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0. Add 'keyword' to the entity_type CHECK constraint
    op.execute("ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_entity_type_check")
    op.execute("""
        ALTER TABLE entities ADD CONSTRAINT entities_entity_type_check
        CHECK (entity_type IN ('thought','fact','source','datasource','rule','keyword'))
    """)

    # 1. Resize embedding column from 1536 to 1024 (Qwen3-Embedding-0.6B)
    #    Column is currently all NULLs — no data loss.
    op.execute("ALTER TABLE entities ALTER COLUMN embedding TYPE vector(1024)")

    # 2. Vector index for cosine similarity search
    op.execute("""
        CREATE INDEX entities_embedding_idx ON entities
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20)
    """)

    # 3. Unique partial index: prevent duplicate keyword entities
    op.execute("""
        CREATE UNIQUE INDEX keywords_unique_content_idx
        ON entities(content) WHERE entity_type = 'keyword'
    """)

    # 4. Backfill: create keyword entities from existing TEXT[] data
    op.execute("""
        INSERT INTO entities (entity_type, content, importance, source)
        SELECT DISTINCT 'keyword', kw, 0.5, 'system'
        FROM entities, unnest(keywords) AS kw
        WHERE keywords != '{}' AND entity_type != 'keyword'
        ON CONFLICT DO NOTHING
    """)

    # 5. Backfill: create tagged_with relations for existing entities
    op.execute("""
        INSERT INTO relations (from_entity_id, to_entity_id, relation_type, relevance_score)
        SELECT e.id, kw_ent.id, 'tagged_with', 0.8
        FROM entities e, unnest(e.keywords) AS kw
        JOIN entities kw_ent ON kw_ent.entity_type = 'keyword' AND kw_ent.content = kw
        WHERE e.entity_type != 'keyword' AND e.keywords != '{}'
        ON CONFLICT DO NOTHING
    """)


def downgrade() -> None:
    # Remove backfilled relations
    op.execute("DELETE FROM relations WHERE relation_type = 'tagged_with'")
    # Remove keyword entities
    op.execute("DELETE FROM entities WHERE entity_type = 'keyword'")
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS keywords_unique_content_idx")
    op.execute("DROP INDEX IF EXISTS entities_embedding_idx")
    # Resize embedding back to 1536
    op.execute("ALTER TABLE entities ALTER COLUMN embedding TYPE vector(1536)")
    # Restore original entity_type CHECK constraint
    op.execute("ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_entity_type_check")
    op.execute("""
        ALTER TABLE entities ADD CONSTRAINT entities_entity_type_check
        CHECK (entity_type IN ('thought','fact','source','datasource','rule'))
    """)
