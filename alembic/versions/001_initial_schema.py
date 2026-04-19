"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-04-07
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions (already created by init_extensions.sql in Docker,
    # but safe to run again for non-Docker setups)
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------ #
    # entities — base table for all memory types                          #
    # ------------------------------------------------------------------ #
    op.execute("""
        CREATE TABLE entities (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            entity_type   VARCHAR(20) NOT NULL
                          CHECK (entity_type IN ('thought','fact','source','datasource','rule')),
            title         VARCHAR(500),
            content       TEXT NOT NULL,
            summary       TEXT,
            keywords      TEXT[]  DEFAULT '{}',
            importance    FLOAT   DEFAULT 0.5
                          CHECK (importance BETWEEN 0 AND 1),
            embedding     vector(1536),
            search_vector tsvector,
            notes         TEXT,
            metadata      JSONB   DEFAULT '{}',
            created_at    TIMESTAMPTZ DEFAULT now(),
            updated_at    TIMESTAMPTZ DEFAULT now(),
            accessed_at   TIMESTAMPTZ,
            access_count  INTEGER DEFAULT 0
        )
    """)

    op.execute("CREATE INDEX entities_search_idx     ON entities USING GIN(search_vector)")
    op.execute("CREATE INDEX entities_keywords_idx   ON entities USING GIN(keywords)")
    op.execute("CREATE INDEX entities_type_idx       ON entities(entity_type)")
    op.execute("CREATE INDEX entities_importance_idx ON entities(importance DESC)")
    op.execute("CREATE INDEX entities_trgm_idx       ON entities USING GIN(content gin_trgm_ops)")

    # ------------------------------------------------------------------ #
    # extension tables                                                    #
    # ------------------------------------------------------------------ #
    op.execute("""
        CREATE TABLE thoughts_ext (
            entity_id         UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            certainty         FLOAT DEFAULT 0.5 CHECK (certainty BETWEEN 0 AND 1),
            context           TEXT,
            emotional_valence FLOAT DEFAULT 0 CHECK (emotional_valence BETWEEN -1 AND 1)
        )
    """)

    op.execute("""
        CREATE TABLE facts_ext (
            entity_id        UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            certainty        FLOAT DEFAULT 0.8 CHECK (certainty BETWEEN 0 AND 1),
            is_verified      BOOLEAN DEFAULT FALSE,
            source_entity_id UUID REFERENCES entities(id) ON DELETE SET NULL
        )
    """)

    op.execute("""
        CREATE TABLE sources_ext (
            entity_id       UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            url             TEXT NOT NULL,
            domain          TEXT,
            http_status     INTEGER,
            last_checked_at TIMESTAMPTZ
        )
    """)

    op.execute("""
        CREATE TABLE datasources_ext (
            entity_id    UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            file_path    TEXT,
            url          TEXT,
            content_hash TEXT,
            word_count   INTEGER,
            language     VARCHAR(10) DEFAULT 'en'
        )
    """)

    op.execute("""
        CREATE TABLE rules_ext (
            entity_id UUID PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
            always_on BOOLEAN DEFAULT FALSE,
            category  VARCHAR(100),
            priority  INTEGER DEFAULT 50 CHECK (priority BETWEEN 1 AND 100),
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    # ------------------------------------------------------------------ #
    # relations                                                           #
    # ------------------------------------------------------------------ #
    op.execute("""
        CREATE TABLE relations (
            id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            from_entity_id   UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            to_entity_id     UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            relation_type    VARCHAR(100) NOT NULL,
            relevance_score  FLOAT DEFAULT 0.5 CHECK (relevance_score BETWEEN 0 AND 1),
            importance_score FLOAT DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1),
            is_bidirectional BOOLEAN DEFAULT FALSE,
            description      TEXT,
            notes            TEXT,
            created_at       TIMESTAMPTZ DEFAULT now(),
            updated_at       TIMESTAMPTZ DEFAULT now(),
            UNIQUE(from_entity_id, to_entity_id, relation_type)
        )
    """)

    op.execute("CREATE INDEX relations_from_idx ON relations(from_entity_id)")
    op.execute("CREATE INDEX relations_to_idx   ON relations(to_entity_id)")
    op.execute("CREATE INDEX relations_type_idx ON relations(relation_type)")

    # trigger: maintain search_vector on insert/update
    op.execute("""
        CREATE OR REPLACE FUNCTION update_search_vector()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN
            NEW.search_vector := to_tsvector('english',
                coalesce(NEW.title,'') || ' ' ||
                NEW.content            || ' ' ||
                array_to_string(coalesce(NEW.keywords, '{}'), ' '));
            RETURN NEW;
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER entities_search_vector_update
        BEFORE INSERT OR UPDATE ON entities
        FOR EACH ROW EXECUTE FUNCTION update_search_vector()
    """)

    # auto-update updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER entities_updated_at
        BEFORE UPDATE ON entities
        FOR EACH ROW EXECUTE FUNCTION update_updated_at()
    """)
    op.execute("""
        CREATE TRIGGER relations_updated_at
        BEFORE UPDATE ON relations
        FOR EACH ROW EXECUTE FUNCTION update_updated_at()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS relations_updated_at        ON relations")
    op.execute("DROP TRIGGER IF EXISTS entities_updated_at         ON entities")
    op.execute("DROP TRIGGER IF EXISTS entities_search_vector_update ON entities")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at()")
    op.execute("DROP FUNCTION IF EXISTS update_search_vector()")
    op.execute("DROP TABLE IF EXISTS relations")
    op.execute("DROP TABLE IF EXISTS rules_ext")
    op.execute("DROP TABLE IF EXISTS datasources_ext")
    op.execute("DROP TABLE IF EXISTS sources_ext")
    op.execute("DROP TABLE IF EXISTS facts_ext")
    op.execute("DROP TABLE IF EXISTS thoughts_ext")
    op.execute("DROP TABLE IF EXISTS entities")
