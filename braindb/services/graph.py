"""
Graph traversal via recursive CTE up to 3 hops with relevance fading.
"""
import psycopg2.extras


GRAPH_SQL = """
WITH RECURSIVE traversal AS (
    SELECT
        e.id, e.entity_type, e.title, e.content, e.summary,
        e.keywords, e.importance, e.source, e.notes,
        e.created_at, e.updated_at, e.accessed_at, e.access_count, e.metadata,
        0                   AS depth,
        1.0::FLOAT          AS accumulated_relevance,
        ARRAY[e.id]         AS visited,
        NULL::UUID          AS via_relation_id,
        NULL::TEXT          AS via_relation_type,
        NULL::TEXT          AS via_description,
        NULL::TEXT          AS via_notes
    FROM entities e
    WHERE e.id = ANY(%s::uuid[])

    UNION ALL

    SELECT
        target.id, target.entity_type, target.title, target.content, target.summary,
        target.keywords, target.importance, target.source, target.notes,
        target.created_at, target.updated_at, target.accessed_at, target.access_count, target.metadata,
        t.depth + 1,
        (
            t.accumulated_relevance
            * r.relevance_score
            * CASE t.depth + 1
                WHEN 1 THEN 1.0
                WHEN 2 THEN 0.6
                ELSE        0.3
              END
        )::FLOAT,
        t.visited || target.id,
        r.id,
        r.relation_type,
        r.description,
        r.notes
    FROM traversal t
    JOIN relations r ON (
        r.from_entity_id = t.id
        OR (r.is_bidirectional AND r.to_entity_id = t.id)
    )
    JOIN entities target ON
        target.id = CASE
            WHEN r.from_entity_id = t.id THEN r.to_entity_id
            ELSE r.from_entity_id
        END
    WHERE t.depth < %s
      AND NOT (target.id = ANY(t.visited))
      AND (
            t.accumulated_relevance
            * r.relevance_score
            * CASE t.depth + 1 WHEN 1 THEN 1.0 WHEN 2 THEN 0.6 ELSE 0.3 END
          ) > %s
)
SELECT DISTINCT ON (id)
    id, entity_type, title, content, summary,
    keywords, importance, source, notes,
    created_at, updated_at, accessed_at, access_count, metadata,
    depth           AS min_depth,
    accumulated_relevance AS relevance,
    via_relation_id, via_relation_type, via_description, via_notes
FROM traversal
ORDER BY id, depth, accumulated_relevance DESC
"""


def graph_expand(conn, seed_ids: list, max_depth: int, min_relevance: float) -> list[dict]:
    if not seed_ids:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(GRAPH_SQL, (seed_ids, max_depth, min_relevance))
        return [dict(r) for r in cur.fetchall()]
