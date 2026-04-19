"""
Keyword service — manages keyword entities and their relations.
Keywords are entities (entity_type='keyword') linked to other entities via 'tagged_with' relations.
Each keyword has an embedding for semantic search.
"""
import psycopg2.extras

from braindb.services.embedding_service import EmbeddingService


def ensure_keyword_entities(
    conn, keywords: list[str], embedding_service: EmbeddingService | None = None
) -> dict[str, str]:
    """
    For each keyword, ensure a keyword entity exists. Generate embeddings for new ones.
    Returns {keyword_text: entity_id} mapping.
    """
    if not keywords:
        return {}

    result = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for kw in keywords:
            kw = kw.strip()
            if not kw:
                continue

            # Check if keyword entity already exists
            cur.execute(
                "SELECT id FROM entities WHERE entity_type = 'keyword' AND content = %s",
                (kw,),
            )
            row = cur.fetchone()

            if row:
                result[kw] = row["id"]
            else:
                # Create new keyword entity
                embedding = None
                if embedding_service and embedding_service.is_available():
                    embedding = embedding_service.embed(kw)

                cur.execute(
                    """
                    INSERT INTO entities (entity_type, content, importance, source, embedding)
                    VALUES ('keyword', %s, 0.5, 'system', %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """,
                    (kw, str(embedding) if embedding else None),
                )
                new_row = cur.fetchone()
                if new_row:
                    result[kw] = new_row["id"]
                else:
                    # Race condition: another request created it between SELECT and INSERT
                    cur.execute(
                        "SELECT id FROM entities WHERE entity_type = 'keyword' AND content = %s",
                        (kw,),
                    )
                    result[kw] = cur.fetchone()["id"]

    return result


def link_entity_to_keywords(conn, entity_id: str, keyword_entity_ids: list[str]) -> None:
    """Create 'tagged_with' relations from entity to keyword entities."""
    if not keyword_entity_ids:
        return
    with conn.cursor() as cur:
        for kw_id in keyword_entity_ids:
            try:
                cur.execute(
                    """
                    INSERT INTO relations (from_entity_id, to_entity_id, relation_type, relevance_score)
                    VALUES (%s, %s, 'tagged_with', 0.8)
                    ON CONFLICT (from_entity_id, to_entity_id, relation_type) DO NOTHING
                    """,
                    (str(entity_id), str(kw_id)),
                )
            except Exception:
                pass  # skip on any error (e.g., FK violation if entity was deleted concurrently)


def sync_keywords_for_entity(
    conn, entity_id: str, new_keywords: list[str], embedding_service: EmbeddingService | None = None
) -> None:
    """
    Full sync for entity updates: diff old vs new keywords,
    create/remove keyword entities and tagged_with relations.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # Get current keyword relations
        cur.execute(
            """
            SELECT r.id AS rel_id, e.content AS keyword, e.id AS kw_entity_id
            FROM relations r
            JOIN entities e ON e.id = r.to_entity_id
            WHERE r.from_entity_id = %s AND r.relation_type = 'tagged_with'
            """,
            (str(entity_id),),
        )
        existing = {row["keyword"]: row for row in cur.fetchall()}

    current_set = set(existing.keys())
    new_set = set(kw.strip() for kw in new_keywords if kw.strip())

    # Remove old
    to_remove = current_set - new_set
    if to_remove:
        rel_ids = [str(existing[kw]["rel_id"]) for kw in to_remove]
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM relations WHERE id = ANY(%s::uuid[])", (rel_ids,)
            )

    # Add new
    to_add = new_set - current_set
    if to_add:
        kw_map = ensure_keyword_entities(conn, list(to_add), embedding_service)
        link_entity_to_keywords(conn, entity_id, list(kw_map.values()))


def find_similar_keywords(conn, query_embedding: list[float], limit: int = 20) -> list[dict]:
    """Vector search against keyword entities. Returns keywords sorted by cosine similarity."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, content AS keyword,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM entities
            WHERE entity_type = 'keyword' AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(query_embedding), str(query_embedding), limit),
        )
        return [dict(r) for r in cur.fetchall()]


def find_entities_for_keywords(conn, keyword_entity_ids: list[str]) -> list[dict]:
    """
    Find all non-keyword entities tagged with the given keyword entities.
    Returns entity rows with their matched keyword IDs.
    """
    if not keyword_entity_ids:
        return []

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT e.*, array_agg(r.to_entity_id) AS matched_keyword_ids
            FROM entities e
            JOIN relations r ON r.from_entity_id = e.id
            WHERE r.to_entity_id = ANY(%s::uuid[])
              AND r.relation_type = 'tagged_with'
              AND e.entity_type != 'keyword'
            GROUP BY e.id
            """,
            ([str(kid) for kid in keyword_entity_ids],),
        )
        return [dict(r) for r in cur.fetchall()]


def generate_missing_embeddings(conn, embedding_service: EmbeddingService) -> dict:
    """Generate embeddings for all keyword entities that don't have one. Returns stats."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, content FROM entities WHERE entity_type = 'keyword' AND embedding IS NULL"
        )
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        return {"total": 0, "generated": 0}

    texts = [r["content"] for r in rows]
    embeddings = embedding_service.embed_batch(texts)

    if not embeddings:
        return {"total": len(rows), "generated": 0, "error": "embedding generation failed"}

    generated = 0
    with conn.cursor() as cur:
        for row, emb in zip(rows, embeddings):
            cur.execute(
                "UPDATE entities SET embedding = %s WHERE id = %s",
                (str(emb), str(row["id"])),
            )
            generated += 1

    return {"total": len(rows), "generated": generated}
