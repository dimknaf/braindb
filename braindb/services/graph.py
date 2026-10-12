"""
Graph traversal up to 3 hops with relevance fading.

A hop's relevance is the relevance so far times the relation's
relevance_score and importance_score (0.5 when unset), and a hop is kept only
above min_relevance. Every entity reached gets its smallest depth and, at that
depth, its best relevance and the seed it descends from.

The search goes level by level and keeps one state per (entity, depth): the
best relevance reaching it and its seed. It never lists paths, so an entity
with many relations costs one row per depth, not one per path through it,
and only the entities reached are read, once, at the end.

Relation scores lie in [0, 1] (a CHECK constraint), so relevance never grows
along a walk: a walk that revisits an entity can always be shortened into a
valid, shorter walk at least as relevant. At an entity's smallest depth every
valid walk is therefore a simple path, and the result equals the best simple
path's. Between seeds that reach an entity at the same depth with the same
relevance, the smallest seed id wins.
"""
import psycopg2.extras

_EDGES_SQL = """
SELECT id, from_entity_id, to_entity_id, relation_type,
       relevance_score, importance_score, description, notes
FROM relations
WHERE from_entity_id = ANY(%s::uuid[]) OR to_entity_id = ANY(%s::uuid[])
"""

_ENTITIES_SQL = """
SELECT id, entity_type, title, content, summary,
       keywords, importance, source, notes,
       created_at, updated_at, accessed_at, access_count, metadata
FROM entities
WHERE id = ANY(%s::uuid[])
"""


def _better(acc: float, origin: str, edge_id: str, held: tuple | None) -> bool:
    """Whether (acc, origin, edge) beats the state held for an entity: higher
    relevance, then the smaller seed id, then the smaller relation id."""
    if held is None:
        return True
    if acc != held[0]:
        return acc > held[0]
    if origin != held[1]:
        return origin < held[1]
    return edge_id < held[2]["id"]


def graph_expand(conn, seed_ids: list, max_depth: int, min_relevance: float) -> list[dict]:
    if not seed_ids:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        seeds = sorted({str(s) for s in seed_ids})
        # (relevance, seed, last relation) per entity at the current depth
        level: dict[str, tuple] = {s: (1.0, s, None) for s in seeds}
        reached: dict[str, tuple] = {s: (0, 1.0, s, None) for s in seeds}
        for depth in range(1, max_depth + 1):
            frontier = list(level)
            cur.execute(_EDGES_SQL, (frontier, frontier))
            following: dict[str, tuple] = {}
            for edge in cur.fetchall():
                ends = (str(edge["from_entity_id"]), str(edge["to_entity_id"]))
                if ends[0] == ends[1] or edge["relevance_score"] is None:
                    continue
                importance = edge["importance_score"]
                for here, there in (ends, ends[::-1]):
                    if here not in level:
                        continue
                    so_far, origin, _ = level[here]
                    acc = float(so_far * edge["relevance_score"] * (0.5 if importance is None else importance))
                    if not acc > min_relevance:
                        continue
                    if _better(acc, origin, str(edge["id"]), following.get(there)):
                        following[there] = (acc, origin, edge)
            for eid, (acc, origin, edge) in following.items():
                if eid not in reached:
                    reached[eid] = (depth, acc, origin, edge)
            level = following
            if not level:
                break

        cur.execute(_ENTITIES_SQL, (list(reached),))
        rows = {str(r["id"]): dict(r) for r in cur.fetchall()}

    out = []
    for eid in sorted(rows):
        depth, acc, origin, edge = reached[eid]
        out.append({
            **rows[eid],
            "min_depth": depth,
            "relevance": acc,
            "via_relation_id": edge["id"] if edge else None,
            "via_relation_type": edge["relation_type"] if edge else None,
            "via_description": edge["description"] if edge else None,
            "via_notes": edge["notes"] if edge else None,
            "seed_origin_id": origin,
        })
    return out
