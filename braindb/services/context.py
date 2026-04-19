"""
Context assembly:
  1. Fuzzy search → seed entities
  2. Graph expand → up to 3 hops
  3. Temporal decay + reinforcement → effective_importance
  4. Final rank = search_score * effective_importance * accumulated_relevance
  5. Fetch always-on rules
  6. Increment access_count / accessed_at for all returned entities
"""
import math
from datetime import UTC, datetime, timezone
from uuid import UUID

import psycopg2.extras

from braindb.config import settings
from braindb.schemas.search import ContextRequest, ContextResponse, SearchResultItem
from braindb.services.embedding_service import get_embedding_service
from braindb.services.graph import graph_expand
from braindb.services.keyword_service import find_entities_for_keywords, find_similar_keywords
from braindb.services.search import fuzzy_search

DECAY_RATES = {
    "thought":    settings.decay_rate_thought,
    "fact":       settings.decay_rate_fact,
    "source":     settings.decay_rate_source,
    "datasource": settings.decay_rate_datasource,
    "rule":       settings.decay_rate_rule,
}


def effective_importance(importance: float, created_at: datetime, access_count: int, entity_type: str) -> float:
    rate = DECAY_RATES.get(entity_type, 0.003)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    age_days = max(0, (datetime.now(UTC) - created_at).days)
    decay = math.exp(-rate * age_days)
    reinforce = 1.0 + 0.05 * math.log(1 + access_count)
    return min(1.0, importance * decay * reinforce)


# ------------------------------------------------------------------ #
# Extension fields                                                    #
# ------------------------------------------------------------------ #

EXT_QUERIES = {
    "thought":    ("thoughts_ext",    "entity_id, certainty, context, emotional_valence"),
    "fact":       ("facts_ext",       "entity_id, certainty, is_verified, source_entity_id"),
    "source":     ("sources_ext",     "entity_id, url, domain, http_status, last_checked_at"),
    "datasource": ("datasources_ext", "entity_id, file_path, url, content_hash, word_count, language"),
    "rule":       ("rules_ext",       "entity_id, always_on, category, priority, is_active"),
}


def fetch_ext(conn, rows: list[dict]) -> dict:
    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["entity_type"], []).append(str(r["id"]))

    ext_map = {}
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for etype, ids in by_type.items():
            if etype not in EXT_QUERIES:
                continue
            table, cols = EXT_QUERIES[etype]
            cur.execute(f"SELECT {cols} FROM {table} WHERE entity_id = ANY(%s::uuid[])", (ids,))
            for row in cur.fetchall():
                eid = row["entity_id"]
                ext_map[eid] = {k: v for k, v in row.items() if k != "entity_id"}
    return ext_map


# ------------------------------------------------------------------ #
# Always-on rules                                                     #
# ------------------------------------------------------------------ #

def fetch_always_on_rules(conn) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT e.*, r.always_on, r.category, r.priority, r.is_active
            FROM entities e
            JOIN rules_ext r ON r.entity_id = e.id
            WHERE r.always_on = TRUE AND r.is_active = TRUE
            ORDER BY r.priority DESC
            LIMIT %s
        """, (settings.max_always_on_rules,))
        return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------ #
# Access tracking                                                     #
# ------------------------------------------------------------------ #

def track_access(conn, ids: list) -> None:
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE entities SET access_count = access_count + 1, accessed_at = now() WHERE id = ANY(%s::uuid[])",
            ([str(i) for i in ids],),
        )


# ------------------------------------------------------------------ #
# Row → SearchResultItem                                              #
# ------------------------------------------------------------------ #

def _to_item(row: dict, search_score: float, depth: int, relevance: float, ext: dict) -> SearchResultItem:
    eff = effective_importance(row["importance"], row["created_at"], row["access_count"], row["entity_type"])
    return SearchResultItem(
        id=row["id"],
        entity_type=row["entity_type"],
        title=row.get("title"),
        content=row["content"],
        summary=row.get("summary"),
        keywords=row.get("keywords") or [],
        importance=row["importance"],
        source=row.get("source"),
        notes=row.get("notes"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        accessed_at=row.get("accessed_at"),
        access_count=row.get("access_count", 0),
        search_score=search_score,
        effective_importance=eff,
        depth=depth,
        accumulated_relevance=relevance,
        final_rank=search_score * eff * relevance,
        ext=ext,
    )


# ------------------------------------------------------------------ #
# Main context assembly                                               #
# ------------------------------------------------------------------ #

def assemble_context(conn, req: ContextRequest) -> ContextResponse:
    # Normalize to list of queries
    query_list = req.queries if req.queries else [req.query]

    # ------------------------------------------------------------------ #
    # 1. TEXT SEARCH (existing) — fuzzy + fulltext per query              #
    # ------------------------------------------------------------------ #
    text_scores: dict = {}       # entity_id → best text score
    seed_rows_by_id: dict = {}   # entity_id → row data

    for q in query_list:
        rows = fuzzy_search(conn, q, req.entity_types, req.min_importance, limit=max(req.max_results, 20))
        for r in rows:
            eid = r["id"]
            score = r["score"]
            if eid not in text_scores or score > text_scores[eid]:
                text_scores[eid] = score
                seed_rows_by_id[eid] = r

    # ------------------------------------------------------------------ #
    # 2. KEYWORD EMBEDDING SEARCH (new) — semantic via keyword vectors    #
    # ------------------------------------------------------------------ #
    embedding_scores: dict = {}  # entity_id → best keyword similarity
    embedding_rows: dict = {}    # entity_id → row data (for entities found only via embedding)

    emb_svc = get_embedding_service()
    if emb_svc.is_available():
        for q in query_list:
            query_emb = emb_svc.embed(q)
            if not query_emb:
                continue
            similar_kw = find_similar_keywords(conn, query_emb, limit=30)
            if not similar_kw:
                continue
            kw_sim = {str(kw["id"]): kw["similarity"] for kw in similar_kw}
            kw_ids = list(kw_sim.keys())
            entities = find_entities_for_keywords(conn, kw_ids)
            for ent in entities:
                eid = ent["id"]
                matched_ids = [str(mid) for mid in (ent.get("matched_keyword_ids") or [])]
                if matched_ids:
                    best_sim = max(kw_sim.get(mid, 0) for mid in matched_ids)
                    if eid not in embedding_scores or best_sim > embedding_scores[eid]:
                        embedding_scores[eid] = best_sim
                        if eid not in seed_rows_by_id:
                            embedding_rows[eid] = ent

    # ------------------------------------------------------------------ #
    # 3. MERGE — geometric mean when both, penalty when single signal     #
    # ------------------------------------------------------------------ #
    all_entity_ids = set(text_scores.keys()) | set(embedding_scores.keys())
    seed_scores: dict = {}
    penalty = settings.missing_signal_penalty
    for eid in all_entity_ids:
        text_s = text_scores.get(eid)
        emb_s = embedding_scores.get(eid)
        if text_s and emb_s:
            seed_scores[eid] = math.sqrt(text_s * emb_s)  # geometric mean — both agree
        elif text_s:
            seed_scores[eid] = text_s * penalty            # text only — penalized
        elif emb_s:
            seed_scores[eid] = emb_s * penalty             # embedding only — penalized
        # Ensure we have row data for embedding-only entities
        if eid not in seed_rows_by_id and eid in embedding_rows:
            seed_rows_by_id[eid] = embedding_rows[eid]

    # ------------------------------------------------------------------ #
    # 4. GRAPH EXPAND + RANK (existing pipeline)                          #
    # ------------------------------------------------------------------ #
    seed_ids = list(seed_scores.keys())

    if seed_ids and req.max_depth > 0:
        graph_rows = graph_expand(conn, seed_ids, req.max_depth, req.min_relevance)
    else:
        graph_rows = [{**r, "min_depth": 0, "relevance": 1.0} for r in seed_rows_by_id.values()]

    seen = {}
    for r in graph_rows:
        eid = r["id"]
        if eid not in seen or r.get("relevance", 1.0) > seen[eid].get("relevance", 0):
            seen[eid] = r

    # Filter out keyword entities from results — they're infrastructure, not content
    all_rows = [r for r in seen.values() if r.get("entity_type") != "keyword"]
    ext_map = fetch_ext(conn, all_rows)

    items = []
    for row in all_rows:
        eid = row["id"]
        score = seed_scores.get(eid, 0.3)
        depth = row.get("min_depth", 0)
        relevance = row.get("relevance", 1.0)
        items.append(_to_item(row, score, depth, relevance, ext_map.get(eid, {})))

    items.sort(key=lambda x: x.final_rank, reverse=True)
    items = items[: req.max_results]

    always_on = []
    if req.include_always_on_rules:
        rule_rows = fetch_always_on_rules(conn)
        rule_ext = fetch_ext(conn, rule_rows)
        for row in rule_rows:
            eid = row["id"]
            always_on.append(_to_item(row, 1.0, 0, 1.0, rule_ext.get(eid, {})))

    track_access(conn, [item.id for item in items] + [item.id for item in always_on])

    return ContextResponse(
        query=req.query or (req.queries[0] if req.queries else ""),
        queries=query_list,
        items=items,
        always_on_rules=always_on,
        total_found=len(items),
    )
