"""
Function tools for the BrainDB agent.

Each tool wraps a BrainDB operation. Tools call services directly
(same Python process, no HTTP hop). All tools return str — either a
compact human-readable summary or a short JSON blob. Errors are
returned as strings starting with "ERROR:".

When `settings.agent_verbose` is True, every tool logs its entry
(with args) and exit (with result preview) via the standard logger.
These logs go to stdout and are visible via `docker logs braindb_api`.

Follows the pattern in fa-automation/tasks/linkedin_research/tools.py.
"""
import functools
import json
import logging
import time
from contextvars import ContextVar
from typing import Optional
from uuid import UUID

import psycopg2.extras
from agents import function_tool

from braindb.config import settings
from braindb.db import get_conn
from braindb.schemas.search import ContextRequest
from braindb.services.activity_log import log_activity, query_log
from braindb.services.context import assemble_context, effective_importance, track_access
from braindb.services.embedding_service import get_embedding_service
from braindb.services.keyword_service import (
    ensure_keyword_entities,
    generate_missing_embeddings,
    link_entity_to_keywords,
    sync_keywords_for_entity,
)
from braindb.services.search import fuzzy_search, preview, slice_content
from braindb.services import wiki_sections as ws
from braindb.services import wiki_jobs as wj
from braindb.agent.run_state import record_handoff, record_submit
from braindb.agent.schemas import (
    AgentAnswer,
    MaintainerClusterDecision,
    MaintainerDecision,
    SubagentResult,
    WikiWriteResult,
)

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000

# Entity types whose BODY is not editable through the generic `update_entity`
# path, mapped to the redirect the model should follow instead. A wiki body is
# owned by the section tools: they carry the revision CAS, the pre-write
# snapshot and the `summarises` reconcile, and a bare overwrite here bypasses
# all three. It is also how a cited UUID loses a digit — retyping a 50k-char
# body by hand to change one line — after which every later write on that page
# fails on a dangling reference.
_CONTENT_READONLY = {
    "datasource": "datasource bodies are read-only; use notes for analysis",
    "wiki": 'wiki bodies are owned by the section tools; use '
            'edit_wiki_section(..., mode="append") to add, or mode="replace" '
            'to rewrite a section you have read in full',
}


def _truncate(s: str) -> str:
    if len(s) > MAX_OUTPUT_CHARS:
        return s[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
    return s


def _err(msg: str) -> str:
    logger.warning("Tool error: %s", msg)
    return f"ERROR: {msg}"


def _verbose(name: str):
    """Decorator that logs tool entry and exit when settings.agent_verbose is True.
    Placed BELOW @function_tool so the SDK still introspects the real signature.
    Uses inspect to bind positional + keyword args to parameter names.
    """
    import inspect as _inspect

    def decorator(fn):
        sig = _inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = None
            if settings.agent_verbose:
                # Bind positional args to param names for readable logging
                bound = {}
                for i, val in enumerate(args):
                    if i < len(param_names):
                        bound[param_names[i]] = val
                bound.update(kwargs)
                try:
                    args_preview = json.dumps(bound, default=str)[:500]
                except Exception:
                    args_preview = str(bound)[:500]
                logger.info("TOOL  %s  args=%s", name, args_preview)
                t0 = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
            except Exception as e:
                if settings.agent_verbose:
                    logger.error("TOOL! %s  exception=%s", name, e)
                raise
            if settings.agent_verbose and t0 is not None:
                elapsed = time.perf_counter() - t0
                preview = str(result)[:500].replace("\n", " | ")
                logger.info("TOOL  %s  elapsed=%.2fs  result=%s", name, elapsed, preview)
            return result
        return wrapper
    return decorator


# ====================================================================== #
# RECALL / SEARCH                                                        #
# ====================================================================== #

@function_tool
@_verbose("recall_memory")
async def recall_memory(
    queries: list[str],
    max_results: int = settings.recall_default_max_results,
) -> str:
    """⭐ Primary recall tool — use FIRST for ANY "what do we know about X" question.

    Runs fuzzy + fulltext + keyword embedding search, merges with geometric mean,
    traverses the graph up to 3 hops, applies temporal decay.

    QUERY STRATEGY — IMPORTANT for high-recall on narrow subjects:

    BrainDB indexes via short keyword entities. A 1-word query like
    "Petros" matches the keyword "Petros" cleanly (similarity ~1.0). A
    long phrase like "Petros person identity profile" matches the same
    keyword at much lower similarity (~0.4) because pg_trgm dilutes
    when comparing short keywords to long query strings.

    Therefore: prefer MULTIPLE narrow queries over one long phrase. The
    sweet spot for a focused subject is:
      - one or two SINGLE-KEYWORD queries (the names you care about),
      - plus 1-2 broader semantic phrases for adjacent context.

    Examples:
      GOOD:  ["Petros", "Selonda Saronikos fish farm", "Dimitrios manager"]
      BAD:   ["Petros person identity profile relation to Dimitris"]

    Each query you provide gets a reserved share of the top results
    (per-search-term quota), so adding the bare keyword as one of your
    queries GUARANTEES that subject surfaces — it doesn't compete with
    the broader phrases.

    Args:
        queries: List of search queries. Prefer 2-4 short focused queries
            over one long phrase. Include the bare keyword(s) of the
            subject you're investigating as standalone queries.
        max_results: Max items to return (1-100, default 30).
    """
    try:
        req = ContextRequest(queries=queries, max_results=max_results)
        with get_conn() as conn:
            result = assemble_context(conn, req)
        # EVERY item must survive into the rendered output — the per-query
        # quota guarantees membership, and a blind tail-truncate would undo
        # that guarantee for low-ranked reserved items (items are sorted by
        # final_rank, so reserved picks sit at the tail). So: budget the
        # content preview per item and shrink previews, never drop items.
        n = max(1, len(result.items))
        per_item_overhead = 130   # header + id + keywords lines, roughly
        content_budget = max(
            150, (MAX_OUTPUT_CHARS - 800) // n - per_item_overhead
        )

        def _clip(text: str, cap: int) -> str:
            s = text or ""
            if len(s) <= cap:
                return s
            return s[:cap] + f"...(+{len(s) - cap} chars; get_entity for full)"

        lines = [f"Found {result.total_found} items:"]
        for item in result.items:
            lines.append(
                f"[{item.entity_type}] rank={item.final_rank:.3f} src={item.source or '-'}\n"
                f"  id: {item.id}\n"
                f"  content: {_clip(item.content, content_budget)}\n"
                f"  keywords: {_clip(', '.join(item.keywords), 120)}"
            )
        for rule in result.always_on_rules:
            lines.append(f"[RULE priority={rule.ext.get('priority')}] {rule.content}")
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("quick_search")
async def quick_search(query: str, limit: int = 10) -> str:
    """Fast single-query search without graph traversal. Use for simple lookups.

    Args:
        query: Single search query.
        limit: Max results (default 10).
    """
    try:
        with get_conn() as conn:
            rows = fuzzy_search(conn, query, None, 0.0, limit)
            track_access(conn, [r["id"] for r in rows])
        lines = [f"Found {len(rows)} items:"]
        for r in rows:
            lines.append(
                f"[{r['entity_type']}] score={r['score']:.3f}\n"
                f"  id: {r['id']}\n"
                f"  content: {r['content']}"
            )
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# SAVE — creates                                                         #
# ====================================================================== #

def _insert_entity_raw(conn, entity_type: str, content: str, keywords: list[str],
                      source: Optional[str], importance: float, notes: Optional[str],
                      title: Optional[str] = None) -> str:
    """Shared helper for inserting a base entity + keyword entities + tagged_with relations."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entities (entity_type, title, content, keywords, importance, source, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (entity_type, title, content, keywords or [], importance, source, notes))
        eid = cur.fetchone()[0]

    if keywords:
        kw_map = ensure_keyword_entities(conn, keywords, get_embedding_service())
        link_entity_to_keywords(conn, str(eid), list(kw_map.values()))
    log_activity(conn, "create", entity_type, eid, details={
        "content_preview": (content or "")[:100],
        "source": source,
        "keywords": keywords,
    })
    return str(eid)


@function_tool
@_verbose("save_fact")
async def save_fact(
    content: str,
    keywords: list[str],
    source: str = "user-stated",
    certainty: float = 0.8,
    importance: float = 0.6,
    notes: Optional[str] = None,
) -> str:
    """Save an objective fact. Use for information the user stated, decisions, project info.

    Args:
        content: The fact (1-2 sentences, concise, standalone).
        keywords: Topic keywords (include full terms + abbreviations).
        source: Provenance. One of: user-stated, agent-inference, document, third-party.
        certainty: Confidence 0-1 (default 0.8).
        importance: Weight 0-1 (default 0.6).
        notes: Optional running commentary.
    """
    try:
        with get_conn() as conn:
            eid = _insert_entity_raw(conn, "fact", content, keywords, source, importance, notes)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO facts_ext (entity_id, certainty, is_verified) VALUES (%s, %s, FALSE)",
                    (eid, certainty),
                )
        return f"Saved fact id={eid}"
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("save_thought")
async def save_thought(
    content: str,
    keywords: list[str],
    source: str = "agent-inference",
    certainty: float = 0.6,
    context: Optional[str] = None,
    importance: float = 0.5,
) -> str:
    """Save a thought/inference (subjective, less certain than a fact).

    Args:
        content: The thought/inference.
        keywords: Topic keywords.
        source: Provenance (default agent-inference).
        certainty: Confidence 0-1 (default 0.6).
        context: What triggered this inference.
        importance: Weight 0-1 (default 0.5).
    """
    try:
        with get_conn() as conn:
            eid = _insert_entity_raw(conn, "thought", content, keywords, source, importance, None)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO thoughts_ext (entity_id, certainty, context, emotional_valence) VALUES (%s, %s, %s, 0.0)",
                    (eid, certainty, context),
                )
        return f"Saved thought id={eid}"
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("save_source")
async def save_source(
    content: str,
    url: str,
    keywords: list[str],
    importance: float = 0.5,
) -> str:
    """Save a URL bookmark (external link, web page).

    Args:
        content: Description of what the source contains.
        url: The URL.
        keywords: Topic keywords.
        importance: Weight 0-1 (default 0.5).
    """
    try:
        with get_conn() as conn:
            eid = _insert_entity_raw(conn, "source", content, keywords, "third-party", importance, None)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sources_ext (entity_id, url) VALUES (%s, %s)",
                    (eid, url),
                )
        return f"Saved source id={eid}"
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("save_rule")
async def save_rule(
    content: str,
    category: str = "behavior",
    priority: int = 50,
    always_on: bool = False,
    keywords: Optional[list[str]] = None,
    importance: float = 0.8,
) -> str:
    """Save a behavioral rule.

    Args:
        content: The rule text.
        category: One of behavior, ethics, personality, task, constraint.
        priority: 1-100 (default 50).
        always_on: If true, injected into every context call.
        keywords: Optional topic keywords.
        importance: Weight 0-1 (default 0.8).
    """
    try:
        with get_conn() as conn:
            eid = _insert_entity_raw(conn, "rule", content, keywords or [], "user-stated", importance, None)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO rules_ext (entity_id, always_on, category, priority, is_active) VALUES (%s, %s, %s, %s, TRUE)",
                    (eid, always_on, category, priority),
                )
        return f"Saved rule id={eid}"
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# ENTITIES — read / update / delete                                      #
# ====================================================================== #

@function_tool
@_verbose("get_entity")
async def get_entity(entity_id: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Fetch ONE entity by ID — the full-content read (recall/list only give
    previews; come here to read a thing fully).

    For a LARGE body, page it with offset/limit instead of pulling it whole:
    the response includes `content_meta` {total_chars, offset, returned,
    next_offset}. Loop `next_offset` until null. To avoid polluting your own
    context, hand each slice to `delegate_to_subagent` ("process THIS slice…")
    and aggregate — never load a huge document into your main context.

    Args:
        entity_id: UUID of the entity.
        offset: start char of the content slice (default 0).
        limit: max chars of this slice (clamped to the server slice max).
               If offset and limit are both omitted, the full body is returned
               (legacy behaviour, unchanged).
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
                row = cur.fetchone()
        if not row:
            return _err(f"entity {entity_id} not found")
        d = dict(row)
        d.pop("embedding", None)
        d.pop("search_vector", None)
        if offset == 0 and limit is None:
            return _truncate(json.dumps(d, default=str, indent=2))
        # Explicit slice request → return exactly that slice + paging meta,
        # NOT re-clipped by _truncate (slice is already bounded by SLICE_MAX).
        chunk, meta = slice_content(d.get("content"), offset, limit)
        d["content"] = chunk
        d["content_meta"] = meta
        return json.dumps(d, default=str, indent=2)
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("list_entities")
async def list_entities(
    entity_type: Optional[str] = None,
    keyword: Optional[str] = None,
    source: Optional[str] = None,
    min_importance: float = 0.0,
    limit: int = 30,
) -> str:
    """List/filter entities. Use to find candidates for relation creation or inspection.

    Args:
        entity_type: Filter by type (thought, fact, source, datasource, rule, keyword).
        keyword: Filter by keyword (exact match in TEXT[] column).
        source: Filter by provenance (user-stated, agent-inference, document, third-party).
        min_importance: Minimum importance (0-1).
        limit: Max items (default 30).
    """
    try:
        conditions = ["e.importance >= %s"]
        params: list = [min_importance]
        if entity_type:
            conditions.append("e.entity_type = %s")
            params.append(entity_type)
        if keyword:
            conditions.append("%s = ANY(e.keywords)")
            params.append(keyword)
        if source:
            conditions.append("e.source = %s")
            params.append(source)
        where = " AND ".join(conditions)
        params.append(limit)
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"""SELECT id, entity_type, content, keywords, importance, source
                        FROM entities e WHERE {where}
                        ORDER BY e.importance DESC, e.created_at DESC LIMIT %s""",
                    params,
                )
                rows = [dict(r) for r in cur.fetchall()]
        lines = [f"Found {len(rows)} entities:"]
        for r in rows:
            lines.append(
                f"[{r['entity_type']}] imp={r['importance']} src={r.get('source', '-')}\n"
                f"  id: {r['id']}\n"
                f"  content: {preview(r['content'], r['id'])}\n"
                f"  keywords: {r.get('keywords', [])}"
            )
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("update_entity")
async def update_entity(
    entity_id: str,
    content: Optional[str] = None,
    keywords: Optional[list[str]] = None,
    notes: Optional[str] = None,
    importance: Optional[float] = None,
) -> str:
    """Update an entity's mutable fields. Any unspecified field is left unchanged.

    IMPORTANT: `content` is read-only via this tool for a datasource (it is the
    original document body — use `notes` for analysis) and for a wiki (its body
    belongs to the section tools). Any `content` passed for those is dropped and
    the tool returns a warning; every other field still applies.

    Args:
        entity_id: UUID of the entity.
        content: New content (ignored for datasources and wikis).
        keywords: New keywords list (replaces current).
        notes: New notes.
        importance: New importance 0-1.
    """
    try:
        # Body guardrail — look up type and strip content if protected.
        content_dropped = ""
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return _err(f"entity {entity_id} not found")
                entity_type = row[0]
        if content is not None and entity_type in _CONTENT_READONLY:
            content = None
            content_dropped = _CONTENT_READONLY[entity_type]

        fields = {}
        if content is not None:
            fields["content"] = content
        if keywords is not None:
            fields["keywords"] = keywords
        if notes is not None:
            fields["notes"] = notes
        if importance is not None:
            fields["importance"] = importance
        if not fields:
            if content_dropped:
                return f"No changes (content ignored: {content_dropped})"
            return "No changes."
        sets = ", ".join(f"{k} = %s" for k in fields)
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE entities SET {sets} WHERE id = %s",
                    list(fields.values()) + [entity_id],
                )
            if keywords is not None:
                sync_keywords_for_entity(conn, entity_id, keywords, get_embedding_service())
            log_activity(conn, "update", None, entity_id, details={"fields": list(fields.keys())})
        msg = f"Updated entity {entity_id}"
        if content_dropped:
            msg += f" (content ignored: {content_dropped})"
        return msg
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("delete_entity")
async def delete_entity(entity_id: str) -> str:
    """Delete an entity. CASCADE removes its relations.

    Args:
        entity_id: UUID to delete.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return _err(f"entity {entity_id} not found")
                cur.execute("DELETE FROM entities WHERE id = %s", (entity_id,))
            log_activity(conn, "delete", row["entity_type"], entity_id)
        return f"Deleted entity {entity_id}"
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# RELATIONS                                                              #
# ====================================================================== #

@function_tool
@_verbose("create_relation")
async def create_relation(
    from_entity_id: str,
    to_entity_id: str,
    relation_type: str,
    relevance_score: float = 0.5,
    importance_score: float = 0.5,
    description: Optional[str] = None,
) -> str:
    """Create a relation between two entities.

    Args:
        from_entity_id: Source entity UUID.
        to_entity_id: Target entity UUID.
        relation_type: One of: supports, contradicts, elaborates, refers_to, derived_from, similar_to, is_example_of, challenges, tagged_with.
        relevance_score: 0-1 — how tight the semantic link is (0.9 = strong, 0.5 = neutral / "didn't judge", 0.2 = weak).
        importance_score: 0-1 — how much losing this edge would degrade recall (0.9 = critical, 0.5 = neutral / "didn't judge", 0.2 = trivial).
        description: Why this relation exists.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                try:
                    cur.execute(
                        """INSERT INTO relations (from_entity_id, to_entity_id, relation_type, relevance_score, importance_score, description)
                           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                        (from_entity_id, to_entity_id, relation_type, relevance_score, importance_score, description),
                    )
                    rid = cur.fetchone()["id"]
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    return _err(f"relation {relation_type} already exists between these entities")
            log_activity(conn, "create", "relation", rid, details={
                "from": from_entity_id, "to": to_entity_id, "type": relation_type,
            })
        return f"Created relation id={rid}"
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("view_entity_relations")
async def view_entity_relations(entity_id: str) -> str:
    """List all relations (incoming + outgoing) for a given entity.

    Args:
        entity_id: UUID of the entity.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT id, from_entity_id, to_entity_id, relation_type, relevance_score, description
                       FROM relations WHERE from_entity_id = %s OR to_entity_id = %s""",
                    (entity_id, entity_id),
                )
                rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return "No relations."
        lines = [f"{len(rows)} relations:"]
        for r in rows:
            direction = "->" if str(r["from_entity_id"]) == entity_id else "<-"
            other = r["to_entity_id"] if direction == "->" else r["from_entity_id"]
            lines.append(
                f"  {direction} {r['relation_type']} (rel={r['relevance_score']}) to {other}"
                + (f"\n     {r['description']}" if r.get("description") else "")
            )
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("delete_relation")
async def delete_relation(relation_id: str) -> str:
    """Delete a relation by its ID.

    Args:
        relation_id: UUID of the relation.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM relations WHERE id = %s RETURNING id", (relation_id,))
                if not cur.fetchone():
                    return _err(f"relation {relation_id} not found")
            log_activity(conn, "delete", "relation", relation_id)
        return f"Deleted relation {relation_id}"
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# EXPLORE                                                                #
# ====================================================================== #

@function_tool
@_verbose("view_tree")
async def view_tree(entity_id: str, max_depth: int = 2) -> str:
    """⭐ Reveals an entity's neighbourhood as a nested JSON tree:
    root keyed by ``entity_type``, ``children`` arrays per node, multi-path
    first-wins, keyword/retired-wiki noise filtered, ``_truncated`` marker
    when more remain. Especially useful when you have an entity ID (from a
    previous result) and want its graph context — often a sharper choice
    than another `recall_memory` about the same entity. Pass `max_depth=3`
    on hub entities (wikis with many connections) to see narrative chains.

    Args:
        entity_id: UUID of the root entity.
        max_depth: How far to traverse (1-3, default 2).
    """
    if max_depth < 1 or max_depth > 3:
        return _err("max_depth must be 1-3")
    try:
        from braindb.services.tree import build_entity_tree
        with get_conn() as conn:
            tree = build_entity_tree(conn, entity_id, max_depth=max_depth)
        if tree is None:
            return _err(f"Entity not found: {entity_id}")
        return _truncate(json.dumps(tree, indent=2, default=str, ensure_ascii=False))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("search_sql")
async def search_sql(query: str) -> str:
    """⚠ Aggregates ONLY (counts, GROUP BY, joins for stats). NEVER for recall /
    discovery / understanding — that's recall_memory. NEVER for "what's around
    this entity" — that's view_tree. If you're using SQL to find or understand
    something, stop and pick the right tool.

    Args:
        query: SQL query — must start with SELECT or WITH.
    """
    import re
    if not re.match(r"^\s*(SELECT|WITH)\s", query, re.IGNORECASE):
        return _err("Only SELECT/WITH allowed")
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '5s'")
                cur.execute("SET LOCAL transaction_read_only = on")
                cur.execute(query)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(1000)
            log_activity(conn, "sql_query", details={"query": query[:500], "rows": len(rows)})
        result = {"columns": columns,
                  "rows": [[preview(v) if v is not None else None for v in r] for r in rows],
                  "row_count": len(rows)}
        return _truncate(json.dumps(result, default=str, indent=2))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("view_log")
async def view_log(
    operation: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = 30,
) -> str:
    """View recent activity log — when and how things happened.

    Args:
        operation: Filter by operation (create, update, delete, search, context, ingest, sql_query, agent_query).
        entity_id: Filter by entity ID.
        limit: Max entries (default 30).
    """
    try:
        with get_conn() as conn:
            rows = query_log(conn, operation=operation, entity_id=entity_id, limit=limit)
        if not rows:
            return "No log entries."
        lines = [f"{len(rows)} log entries:"]
        for r in rows:
            lines.append(
                f"[{str(r['timestamp'])[:19]}] {r['operation']} {r.get('entity_type') or '-'} {str(r.get('entity_id') or '-')[:8]}"
                + (f"\n  details: {json.dumps(r.get('details', {}), default=str)[:150]}" if r.get("details") else "")
            )
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("get_stats")
async def get_stats() -> str:
    """Get database stats: entity counts, relations, recent activity."""
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT entity_type, COUNT(*) AS cnt FROM entities GROUP BY entity_type")
                counts = {r["entity_type"]: r["cnt"] for r in cur.fetchall()}
                cur.execute("SELECT COUNT(*) AS cnt FROM relations")
                rel_count = cur.fetchone()["cnt"]
        return _truncate(json.dumps({
            "entity_counts": counts,
            "total_entities": sum(counts.values()),
            "total_relations": rel_count,
        }, indent=2))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("generate_embeddings")
async def generate_embeddings() -> str:
    """Generate embeddings for keyword entities that don't have one yet."""
    try:
        emb = get_embedding_service()
        if not emb.is_available():
            return _err("embedding service not available")
        with get_conn() as conn:
            result = generate_missing_embeddings(conn, emb)
            log_activity(conn, "generate_embeddings", details=result)
        return json.dumps(result)
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# INGEST                                                                 #
# ====================================================================== #

@function_tool
@_verbose("ingest_file")
async def ingest_file(
    file_path: str,
    keywords: list[str],
    importance: float = 0.6,
) -> str:
    """Read a file from data/sources/ and store as a datasource entity.
    Paths are resolved relative to /app (repo root in the container).

    Args:
        file_path: Path relative to repo root, e.g. data/sources/article.md
        keywords: Topic keywords.
        importance: Weight 0-1 (default 0.6).
    """
    import hashlib
    from pathlib import Path

    try:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path("/app") / path
        if not path.exists() or not path.is_file():
            return _err(f"file not found: {file_path}")
        raw = path.read_bytes()
        if len(raw) > 5 * 1024 * 1024:
            return _err(f"file too large: {len(raw)} bytes (max 5MB)")
        text = raw.decode("utf-8", errors="replace")
        content_hash = hashlib.sha256(raw).hexdigest()
        word_count = len(text.split())

        with get_conn() as conn:
            eid = _insert_entity_raw(
                conn, "datasource", text, keywords, "document", importance, None,
                title=path.name,
            )
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO datasources_ext (entity_id, file_path, content_hash, word_count, language) VALUES (%s, %s, %s, %s, 'en')",
                    (eid, str(file_path), content_hash, word_count),
                )
            log_activity(conn, "ingest", "datasource", eid, details={
                "file_path": str(file_path),
                "bytes": len(raw),
                "word_count": word_count,
            })
        return f"Ingested {file_path} as datasource id={eid} (words={word_count})"
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# DELEGATION — spawn a subagent for focused work                         #
# ====================================================================== #

# Delegation depth, scoped to the current run context — NOT a module global.
#
# A plain global counts every delegation in flight across the PROCESS. With
# WIKI_WRITE_PARALLELISM writers plus a maintainer sharing one event loop,
# one agent's in-flight delegation made every OTHER agent's next delegation
# fail with "max delegation depth reached" — bounding BREADTH while trying to
# bound DEPTH. Observed live: two sibling delegations in a single parallel
# tool batch 1.8ms apart, the second rejected; 25 such refusals in 70h,
# including a writer's mandatory identity-resolution step, whose rejection
# message then told it to do the work inline.
#
# A ContextVar is inherited by the child Tasks the SDK runs tool bodies in,
# so a subagent spawned from here sees depth+1 and is correctly barred from
# delegating further, while a concurrent sibling run has its own context and
# is unaffected. Unlike run_state's submit slot this needs no mutable
# container: the value is only ever read DOWNWARD into nested runs, never
# written back up to the caller.
_depth_var: ContextVar[int] = ContextVar("braindb_delegation_depth", default=0)
_MAX_DEPTH = 1


@function_tool
@_verbose("delegate_to_subagent")
async def delegate_to_subagent(task: str) -> str:
    """Delegate a focused task to a fresh subagent running in its own context.
    Use for deep searches, duplicate-finding, relation work, or any task where
    you want the result without polluting your main context with intermediate
    tool outputs.

    The subagent gets the memory tools plus the wiki READ tools
    (`read_wiki_outline`, `read_wiki_section`, `check_members_cited`,
    `validate_wiki`). It CANNOT write: no `edit_wiki_section`, no
    `delete_wiki_section`, no handoff. Ask it to investigate, read and report
    — never to perform an edit, because it has no way to make one safely.

    Its answer comes back to you as ONE string, so ask for a distilled result,
    not a transcript. Write a clear, self-contained task description — the
    subagent doesn't see your prior context. End by telling it to call
    final_answer with a summary.

    Args:
        task: A self-contained task description for the subagent.
    """
    depth = _depth_var.get()
    if depth >= _MAX_DEPTH:
        return "ERROR: max delegation depth reached. Do the task yourself."
    token = _depth_var.set(depth + 1)
    try:
        # Local imports to avoid circular dependency on agent.py
        from braindb.agent.agent import get_subagent, run_typed
        from braindb.config import settings

        logger.info("Subagent starting: %s", task[:200])
        # run_typed isolates the subagent's submit slot from ours (its own
        # `last_submit.set(None)` token + reset in `finally`), so we cannot
        # leak the subagent's SubagentResult into the parent's run_typed.
        payload: SubagentResult = await run_typed(
            task,
            get_subagent(),
            SubagentResult,
            max_turns=settings.agent_subagent_max_turns,
        )
        logger.info("Subagent completed.")
        return _truncate(payload.result)
    except Exception as e:
        logger.exception("Subagent failed")
        return _err(f"subagent failed: {e}")
    finally:
        _depth_var.reset(token)


# ====================================================================== #
# WIKI SECTION EDITS — read/write slices of a wiki body (writer-only)    #
# ====================================================================== #
#
# Wiki bodies can grow past the writer's context window. These tools let
# the writer read just an outline (cheap) and edit one section at a time
# instead of re-emitting the whole markdown blob every turn. Wired into
# the writer agent only (see braindb/agent/agent.py).
#
# Strict-markers contract: tools error if the target body has no
# `<!-- section:X -->` markers. Phase 0 confirmed all active wikis
# already do.
#
# Optimistic concurrency via `wikis_ext.revision`: every read returns
# the current revision; every write requires it as `expect_revision`. A
# mismatch returns a "stale" ERROR string so the model re-reads instead
# of stomping a concurrent edit (or its own stale mental state).

import re as _re
_SECTION_NAME_RE = _re.compile(r"[A-Za-z0-9_\-]+")


@function_tool
@_verbose("read_wiki_outline")
async def read_wiki_outline(wiki_id: str) -> str:
    """Outline of a wiki — section names + char counts + current revision.
    Call before editing.

    Args:
        wiki_id: The wiki's entity UUID.
    """
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
        if fetched is None:
            return _err(f"wiki not found: {wiki_id}")
        body, revision = fetched
        _, sections = ws.parse_sections(body)
        if not sections:
            return _err(
                f"wiki {wiki_id} body has no <!-- section:X --> markers "
                f"(strict-markers contract violated; cannot edit)"
            )
        lines = [f"revision: {revision}", f"sections: {len(sections)}"]
        for s in sections:
            lines.append(f"  - {s.name}: {s.char_count}ch")
        return "\n".join(lines)
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("read_wiki_section")
async def read_wiki_section(
    wiki_id: str,
    section_name: str,
    offset: int = 0,
    limit: Optional[int] = None,
) -> str:
    """Read one section's content + the wiki's current revision token.

    A section bigger than one slice is PAGED, never silently cut: the reply
    carries `content_meta` {total_chars, offset, returned, next_offset}. Loop
    `next_offset` until it is null to hold the whole section — do that before
    any `mode="replace"` edit, or you will drop what you never read.

    To ADD to a section you are not restructuring, don't read it at all:
    `edit_wiki_section(..., mode="append")` keeps the existing content for you.

    Args:
        wiki_id: The wiki's entity UUID.
        section_name: Section name as listed by read_wiki_outline.
        offset: start char of the slice (default 0).
        limit: max chars of this slice (clamped to the server slice max).
    """
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
        if fetched is None:
            return _err(f"wiki not found: {wiki_id}")
        body, revision = fetched
        _, sections = ws.parse_sections(body)
        match = next((s for s in sections if s.name == section_name), None)
        if match is None:
            names = ", ".join(s.name for s in sections) or "(none)"
            return _err(f"section '{section_name}' not found. Existing: {names}")
        # Sliced, NOT _truncate'd: the slice is already bounded by SLICE_MAX,
        # and a bare truncation here would hide from the model that content
        # exists past the cap — which is precisely how a "preserve every prior
        # claim" replace turns into silent data loss on a large section.
        chunk, meta = slice_content(match.content, offset, limit)
        return (
            f"revision: {revision}\nsection: {match.name}\n"
            f"content_meta: {json.dumps(meta)}\ncontent:\n{chunk}"
        )
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("check_members_cited")
async def check_members_cited(wiki_id: str, entity_ids: list[str]) -> str:
    """Check which of these entity ids are ALREADY cited in the wiki body.

    One cheap call, exact answer. This is the SAME check the router runs when
    your run ends, so it tells you directly whether any work is left. If every
    assigned MEMBER comes back cited, the page already covers them: finish with
    `final_answer(mode="attach", body="")`. Do not re-read the page to satisfy
    yourself — that is what this tool is for.

    Args:
        wiki_id: The wiki's entity UUID.
        entity_ids: The ids to check — normally the MEMBERS of your job.
    """
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
        if fetched is None:
            return _err(f"wiki not found: {wiki_id}")
        body, revision = fetched
        cited = wj.parse_refs(body)  # lower-cased set
        present = [e for e in entity_ids if e.lower() in cited]
        missing = [e for e in entity_ids if e.lower() not in cited]
        return (
            f"revision: {revision}\n"
            f"cited: {len(present)}/{len(entity_ids)}\n"
            f"already_cited: {', '.join(present) or '(none)'}\n"
            f"NOT_cited: {', '.join(missing) or '(none)'}"
        )
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("edit_wiki_section")
async def edit_wiki_section(
    wiki_id: str,
    section_name: str,
    new_content: str,
    expect_revision: int,
    mode: str = "replace",
) -> str:
    """Replace a section, or append to it. If section_name is new, a fresh
    section is created at the end. Revision mismatch → returns ERROR:
    re-read first.

    Args:
        wiki_id: The wiki's entity UUID.
        section_name: Section to edit (or new section to create).
            Use lowercase letters, digits, dashes, underscores only.
        new_content: With mode="replace", the FULL new content of the
            section (without the marker line — the tool re-emits it). With
            mode="append", ONLY the text to add at the end; everything
            already in the section is kept for you, byte for byte.
        expect_revision: Revision token from the last read on this wiki.
        mode: "replace" (default) or "append". Prefer "append" to add a
            `[[ref:UUID]]` bullet or a new claim to a section you are not
            restructuring: you do NOT need to read the section first and
            nothing already in it can be lost. Use "replace" only when
            genuinely rewriting a section — and read it in full first.
    """
    if mode not in ("replace", "append"):
        return _err(f"invalid mode '{mode}': use 'replace' or 'append'")
    if not _SECTION_NAME_RE.fullmatch(section_name):
        return _err(
            f"invalid section_name '{section_name}': use only letters, "
            f"digits, dashes, underscores"
        )
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
            if fetched is None:
                return _err(f"wiki not found: {wiki_id}")
            body, current_rev = fetched
            if current_rev != expect_revision:
                return _err(
                    f"stale revision: you passed {expect_revision}, "
                    f"current is {current_rev}. Re-read the section first."
                )
            _, sections = ws.parse_sections(body)
            if not sections:
                return _err(
                    f"wiki {wiki_id} body has no <!-- section:X --> markers; "
                    f"strict-markers contract violated"
                )
            created = all(s.name != section_name for s in sections)
            if mode == "append":
                new_body = ws.append_to_section(body, section_name, new_content)
            else:
                new_body = ws.splice_section(body, section_name, new_content)
            new_rev = ws.apply_section_write(conn, wiki_id, new_body, expect_revision)
            log_activity(conn, "update", "wiki", wiki_id, details={
                "op": "edit_wiki_section",
                "section": section_name,
                "mode": mode,
                "appended": created,
                "revision": new_rev,
            })
        verb = "created" if created else ("appended to" if mode == "append"
                                          else "replaced")
        return f"ok — section '{section_name}' {verb}. new revision: {new_rev}"
    except ws.StaleRevisionError as e:
        return _err(str(e))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("delete_wiki_section")
async def delete_wiki_section(
    wiki_id: str,
    section_name: str,
    expect_revision: int,
) -> str:
    """Remove a section. Revision mismatch → ERROR: re-read first.

    Args:
        wiki_id: The wiki's entity UUID.
        section_name: Section to remove.
        expect_revision: Revision token from the last read on this wiki.
    """
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
            if fetched is None:
                return _err(f"wiki not found: {wiki_id}")
            body, current_rev = fetched
            if current_rev != expect_revision:
                return _err(
                    f"stale revision: you passed {expect_revision}, "
                    f"current is {current_rev}. Re-read first."
                )
            try:
                new_body = ws.delete_section(body, section_name)
            except KeyError:
                _, sections = ws.parse_sections(body)
                names = ", ".join(s.name for s in sections) or "(none)"
                return _err(f"section '{section_name}' not found. Existing: {names}")
            new_rev = ws.apply_section_write(conn, wiki_id, new_body, expect_revision)
            log_activity(conn, "update", "wiki", wiki_id, details={
                "op": "delete_wiki_section",
                "section": section_name,
                "revision": new_rev,
            })
        return f"ok — section '{section_name}' deleted. new revision: {new_rev}"
    except ws.StaleRevisionError as e:
        return _err(str(e))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("validate_wiki")
async def validate_wiki(wiki_id: str) -> str:
    """Check the wiki body grammar: section markers present, refs
    well-formed, summary callout present. Returns 'ok' or one issue per
    line.

    Args:
        wiki_id: The wiki's entity UUID.
    """
    try:
        with get_conn() as conn:
            fetched = ws.fetch_wiki_for_section_op(conn, wiki_id)
        if fetched is None:
            return _err(f"wiki not found: {wiki_id}")
        body, revision = fetched
        issues = ws.check_grammar(body)
        if not issues:
            return f"ok — revision: {revision}, no issues"
        return (
            f"revision: {revision}\nissues:\n"
            + "\n".join(f"  - {i}" for i in issues)
        )
    except Exception as e:
        return _err(str(e))


# ====================================================================== #
# CONTEXT HANDOFF — end this run, successor continues (writer-only)      #
# ====================================================================== #
#
# Called by the writer when it gets a context-near-full nudge from
# `CountdownHooks` and decides remaining work doesn't fit. The router's
# writer wrapper (braindb/routers/wiki.py) detects the handoff slot was
# filled and spawns a successor agent — same prompt, same tools, fresh
# context, seeded with the brief.
#
# The tool ALSO parks a placeholder `WikiWriteResult` via `record_submit`
# so `run_typed`'s typed-final contract is satisfied — the placeholder
# is never the authoritative output; the wrapper reads the handoff slot
# instead. This avoids any change to `run_typed`'s shape.

@function_tool
@_verbose("handoff_to_successor")
async def handoff_to_successor(progress_summary: str, remaining_work: str) -> str:
    """End this run early; a successor with the SAME prompt and tools
    will continue from your brief. Use when you've been nudged about
    context approaching the limit AND remaining work doesn't fit in 1-2
    turns.

    Args:
        progress_summary: Tools you've called, key findings, and any
            ACTIVE revision tokens (for the wiki you've been editing).
            The successor only sees this — be precise.
        remaining_work: The concrete next tool call(s) the successor
            must make — name wikis, section names, current revisions.
            Example: "Call read_wiki_section(wiki_id='abc', section_name='timeline')
            with expect_revision=15, then edit_wiki_section(...) with the
            new timeline content merging facts from member fact-id xyz."
    """
    record_handoff(progress_summary, remaining_work)
    # Park a placeholder WikiWriteResult so run_typed's typed-final
    # contract is satisfied. mode/body are intentionally minimal — the
    # router consults the handoff slot first when this run ends. The
    # writer's StopAtTools list includes `handoff_to_successor`, so
    # the loop halts cleanly after this returns.
    record_submit(WikiWriteResult(mode="attach", body=""))
    return "handoff registered; this run is ending — successor will continue from your brief"


# ====================================================================== #
# FINAL TOOL — stops the loop                                            #
# ====================================================================== #

# Convention (absolute): the run finishes ONLY by calling `final_answer`,
# and its argument is ALWAYS a typed Pydantic model — never a loose string.
# `@function_tool` validates the LLM's call args against the model BEFORE
# invoking the body, so `payload` is guaranteed-valid inside each function.
#
# strict_mode=False: critical. The default `strict_mode=True` activates
# OpenAI structured-outputs strict JSON schema, which forces EVERY
# property of the embedded Pydantic model into the schema's `required`
# list — overriding Pydantic's own view that fields with `= None` or
# `default_factory=...` are optional. On `MaintainerDecision` and
# `WikiWriteResult`, that inflation makes the LLM emit args that pass
# Pydantic but fail the over-strict schema, producing endless
# "Invalid JSON input: 1 validation error" loops the Layer 4 retry
# can't escape (verified live on deepinfra/Gemma against the wiki
# maintainer). Turning strict_mode off makes the LLM-visible schema
# match Pydantic's required list exactly; Pydantic still validates the
# parsed args inside the tool body, so the typed-final contract is
# unchanged — we just stop demanding the model emit fields it doesn't
# need.
# There is one typed variant per agent purpose; every variant keeps the
# name "final_answer" so prompts and `StopAtTools(["final_answer"])`
# stay generic.
#
# Each variant parks the validated payload into the per-Task ContextVar
# (see braindb/agent/run_state.py) so `run_typed` can hand it back
# typed. The returned "ok" string is irrelevant — we never read
# `result.final_output`; `StopAtTools` only needs the loop to stop.
#
# Why a ContextVar instead of `output_type=<Model>` on the Agent:
# `output_type` makes the SDK pass `response_format: json_schema` on
# EVERY LLM turn (not just the final one), which steers weaker models to
# satisfy the schema on turn 1 and never call tools. The side-channel
# capture keeps middle turns free while still delivering a typed final.

@function_tool(name_override="final_answer", strict_mode=False)
@_verbose("final_answer")
async def submit_answer(payload: AgentAnswer) -> str:
    """Submit the final answer. Call this exactly once when you're done."""
    record_submit(payload)
    return "ok"


@function_tool(name_override="final_answer", strict_mode=False)
@_verbose("final_answer")
async def submit_maintainer(payload: MaintainerClusterDecision) -> str:
    """Submit the maintainer decisions (one entry per seed). Call this exactly once when you're done."""
    record_submit(payload)
    return "ok"


@function_tool(name_override="final_answer", strict_mode=False)
@_verbose("final_answer")
async def submit_wiki(payload: WikiWriteResult) -> str:
    """Submit the finished wiki. Call this exactly once when you're done."""
    record_submit(payload)
    return "ok"


@function_tool(name_override="final_answer", strict_mode=False)
@_verbose("final_answer")
async def submit_subagent(payload: SubagentResult) -> str:
    """Submit the delegated task result. Call this exactly once when you're done."""
    record_submit(payload)
    return "ok"
