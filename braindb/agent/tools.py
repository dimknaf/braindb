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
from braindb.services.search import fuzzy_search

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 8000


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
async def recall_memory(queries: list[str], max_results: int = 15) -> str:
    """Search BrainDB memory with multiple natural language queries.
    Runs fuzzy + fulltext + keyword embedding search, merges with geometric mean,
    traverses the graph up to 3 hops, applies temporal decay.
    Use this as the primary recall tool.

    Args:
        queries: List of search queries (use multiple angles for better coverage).
        max_results: Max items to return (1-100, default 15).
    """
    try:
        req = ContextRequest(queries=queries, max_results=max_results)
        with get_conn() as conn:
            result = assemble_context(conn, req)
        lines = [f"Found {result.total_found} items:"]
        for item in result.items:
            lines.append(
                f"[{item.entity_type}] rank={item.final_rank:.3f} src={item.source or '-'}\n"
                f"  id: {item.id}\n"
                f"  content: {item.content}\n"
                f"  keywords: {', '.join(item.keywords)}"
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
async def get_entity(entity_id: str) -> str:
    """Fetch a single entity by ID (returns JSON blob with all fields).

    Args:
        entity_id: UUID of the entity.
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
        return _truncate(json.dumps(d, default=str, indent=2))
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
                f"  content: {r['content']}\n"
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

    IMPORTANT: `content` on a datasource is the original document body and is
    read-only via this tool. Any `content` passed for a datasource is dropped
    and the tool returns a warning. Use the `notes` field for analysis/summary.

    Args:
        entity_id: UUID of the entity.
        content: New content (ignored for datasources).
        keywords: New keywords list (replaces current).
        notes: New notes.
        importance: New importance 0-1.
    """
    try:
        # Datasource guardrail — look up type and strip content if protected.
        content_dropped = False
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
                row = cur.fetchone()
                if not row:
                    return _err(f"entity {entity_id} not found")
                entity_type = row[0]
        if content is not None and entity_type == "datasource":
            content = None
            content_dropped = True

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
                return "No changes (content ignored: datasource bodies are read-only; use notes for analysis)"
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
            msg += " (content ignored: datasource bodies are read-only; use notes for analysis)"
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
    relevance_score: float = 0.7,
    description: Optional[str] = None,
) -> str:
    """Create a relation between two entities.

    Args:
        from_entity_id: Source entity UUID.
        to_entity_id: Target entity UUID.
        relation_type: One of: supports, contradicts, elaborates, refers_to, derived_from, similar_to, is_example_of, challenges, tagged_with.
        relevance_score: 0-1 (default 0.7).
        description: Why this relation exists.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                try:
                    cur.execute(
                        """INSERT INTO relations (from_entity_id, to_entity_id, relation_type, relevance_score, description)
                           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                        (from_entity_id, to_entity_id, relation_type, relevance_score, description),
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
    """View the graph of connections around an entity (incoming + outgoing).

    Args:
        entity_id: UUID of the root entity.
        max_depth: How far to traverse (1-3, default 2).
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """SELECT e.*, r.relation_type, r.relevance_score, r.description AS rel_desc,
                              CASE WHEN r.from_entity_id = %s THEN 'out' ELSE 'in' END AS dir
                       FROM relations r
                       JOIN entities e ON e.id = CASE WHEN r.from_entity_id = %s THEN r.to_entity_id ELSE r.from_entity_id END
                       WHERE r.from_entity_id = %s OR r.to_entity_id = %s""",
                    (entity_id, entity_id, entity_id, entity_id),
                )
                rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return "No connections."
        lines = [f"{len(rows)} connections from {entity_id}:"]
        for r in rows:
            lines.append(
                f"  [{r['dir']}] {r['relation_type']} (rel={r['relevance_score']})\n"
                f"    [{r['entity_type']}] {r['content'][:80]} (id: {r['id']})"
            )
        return _truncate("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@function_tool
@_verbose("search_sql")
async def search_sql(query: str) -> str:
    """Run a read-only SQL query (SELECT/WITH only). Use for complex exploration.

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
        result = {"columns": columns, "rows": [[str(v) if v is not None else None for v in r] for r in rows], "row_count": len(rows)}
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

_call_depth = 0
_MAX_DEPTH = 1


@function_tool
@_verbose("delegate_to_subagent")
async def delegate_to_subagent(task: str) -> str:
    """Delegate a focused task to a fresh subagent running in its own context.
    Use for deep searches, duplicate-finding, relation work, or any task where
    you want the result without polluting your main context with intermediate
    tool outputs. The subagent has access to all the same BrainDB tools.

    Write a clear, self-contained task description — the subagent doesn't see
    your prior context. End by telling it to call submit_result with a summary.

    Args:
        task: A self-contained task description for the subagent.
    """
    global _call_depth
    if _call_depth >= _MAX_DEPTH:
        return "ERROR: max delegation depth reached. Do the task yourself."
    _call_depth += 1
    try:
        # Local imports to avoid circular dependency on agent.py
        from agents import Runner
        from braindb.agent.agent import create_braindb_agent
        from braindb.config import settings

        logger.info("Subagent starting: %s", task[:200])
        subagent = create_braindb_agent()
        result = await Runner.run(
            starting_agent=subagent,
            input=task,
            max_turns=settings.agent_subagent_max_turns,
        )
        answer = str(result.final_output)
        logger.info("Subagent completed.")
        return _truncate(answer)
    except Exception as e:
        logger.exception("Subagent failed")
        return _err(f"subagent failed: {e}")
    finally:
        _call_depth -= 1


# ====================================================================== #
# FINAL TOOL — stops the loop                                            #
# ====================================================================== #

@function_tool
@_verbose("submit_result")
async def submit_result(answer: str) -> str:
    """Submit the final answer to the query. Call this exactly once when you're done.

    Args:
        answer: The full response to send back to the caller.
    """
    return answer
