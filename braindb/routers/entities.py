"""
CRUD for all entity types.
"""
import hashlib
from pathlib import Path
from uuid import UUID

import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from braindb.db import get_conn
from braindb.schemas.entities import (
    DatasourceCreate, DatasourceRead, DatasourceUpdate,
    FactCreate, FactRead, FactUpdate,
    RuleCreate, RuleRead, RuleUpdate,
    SourceCreate, SourceRead, SourceUpdate,
    ThoughtCreate, ThoughtRead, ThoughtUpdate,
)
from braindb.services.activity_log import log_activity
from braindb.services.embedding_service import get_embedding_service
from braindb.services.keyword_service import ensure_keyword_entities, link_entity_to_keywords, sync_keywords_for_entity

INGEST_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
INGEST_ROOT = Path("/app")  # inside container; can absolute or repo-relative paths resolve


class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1)
    title: str | None = None
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    source: str = "document"
    notes: str | None = None

router = APIRouter(prefix="/api/v1/entities", tags=["entities"])


# ------------------------------------------------------------------ #
# Shared helper: fetch one entity with its extension fields           #
# ------------------------------------------------------------------ #

ENTITY_SELECT = """
    SELECT
        e.id, e.entity_type, e.title, e.content, e.summary,
        e.keywords, e.importance, e.source, e.notes, e.metadata,
        e.created_at, e.updated_at, e.accessed_at, e.access_count,
        -- thought
        te.certainty            AS certainty,
        te.context              AS context,
        te.emotional_valence    AS emotional_valence,
        -- fact
        fe.certainty            AS fact_certainty,
        fe.is_verified          AS is_verified,
        fe.source_entity_id     AS source_entity_id,
        -- source
        se.url                  AS url,
        se.domain               AS domain,
        se.http_status          AS http_status,
        se.last_checked_at      AS last_checked_at,
        -- datasource
        de.file_path            AS file_path,
        de.url                  AS ds_url,
        de.content_hash         AS content_hash,
        de.word_count           AS word_count,
        de.language             AS language,
        -- rule
        re.always_on            AS always_on,
        re.category             AS category,
        re.priority             AS priority,
        re.is_active            AS is_active
    FROM entities e
    LEFT JOIN thoughts_ext te    ON te.entity_id = e.id AND e.entity_type = 'thought'
    LEFT JOIN facts_ext fe       ON fe.entity_id = e.id AND e.entity_type = 'fact'
    LEFT JOIN sources_ext se     ON se.entity_id = e.id AND e.entity_type = 'source'
    LEFT JOIN datasources_ext de ON de.entity_id = e.id AND e.entity_type = 'datasource'
    LEFT JOIN rules_ext re       ON re.entity_id = e.id AND e.entity_type = 'rule'
    WHERE e.id = %s
"""


def _fetch(conn, entity_id) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(ENTITY_SELECT, (str(entity_id),))
        row = cur.fetchone()
        return dict(row) if row else None


def _or_404(row: dict | None) -> dict:
    if row is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return row


def _flatten(row: dict) -> dict:
    """Return only the fields relevant for the entity's type."""
    etype = row["entity_type"]
    base = {
        "id": row["id"], "entity_type": etype,
        "title": row["title"], "content": row["content"],
        "summary": row["summary"], "keywords": row["keywords"] or [],
        "importance": row["importance"], "source": row.get("source"),
        "notes": row["notes"], "metadata": row["metadata"] or {},
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "accessed_at": row["accessed_at"], "access_count": row["access_count"],
    }
    if etype == "thought":
        base.update(certainty=row["certainty"], context=row["context"], emotional_valence=row["emotional_valence"])
    elif etype == "fact":
        base.update(certainty=row["fact_certainty"], is_verified=row["is_verified"], source_entity_id=row["source_entity_id"])
    elif etype == "source":
        base.update(url=row["url"], domain=row["domain"], http_status=row["http_status"], last_checked_at=row["last_checked_at"])
    elif etype == "datasource":
        base.update(file_path=row["file_path"], url=row["ds_url"], content_hash=row["content_hash"], word_count=row["word_count"], language=row["language"])
    elif etype == "rule":
        base.update(always_on=row["always_on"], category=row["category"], priority=row["priority"], is_active=row["is_active"])
    return base


# ------------------------------------------------------------------ #
# CREATE                                                              #
# ------------------------------------------------------------------ #

def _insert_entity(conn, entity_type: str, body) -> str:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entities (entity_type, title, content, summary, keywords, importance, source, notes, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (entity_type, body.title, body.content, body.summary,
              body.keywords, body.importance, body.source, body.notes, psycopg2.extras.Json(body.metadata)))
        eid = cur.fetchone()[0]
    # Create keyword entities + tagged_with relations (transparent to caller)
    if body.keywords:
        kw_map = ensure_keyword_entities(conn, body.keywords, get_embedding_service())
        link_entity_to_keywords(conn, str(eid), list(kw_map.values()))
    log_activity(conn, "create", entity_type, eid, details={
        "content_preview": (body.content or "")[:100],
        "source": body.source,
        "importance": body.importance,
        "keywords": body.keywords,
    })
    return eid


@router.post("/thoughts", response_model=ThoughtRead, status_code=201)
def create_thought(body: ThoughtCreate):
    with get_conn() as conn:
        eid = _insert_entity(conn, "thought", body)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO thoughts_ext (entity_id, certainty, context, emotional_valence) VALUES (%s, %s, %s, %s)",
                (str(eid), body.certainty, body.context, body.emotional_valence),
            )
        return _flatten(_fetch(conn, eid))


@router.post("/facts", response_model=FactRead, status_code=201)
def create_fact(body: FactCreate):
    with get_conn() as conn:
        eid = _insert_entity(conn, "fact", body)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO facts_ext (entity_id, certainty, is_verified, source_entity_id) VALUES (%s, %s, %s, %s)",
                (str(eid), body.certainty, body.is_verified, str(body.source_entity_id) if body.source_entity_id else None),
            )
        return _flatten(_fetch(conn, eid))


@router.post("/sources", response_model=SourceRead, status_code=201)
def create_source(body: SourceCreate):
    with get_conn() as conn:
        eid = _insert_entity(conn, "source", body)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sources_ext (entity_id, url, domain, http_status) VALUES (%s, %s, %s, %s)",
                (str(eid), body.url, body.domain, body.http_status),
            )
        return _flatten(_fetch(conn, eid))


@router.post("/datasources", response_model=DatasourceRead, status_code=201)
def create_datasource(body: DatasourceCreate):
    with get_conn() as conn:
        eid = _insert_entity(conn, "datasource", body)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasources_ext (entity_id, file_path, url, content_hash, word_count, language) VALUES (%s, %s, %s, %s, %s, %s)",
                (str(eid), body.file_path, body.url, body.content_hash, body.word_count, body.language),
            )
        return _flatten(_fetch(conn, eid))


@router.post("/datasources/ingest", response_model=DatasourceRead, status_code=201)
def ingest_datasource(body: IngestRequest):
    """
    Read a file from disk, compute metadata, and create a datasource entity.
    The file_path is resolved relative to the container's working directory
    (which is mounted from the repo root).
    """
    # Resolve path (accept both absolute and repo-relative)
    path = Path(body.file_path)
    if not path.is_absolute():
        path = INGEST_ROOT / path

    if not path.exists():
        raise HTTPException(404, f"File not found: {body.file_path}")
    if not path.is_file():
        raise HTTPException(400, f"Not a file: {body.file_path}")

    size = path.stat().st_size
    if size > INGEST_MAX_BYTES:
        raise HTTPException(413, f"File too large: {size} bytes (max {INGEST_MAX_BYTES})")

    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    content_hash = hashlib.sha256(raw).hexdigest()
    word_count = len(text.split())
    title = body.title or path.name

    # Idempotency: if a datasource with this content_hash already exists,
    # return it with 200 (Not Created) so callers can distinguish new from dup.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entity_id FROM datasources_ext WHERE content_hash = %s LIMIT 1",
                (content_hash,),
            )
            existing = cur.fetchone()
        if existing:
            # Plain cursor returns tuples — index by position, not key.
            existing_id = existing[0]
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(_flatten(_fetch(conn, existing_id))),
            )

    # Build a DatasourceCreate-compatible body and insert
    class _Body:
        pass
    b = _Body()
    b.title = title
    b.content = text
    b.summary = None
    b.keywords = body.keywords
    b.importance = body.importance
    b.source = body.source
    b.notes = body.notes
    b.metadata = {}

    with get_conn() as conn:
        eid = _insert_entity(conn, "datasource", b)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasources_ext (entity_id, file_path, url, content_hash, word_count, language) VALUES (%s, %s, %s, %s, %s, %s)",
                (str(eid), str(body.file_path), None, content_hash, word_count, "en"),
            )
        log_activity(conn, "ingest", "datasource", eid, details={
            "file_path": str(body.file_path),
            "bytes": size,
            "word_count": word_count,
            "content_hash": content_hash[:16],
        })
        return _flatten(_fetch(conn, eid))


@router.post("/rules", response_model=RuleRead, status_code=201)
def create_rule(body: RuleCreate):
    with get_conn() as conn:
        eid = _insert_entity(conn, "rule", body)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO rules_ext (entity_id, always_on, category, priority, is_active) VALUES (%s, %s, %s, %s, %s)",
                (str(eid), body.always_on, body.category, body.priority, body.is_active),
            )
        return _flatten(_fetch(conn, eid))


# ------------------------------------------------------------------ #
# READ                                                                #
# ------------------------------------------------------------------ #

@router.get("/{entity_id}")
def get_entity(entity_id: UUID):
    with get_conn() as conn:
        return _flatten(_or_404(_fetch(conn, entity_id)))


# ------------------------------------------------------------------ #
# LIST                                                                #
# ------------------------------------------------------------------ #

@router.get("")
def list_entities(
    entity_type: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    source: str | None = Query(default=None),
    min_importance: float = Query(default=0.0, ge=0, le=1),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
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
    params += [limit, offset]

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT e.id, e.entity_type, e.title, e.content, e.summary,
                       e.keywords, e.importance, e.source, e.notes, e.metadata,
                       e.created_at, e.updated_at, e.accessed_at, e.access_count
                FROM entities e
                WHERE {where}
                ORDER BY e.importance DESC, e.created_at DESC
                LIMIT %s OFFSET %s
            """, params)
            return [dict(r) for r in cur.fetchall()]


# ------------------------------------------------------------------ #
# UPDATE                                                              #
# ------------------------------------------------------------------ #

def _update_base(conn, entity_id, data: dict):
    base_fields = {k: v for k, v in data.items()
                   if k in ("title", "content", "summary", "keywords", "importance", "source", "notes") and v is not None}
    if "metadata" in data and data["metadata"] is not None:
        base_fields["metadata"] = psycopg2.extras.Json(data["metadata"])
    if not base_fields:
        return
    sets = ", ".join(f"{k} = %s" for k in base_fields)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE entities SET {sets} WHERE id = %s", list(base_fields.values()) + [str(entity_id)])
    # Sync keyword entities + relations if keywords changed
    if "keywords" in base_fields:
        sync_keywords_for_entity(conn, str(entity_id), base_fields["keywords"], get_embedding_service())
    log_activity(conn, "update", None, entity_id, details={"fields": list(base_fields.keys())})


def _update_ext(conn, table: str, entity_id, fields: list[str], data: dict):
    ext_data = {k: v for k, v in data.items() if k in fields and v is not None}
    if not ext_data:
        return
    sets = ", ".join(f"{k} = %s" for k in ext_data)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {sets} WHERE entity_id = %s", list(ext_data.values()) + [str(entity_id)])


@router.patch("/thoughts/{entity_id}", response_model=ThoughtRead)
def update_thought(entity_id: UUID, body: ThoughtUpdate):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        if row["entity_type"] != "thought":
            raise HTTPException(400, "Entity is not a thought")
        data = body.model_dump(exclude_unset=True)
        _update_base(conn, entity_id, data)
        _update_ext(conn, "thoughts_ext", entity_id, ["certainty", "context", "emotional_valence"], data)
        return _flatten(_fetch(conn, entity_id))


@router.patch("/facts/{entity_id}", response_model=FactRead)
def update_fact(entity_id: UUID, body: FactUpdate):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        if row["entity_type"] != "fact":
            raise HTTPException(400, "Entity is not a fact")
        data = body.model_dump(exclude_unset=True)
        _update_base(conn, entity_id, data)
        _update_ext(conn, "facts_ext", entity_id, ["certainty", "is_verified", "source_entity_id"], data)
        return _flatten(_fetch(conn, entity_id))


@router.patch("/sources/{entity_id}", response_model=SourceRead)
def update_source(entity_id: UUID, body: SourceUpdate):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        if row["entity_type"] != "source":
            raise HTTPException(400, "Entity is not a source")
        data = body.model_dump(exclude_unset=True)
        _update_base(conn, entity_id, data)
        _update_ext(conn, "sources_ext", entity_id, ["url", "domain", "http_status"], data)
        return _flatten(_fetch(conn, entity_id))


@router.patch("/datasources/{entity_id}", response_model=DatasourceRead)
def update_datasource(entity_id: UUID, body: DatasourceUpdate):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        if row["entity_type"] != "datasource":
            raise HTTPException(400, "Entity is not a datasource")
        data = body.model_dump(exclude_unset=True)
        _update_base(conn, entity_id, data)
        _update_ext(conn, "datasources_ext", entity_id, ["file_path", "url", "content_hash", "word_count", "language"], data)
        return _flatten(_fetch(conn, entity_id))


@router.patch("/rules/{entity_id}", response_model=RuleRead)
def update_rule(entity_id: UUID, body: RuleUpdate):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        if row["entity_type"] != "rule":
            raise HTTPException(400, "Entity is not a rule")
        data = body.model_dump(exclude_unset=True)
        _update_base(conn, entity_id, data)
        _update_ext(conn, "rules_ext", entity_id, ["always_on", "category", "priority", "is_active"], data)
        return _flatten(_fetch(conn, entity_id))


# ------------------------------------------------------------------ #
# DELETE                                                              #
# ------------------------------------------------------------------ #

@router.delete("/{entity_id}", status_code=204)
def delete_entity(entity_id: UUID):
    with get_conn() as conn:
        row = _or_404(_fetch(conn, entity_id))
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entities WHERE id = %s", (str(entity_id),))
        log_activity(conn, "delete", row["entity_type"], entity_id)
