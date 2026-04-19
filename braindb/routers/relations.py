"""
CRUD for relations.
"""
from uuid import UUID

import psycopg2.extras
from fastapi import APIRouter, HTTPException

from braindb.db import get_conn
from braindb.schemas.relations import RelationCreate, RelationRead, RelationUpdate
from braindb.services.activity_log import log_activity

router = APIRouter(tags=["relations"])

RELATION_SELECT = """
    SELECT id, from_entity_id, to_entity_id, relation_type,
           relevance_score, importance_score, is_bidirectional,
           description, notes, created_at, updated_at
    FROM relations
    WHERE id = %s
"""


def _fetch_relation(conn, relation_id) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(RELATION_SELECT, (str(relation_id),))
        row = cur.fetchone()
        return dict(row) if row else None


def _or_404(row: dict | None) -> dict:
    if row is None:
        raise HTTPException(status_code=404, detail="Relation not found")
    return row


@router.post("/api/v1/relations", response_model=RelationRead, status_code=201)
def create_relation(body: RelationCreate):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute("""
                    INSERT INTO relations
                        (from_entity_id, to_entity_id, relation_type,
                         relevance_score, importance_score, is_bidirectional,
                         description, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (str(body.from_entity_id), str(body.to_entity_id), body.relation_type,
                      body.relevance_score, body.importance_score, body.is_bidirectional,
                      body.description, body.notes))
                rid = cur.fetchone()["id"]
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                raise HTTPException(409, "Relation of this type already exists between these entities")
        log_activity(conn, "create", "relation", rid, details={
            "from_entity_id": str(body.from_entity_id),
            "to_entity_id": str(body.to_entity_id),
            "relation_type": body.relation_type,
            "description": body.description,
        })
        return _fetch_relation(conn, rid)


@router.get("/api/v1/relations/{relation_id}", response_model=RelationRead)
def get_relation(relation_id: UUID):
    with get_conn() as conn:
        return _or_404(_fetch_relation(conn, relation_id))


@router.patch("/api/v1/relations/{relation_id}", response_model=RelationRead)
def update_relation(relation_id: UUID, body: RelationUpdate):
    with get_conn() as conn:
        _or_404(_fetch_relation(conn, relation_id))
        data = body.model_dump(exclude_unset=True)
        if data:
            sets = ", ".join(f"{k} = %s" for k in data)
            with conn.cursor() as cur:
                cur.execute(f"UPDATE relations SET {sets} WHERE id = %s", list(data.values()) + [str(relation_id)])
            log_activity(conn, "update", "relation", relation_id, details={"fields": list(data.keys())})
        return _fetch_relation(conn, relation_id)


@router.delete("/api/v1/relations/{relation_id}", status_code=204)
def delete_relation(relation_id: UUID):
    with get_conn() as conn:
        _or_404(_fetch_relation(conn, relation_id))
        with conn.cursor() as cur:
            cur.execute("DELETE FROM relations WHERE id = %s", (str(relation_id),))
        log_activity(conn, "delete", "relation", relation_id)


@router.get("/api/v1/entities/{entity_id}/relations", response_model=list[RelationRead])
def entity_relations(entity_id: UUID):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, from_entity_id, to_entity_id, relation_type,
                       relevance_score, importance_score, is_bidirectional,
                       description, notes, created_at, updated_at
                FROM relations
                WHERE from_entity_id = %s OR to_entity_id = %s
            """, (str(entity_id), str(entity_id)))
            return [dict(r) for r in cur.fetchall()]
