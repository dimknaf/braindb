"""
Activity log — append-only log of every operation on the memory database.
Karpathy-inspired: traceability of what happened, when, and in what context.

The log_activity function is fire-and-forget: it must never fail the main operation.
"""
import psycopg2.extras


def log_activity(
    conn,
    operation: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    details: dict | None = None,
    context_note: str | None = None,
) -> None:
    """Write an activity log entry. Swallows errors so it never breaks the caller."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO activity_log (operation, entity_type, entity_id, details, context_note)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    operation,
                    entity_type,
                    str(entity_id) if entity_id else None,
                    psycopg2.extras.Json(details or {}),
                    context_note,
                ),
            )
    except Exception:
        pass


def query_log(
    conn,
    operation: str | None = None,
    entity_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Query the activity log with optional filters."""
    conditions = []
    params: list = []

    if operation:
        conditions.append("operation = %s")
        params.append(operation)
    if entity_id:
        conditions.append("entity_id = %s")
        params.append(str(entity_id))
    if since:
        conditions.append("timestamp >= %s")
        params.append(since)
    if until:
        conditions.append("timestamp <= %s")
        params.append(until)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT id, timestamp, operation, entity_type, entity_id, details, context_note
            FROM activity_log
            {where}
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]
