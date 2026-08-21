"""
Wiki job queue — non-destructive plumbing only.

This module is deliberately free of any search/scoring/LLM logic (constraint
C3): finding *what* to wiki-ify and *how* to write it is delegated to the
existing recall/agent infra by the routers. Here we only:

  * detect orphans with one read-only SQL pass (no scoring),
  * enqueue exactly one `triage` job per orphan (idempotent),
  * list jobs.

Claim / status-transition / advisory-lock / accounted-change-gate helpers are
added in later steps, alongside the endpoints that use them.
"""
import os
import re
import uuid

import psycopg2.extras

ACTIVE_STATUSES = ("pending", "assigned")

# Freshness window: an entity is only orphan-eligible once it has existed for
# this many minutes, so the maintainer never wikis a subject whose ingest
# burst of facts/relations has not settled yet. Same env-var pattern the
# scheduler uses for its intervals (keeps this plumbing module config-import
# free). MUST be measured on created_at, never updated_at — the unconditional
# entities_updated_at BEFORE UPDATE trigger bumps updated_at on every recall
# access, which would leave recalled entities perpetually "fresh".
FRESHNESS_MINUTES = int(os.getenv("WIKI_FRESHNESS_MINUTES", "30"))

# Stale-lease (visibility-timeout) for claimed jobs. A job sits in `assigned`
# only while a worker is actively running it; if that worker never returns
# (api restart mid-run, agent timeout) the row would wedge forever. Instead
# of a reaper/cycle, an `assigned` job whose lease expired is simply
# claimable again at the EXISTING claim step. 20 min is comfortably above
# the longest legit run (AGENT_TIMEOUT ~10 min), so a still-running job is
# never reclaimed. `attempts`+max_attempts already bound repeated failures.
ASSIGNED_LEASE_MIN = int(os.getenv("WIKI_ASSIGNED_LEASE_MIN", "20"))

# Per-wiki attach grouping — how long to wait before firing a writer on a
# wiki that just received new attaches. Once the OLDEST pending attach for
# a wiki is this old, the writer claims ALL pending attaches for that wiki
# in a single batch (the within-tick batching at next_write_bucket()'s
# second query already groups by target_wiki_id). Lets attaches accumulate
# so the writer fires once per cooldown window instead of once per job —
# directly addresses the "Dimitrios Koutsoumpos rewritten 8x in an hour"
# pattern observed on Qwen. Bigger windows = lower LLM cost but slower
# wiki freshness. Default 300s (5 min). Set to 0 to disable (revert to
# the old "fire on every attach" behaviour). Self-limiting — no force-claim
# valve needed because the bucket scoops up the WHOLE pending queue for
# that wiki on each fire.
ATTACH_COOLDOWN_SEC = int(os.getenv("WIKI_ATTACH_COOLDOWN_SECONDS", "300"))

# Batched triage (clustering). When the maintainer runs, let it CLAIM several
# related triage jobs at once — orphan keywords that share ONE source fact (plus
# that fact's own job when the fact is also an orphan) — and decide them in a
# single LLM call. This changes only how the maintainer GROUPS its claim; it
# never touches run_cron, the per-job lifecycle, or the decision logic (each job
# still closes with its own status). Same os.getenv pattern as the knobs above
# (config-import free). Default on; the A/B harness flips it per run.
WIKI_TRIAGE_CLUSTER = os.getenv("WIKI_TRIAGE_CLUSTER", "true").lower() == "true"
# Max seeds (the shared source fact + its orphan keywords) decided in one
# maintainer call — caps prompt size and keeps the model focused; extras are
# left for a later claim.
WIKI_TRIAGE_CLUSTER_MAX = int(os.getenv("WIKI_TRIAGE_CLUSTER_MAX", "8"))
# A keyword tagged on >= this many entities is a hub (it spans many distinct
# subjects, e.g. "CityFalcon") — triaged alone, never used to pull in a cluster.
WIKI_TRIAGE_HUB_DEGREE = int(os.getenv("WIKI_TRIAGE_HUB_DEGREE", "9"))


def _claimable(alias: str = "") -> str:
    """SQL predicate: a job is claimable if pending, OR assigned but its
    lease expired. Reused verbatim at every claim site (DRY). `alias` is the
    table alias when the query qualifies columns (e.g. 'j')."""
    p = f"{alias}." if alias else ""
    return (f"({p}status = 'pending' OR ({p}status = 'assigned' "
            f"AND {p}assigned_at < now() - make_interval(mins => {ASSIGNED_LEASE_MIN})))")

# Inline reference token: [[ref:UUID]] or [[ref:UUID|display text]]
REF_RE = re.compile(
    r"\[\[ref:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:\|[^\]]*)?\]\]"
)

SUMMARY_RE = re.compile(r">\s*\*\*Summary:\*\*\s*(.+)")
DISAMBIG_RE = re.compile(r">\s*\*\*Disambiguation:\*\*\s*(.+)")
# The LLM authors its own meta header line; we only READ what it declared.
META_KEYWORDS_RE = re.compile(r"<!--\s*wiki:meta[^>]*\bkeywords=([^>]+?)\s*-->")


def parse_refs(body: str) -> set[str]:
    """All entity UUIDs cited inline in the body (lower-cased)."""
    return {m.lower() for m in REF_RE.findall(body or "")}


def keywords_from_meta(body: str) -> list[str]:
    """Read keywords the LLM declared in its own `<!-- wiki:meta ... -->`
    header (e.g. `keywords=a;b;c`). Reading the LLM's declaration is not code
    authoring content. Returns [] if the LLM declared none."""
    m = META_KEYWORDS_RE.search(body or "")
    if not m:
        return []
    raw = m.group(1).replace(",", ";")
    return [k.strip() for k in raw.split(";") if k.strip()]


def snapshot_revision(conn, wiki_id: str, old_content: str, old_refs: set[str],
                      revision: int) -> None:
    """Persist the prior body+refs before mutation so any change is reversible."""
    from braindb.services.activity_log import log_activity
    log_activity(conn, "wiki_revise", "wiki", wiki_id, details={
        "from_revision": revision,
        "prior_content": old_content,
        "prior_refs": sorted(old_refs),
    })


def reconcile_summarises_additive(conn, wiki_id: str, body: str) -> dict:
    """
    Pure bookkeeping: ensure a `wiki --summarises--> e` relation exists for
    every entity the LLM cited inline (`[[ref:UUID]]`). ADDITIVE ONLY — it
    never deletes or re-types a relation behind the LLM. If the LLM wants a
    relation gone it calls `delete_relation` itself. Mirrors LLM-authored
    content into the graph; it does not judge or shape content.
    """
    cited = parse_refs(body)
    added = 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_entity_id::text FROM relations "
            "WHERE from_entity_id = %s AND relation_type = 'summarises'",
            (wiki_id,),
        )
        current = {r[0].lower() for r in cur.fetchall()}
        for e in cited - current:
            cur.execute(
                """INSERT INTO relations
                   (from_entity_id, to_entity_id, relation_type, relevance_score, description)
                   VALUES (%s, %s, 'summarises', 0.9, 'wiki body reference')
                   ON CONFLICT (from_entity_id, to_entity_id, relation_type) DO NOTHING""",
                (wiki_id, e),
            )
            added += 1
    return {"relations_added": added, "relations_removed": 0}


def try_wiki_lock(conn, key: str) -> bool:
    """Transaction-scoped advisory lock so two writers never touch one wiki."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_xact_lock(hashtext(%s))", (f"wiki:{key}",))
        return bool(cur.fetchone()[0])


def claim_jobs(conn, job_ids: list[str]) -> int:
    """Mark a bucket's pending suggestion jobs as assigned (SKIP LOCKED)."""
    if not job_ids:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            f"""UPDATE wiki_job SET status='assigned', assigned_at=now(), attempts=attempts+1
               WHERE id = ANY(%s::uuid[]) AND {_claimable()}
                 AND id IN (SELECT id FROM wiki_job WHERE id = ANY(%s::uuid[])
                            FOR UPDATE SKIP LOCKED)""",
            (job_ids, job_ids),
        )
        return cur.rowcount

# Entity types the cron considers "wiki-able" content. Keywords act as concept
# hubs; thoughts/facts are the substance. (wiki/source/datasource/rule excluded.)
ORPHAN_ENTITY_TYPES = ("keyword", "thought", "fact")


def _orphan_conditions(exclude_job: bool = False) -> str:
    """
    The SINGLE definition of "orphan" (entity not yet covered by a wiki),
    shared by `run_cron` (set-based) and `is_orphan` (per-entity) so the two
    can never drift. References the entity as `e.id`. All conditions are
    param-free EXCEPT the optional `exclude_job` clause (one %s) used by the
    maintainer staleness guard to ignore the just-claimed triage row itself.

    An orphan is an entity that:
      * has settled — `created_at` is older than FRESHNESS_MINUTES (so a
        still-ingesting subject is not wikied half-formed),
      * is not the target of a `wiki --summarises--> e` relation,
      * is not listed in any wiki's `member_keyword_ids`,
      * is not referenced by an active (pending/assigned) wiki_job,
      * does not carry a `rejected` triage (deliberate-skip self-clearing;
        `failed` triage is NOT excluded so transient errors still retry).
    """
    xj = " AND j.id <> %s" if exclude_job else ""
    return f"""
        e.created_at < now() - make_interval(mins => {FRESHNESS_MINUTES})
        AND NOT EXISTS (
            SELECT 1 FROM relations r
            JOIN entities w ON w.id = r.from_entity_id AND w.entity_type = 'wiki'
            WHERE r.relation_type = 'summarises' AND r.to_entity_id = e.id
        )
        AND NOT EXISTS (
            SELECT 1 FROM wikis_ext wx WHERE e.id = ANY(wx.member_keyword_ids)
        )
        AND NOT EXISTS (
            SELECT 1 FROM wiki_job j
            WHERE j.status IN ('pending','assigned')
              AND e.id = ANY(j.entity_ids){xj}
        )
        AND NOT EXISTS (
            SELECT 1 FROM wiki_job j
            WHERE j.job_type = 'triage' AND j.status = 'rejected'
              AND e.id = ANY(j.entity_ids)
        )
    """


def is_orphan(conn, entity_id, exclude_triage_job_id: str | None = None) -> bool:
    """True if the entity is still uncovered by any wiki. Used by the
    maintainer staleness guard: if a prior writer run already absorbed/linked
    the entity (or it is already in an active suggestion), this returns False
    and the maintainer skips it with NO LLM call. Same predicate as cron."""
    cond = _orphan_conditions(exclude_job=exclude_triage_job_id is not None)
    params: list = [str(entity_id)]
    if exclude_triage_job_id is not None:
        params.append(str(exclude_triage_job_id))
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT EXISTS (SELECT 1 FROM entities e WHERE e.id = %s AND {cond})",
            params,
        )
        return bool(cur.fetchone()[0])


def run_cron(conn) -> dict:
    """
    Find entities not yet connected to any wiki and enqueue one `triage`
    job per orphan. Pure SQL, read-only except the additive job insert.
    Orphan-ness is the shared `_orphan_conditions()` (see there).

    Idempotent: the partial-unique index on `dedupe_key WHERE status IN
    ('pending','assigned')` + ON CONFLICT DO NOTHING means re-running cron
    never creates duplicate triage jobs.
    """
    batch_id = str(uuid.uuid4())
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            WITH orphans AS (
                SELECT e.id
                FROM entities e
                WHERE e.entity_type = ANY(%s)
                  AND {_orphan_conditions()}
            )
            INSERT INTO wiki_job (job_type, status, entity_ids, dedupe_key, batch_id)
            SELECT 'triage', 'pending', ARRAY[o.id], 'triage:' || o.id::text, %s::uuid
            FROM orphans o
            ON CONFLICT (dedupe_key) WHERE status IN ('pending','assigned')
            DO NOTHING
            RETURNING id
            """,
            (list(ORPHAN_ENTITY_TYPES), batch_id),
        )
        enqueued = cur.rowcount

        # Counts for visibility (cheap; the heavy filter already ran above).
        cur.execute(
            "SELECT count(*) AS c FROM wiki_job WHERE status = 'pending' AND job_type = 'triage'"
        )
        pending_triage = cur.fetchone()["c"]

    return {
        "batch_id": batch_id,
        "triage_jobs_enqueued": enqueued,
        "pending_triage_total": pending_triage,
    }


def claim_one_triage(conn) -> dict | None:
    """
    Atomically claim a single pending triage job (C1: one case per call).
    FOR UPDATE SKIP LOCKED guarantees two concurrent maintainer calls never
    grab the same case. Highest-importance orphan first, so high-value
    concepts get wikis early and their writer runs absorb neighbourhoods
    (more downstream triage becomes free stale-skips).
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            UPDATE wiki_job
               SET status = 'assigned', assigned_at = now(), attempts = attempts + 1
             WHERE id = (
                 SELECT j.id FROM wiki_job j
                  JOIN entities e ON e.id = j.entity_ids[1]
                  WHERE {_claimable("j")} AND j.job_type = 'triage'
                  ORDER BY e.importance DESC, j.created_at
                  FOR UPDATE OF j SKIP LOCKED
                  LIMIT 1
             )
            RETURNING id, entity_ids::text[] AS entity_ids, batch_id
            """
        )
        row = cur.fetchone()
        return dict(row) if row else None


def claim_related_triage(conn, primary_entity_id: str, exclude_job_id: str,
                         limit: int) -> list[dict]:
    """
    Claim up to `limit` other claimable triage jobs whose entity shares a source
    fact with `primary_entity_id` — its non-hub sibling keywords (tagged on the
    same fact) and the source fact(s)' own jobs — so a fact and its keywords are
    decided in ONE maintainer call. `FOR UPDATE SKIP LOCKED`, importance first.
    Each returned job is still single-entity and is closed individually by the
    caller (per-job lifecycle unchanged). Returns [] if `limit` <= 0 or the
    primary is itself a hub (a hub spans many subjects, so it is triaged alone).
    """
    if limit <= 0:
        return []
    pid = str(primary_entity_id)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # A hub primary would group too many unrelated subjects → triage it alone.
        cur.execute(
            "SELECT count(*) AS c FROM relations "
            "WHERE relation_type='tagged_with' AND (from_entity_id=%s OR to_entity_id=%s)",
            (pid, pid),
        )
        if cur.fetchone()["c"] >= WIKI_TRIAGE_HUB_DEGREE:
            return []
        cur.execute(
            f"""
            WITH src AS (
                -- the primary keyword's source facts/thoughts (tagged_with, undirected)
                SELECT (CASE WHEN r.from_entity_id = %(pid)s THEN r.to_entity_id
                             ELSE r.from_entity_id END) AS fid
                FROM relations r
                JOIN entities f ON f.id = (CASE WHEN r.from_entity_id = %(pid)s
                                               THEN r.to_entity_id ELSE r.from_entity_id END)
                                AND f.entity_type IN ('fact','thought')
                WHERE r.relation_type = 'tagged_with'
                  AND (r.from_entity_id = %(pid)s OR r.to_entity_id = %(pid)s)
            ),
            cand AS (
                SELECT j.id
                FROM wiki_job j
                JOIN entities e ON e.id = j.entity_ids[1]
                WHERE {_claimable("j")} AND j.job_type = 'triage' AND j.id <> %(xjob)s
                  AND (
                      e.id IN (SELECT fid FROM src)               -- a source fact's own job
                      OR (
                          e.entity_type = 'keyword'               -- a non-hub sibling keyword
                          AND (SELECT count(*) FROM relations rk
                               WHERE rk.relation_type = 'tagged_with'
                                 AND (rk.from_entity_id = e.id OR rk.to_entity_id = e.id)
                              ) < %(hub)s
                          AND EXISTS (
                              SELECT 1 FROM relations r2
                              WHERE r2.relation_type = 'tagged_with'
                                AND ((r2.from_entity_id = e.id AND r2.to_entity_id IN (SELECT fid FROM src))
                                  OR (r2.to_entity_id = e.id AND r2.from_entity_id IN (SELECT fid FROM src)))
                          )
                      )
                  )
                ORDER BY e.importance DESC, j.created_at
                FOR UPDATE OF j SKIP LOCKED
                LIMIT %(limit)s
            )
            UPDATE wiki_job
               SET status = 'assigned', assigned_at = now(), attempts = attempts + 1
             WHERE id IN (SELECT id FROM cand)
            RETURNING id, entity_ids::text[] AS entity_ids, batch_id
            """,
            {"pid": pid, "xjob": str(exclude_job_id),
             "hub": WIKI_TRIAGE_HUB_DEGREE, "limit": limit},
        )
        return [dict(r) for r in cur.fetchall()]


def finish_job(conn, job_id: str, status: str, last_error: str | None = None) -> None:
    """Transition a job to a terminal state (done / rejected / failed)."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE wiki_job SET status = %s, completed_at = now(), last_error = %s WHERE id = %s",
            (status, last_error, str(job_id)),
        )


def fetch_entity_brief(conn, entity_id: str) -> dict | None:
    """Minimal entity view for building a focused maintainer prompt."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, entity_type, content, summary, keywords FROM entities WHERE id = %s",
            (str(entity_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def suggestion_dedupe_key(action: str, target_wiki_id: str | None,
                          entity_ids: list[str], consolidate_wiki_ids: list[str]) -> str:
    """Deterministic, service-computed (never LLM-computed) idempotency key."""
    if action == "attach":
        return f"attach:{target_wiki_id}:" + ",".join(sorted(entity_ids))
    if action == "create":
        return "create:" + ",".join(sorted(entity_ids))
    if action == "consolidate":
        return "consolidate:" + ",".join(sorted(consolidate_wiki_ids))
    raise ValueError(f"unknown action {action!r}")


def insert_suggestion(conn, *, job_type: str, target_wiki_id: str | None,
                      entity_ids: list[str], dedupe_key: str, rationale: str | None,
                      proposed_name: str | None, batch_id: str | None) -> str | None:
    """
    Insert a maintainer suggestion job. ON CONFLICT DO NOTHING against the
    partial-unique active dedupe index → re-proposing the same work is a no-op.
    Returns the new job id, or None if it was a duplicate.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO wiki_job
                (job_type, status, target_wiki_id, entity_ids, dedupe_key,
                 rationale, proposed_name, batch_id)
            VALUES (%s, 'pending', %s, %s::uuid[], %s, %s, %s, %s)
            ON CONFLICT (dedupe_key) WHERE status IN ('pending','assigned')
            DO NOTHING
            RETURNING id
            """,
            (job_type, target_wiki_id, entity_ids, dedupe_key,
             rationale, proposed_name, batch_id),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None


def next_write_bucket(conn) -> dict | None:
    """
    Pick the next unit of writer work (one wiki per call). A `create` job is
    its own bucket; `attach` jobs are grouped by target_wiki_id so the writer
    sees every new member of a wiki at once. Consolidate is handled by Step 5.

    Dedup-first priority: pending jobs are ordered consolidate -> attach ->
    create (then created_at). The moment the maintainer emits a `consolidate`
    the writer drains it before creating/expanding more pages, so the wiki
    set converges before it grows.

    Per-wiki cooldown on attach (ATTACH_COOLDOWN_SEC, default 300s = 5 min):
    an attach seed becomes claimable only once its target wiki's oldest
    pending attach is past the cooldown. Once eligible, the existing second
    query below scoops up ALL pending attaches for that wiki (including the
    ones inserted during the cooldown window) into one writer call. Net
    effect: writer fires once per cooldown window per wiki instead of once
    per job. `consolidate` and `create` paths are unaffected; the cooldown
    is attach-only because attach is the only multi-job-per-wiki shape.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""SELECT id, job_type, target_wiki_id, entity_ids::text[] AS entity_ids,
                      proposed_name, rationale, batch_id
               FROM wiki_job j
               WHERE {_claimable("j")}
                 AND job_type IN ('create','attach','consolidate')
                 AND (
                     job_type <> 'attach'
                     OR (
                         SELECT MIN(created_at)
                         FROM wiki_job
                         WHERE target_wiki_id = j.target_wiki_id
                           AND job_type = 'attach'
                           AND status = 'pending'
                     ) <= now() - make_interval(secs => {ATTACH_COOLDOWN_SEC})
                 )
               ORDER BY CASE job_type WHEN 'consolidate' THEN 0
                                      WHEN 'attach'      THEN 1
                                      ELSE 2 END,
                        created_at
               LIMIT 1"""
        )
        seed = cur.fetchone()
        if not seed:
            return None
        seed = dict(seed)
        if seed["job_type"] == "create":
            return {"mode": "create", "jobs": [seed],
                    "target_wiki_id": None, "proposed_name": seed["proposed_name"]}
        if seed["job_type"] == "consolidate":
            # entity_ids holds the wiki ids the maintainer flagged as duplicates.
            return {"mode": "consolidate", "jobs": [seed],
                    "target_wiki_id": None, "proposed_name": None,
                    "wiki_ids": seed["entity_ids"]}
        cur.execute(
            f"""SELECT id, entity_ids::text[] AS entity_ids
               FROM wiki_job
               WHERE {_claimable()} AND job_type='attach'
                 AND target_wiki_id = %s
               ORDER BY created_at""",
            (seed["target_wiki_id"],),
        )
        jobs = [dict(r) for r in cur.fetchall()]
        return {"mode": "attach", "jobs": jobs,
                "target_wiki_id": str(seed["target_wiki_id"]), "proposed_name": None}


def fetch_members(conn, entity_ids: list[str]) -> list[dict]:
    if not entity_ids:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id::text AS id, entity_type, content, keywords "
            "FROM entities WHERE id = ANY(%s::uuid[])",
            (entity_ids,),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_wiki(conn, wiki_id: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT e.id::text AS id, e.content, w.canonical_name, w.revision,
                      w.member_keyword_ids::text[] AS member_keyword_ids
               FROM entities e JOIN wikis_ext w ON w.entity_id = e.id
               WHERE e.id = %s""",
            (str(wiki_id),),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def list_active_wikis(conn) -> list[dict]:
    """All non-retired wikis as {id, canonical_name, char_count},
    deterministically ordered. Plumbing read (mirrors fetch_wiki /
    export_wikis SQL) — the maintainer is shown this as a NUMBERED catalog so
    it references wikis by number, never by uuid; the order here IS the
    numbering. `char_count` lets the maintainer see that a candidate target is
    already large and prefer a narrower `create` over piling on another
    `attach`; without it every page looks equally empty."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT e.id::text AS id, w.canonical_name,
                      length(e.content) AS char_count
               FROM entities e JOIN wikis_ext w ON w.entity_id = e.id
               WHERE e.entity_type = 'wiki' AND w.retired_at IS NULL
               ORDER BY e.importance DESC, e.created_at"""
        )
        return [dict(r) for r in cur.fetchall()]


def list_related_wikis(conn, entity_ids: list[str], exclude_wiki_id: str | None,
                       limit: int = 10) -> list[dict]:
    """Wikis one hop from the entities being written, as
    {canonical_name, char_count}, most-cited first.

    The WRITER gets this — deliberately NOT the full catalog. It only needs to
    know which neighbouring pages already exist so it can name one in prose
    instead of expanding this page. A short, relevant list keeps that cheap;
    the whole catalog would be noise and grows without bound.

    "Related" = a wiki that already `summarises` an entity that one of these
    members is connected to. Reuses the relations the pipeline already
    maintains; creates nothing."""
    if not entity_ids:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT w.canonical_name,
                      length(e.content) AS char_count, count(*) AS cites
               FROM relations seed
               JOIN relations sm ON sm.to_entity_id IN (seed.from_entity_id,
                                                        seed.to_entity_id)
                                AND sm.relation_type = 'summarises'
               JOIN entities e ON e.id = sm.from_entity_id
               JOIN wikis_ext w ON w.entity_id = e.id
               WHERE (seed.from_entity_id = ANY(%s::uuid[])
                      OR seed.to_entity_id = ANY(%s::uuid[]))
                 AND e.entity_type = 'wiki' AND w.retired_at IS NULL
                 AND (%s::uuid IS NULL OR e.id <> %s::uuid)
               GROUP BY e.id, w.canonical_name
               ORDER BY cites DESC, w.canonical_name
               LIMIT %s""",
            (entity_ids, entity_ids, exclude_wiki_id, exclude_wiki_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def release_or_fail_jobs(conn, job_ids: list[str], last_error: str,
                         max_attempts: int = 3) -> str:
    """On a gate failure: return jobs to 'pending' for retry, or 'failed' once
    attempts are exhausted (surfaced via GET /jobs — never a silent bad write)."""
    if not job_ids:
        return "none"
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE wiki_job
                  SET status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                      last_error = %s
                WHERE id = ANY(%s::uuid[])""",
            (max_attempts, last_error[:1000], job_ids),
        )
    return "failed" if _max_attempts_reached(conn, job_ids, max_attempts) else "requeued"


def _max_attempts_reached(conn, job_ids: list[str], max_attempts: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT bool_or(status='failed') FROM wiki_job WHERE id = ANY(%s::uuid[])",
            (job_ids,),
        )
        return bool(cur.fetchone()[0])


def finish_jobs(conn, job_ids: list[str], status: str, last_error: str | None = None) -> None:
    if not job_ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE wiki_job SET status=%s, completed_at=now(), last_error=%s "
            "WHERE id = ANY(%s::uuid[])",
            (status, last_error, job_ids),
        )


def create_wiki_entity(conn, canonical_name: str, body: str, summary: str | None,
                       disambiguation: str | None, member_entity_ids: list[str],
                       keywords: list[str] | None = None) -> str:
    """Scaffolding only — a new wiki page is additive, not destruction. The
    body, summary, disambiguation, and keywords are ALL the LLM's: `keywords`
    is whatever the LLM declared in its meta header (may be empty). Code never
    invents keywords (no `[canonical_name]` default)."""
    from braindb.services.embedding_service import get_embedding_service
    from braindb.services.keyword_service import (
        ensure_keyword_entities, link_entity_to_keywords,
    )
    kws = [k.strip() for k in (keywords or []) if k and k.strip()]
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO entities (entity_type, title, content, summary, keywords,
                                     importance, source)
               VALUES ('wiki', %s, %s, %s, %s, 0.9, 'agent-inference')
               RETURNING id""",
            (canonical_name, body, summary, kws),
        )
        wid = str(cur.fetchone()[0])
    if kws:
        kw_map = ensure_keyword_entities(conn, kws, get_embedding_service())
        link_entity_to_keywords(conn, wid, list(kw_map.values()))
    member_kw = _keyword_ids_among(conn, member_entity_ids)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO wikis_ext
                   (entity_id, canonical_name, disambiguation, language,
                    member_keyword_ids, revision, last_synthesised_at)
               VALUES (%s, %s, %s, 'en', %s::uuid[], 1, now())""",
            (wid, canonical_name, disambiguation, member_kw),
        )
    return wid


def _keyword_ids_among(conn, entity_ids: list[str]) -> list[str]:
    if not entity_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id::text FROM entities "
            "WHERE id = ANY(%s::uuid[]) AND entity_type='keyword'",
            (entity_ids,),
        )
        return [r[0] for r in cur.fetchall()]


def finalize_wiki_write(conn, wiki_id: str, new_body: str, summary: str | None,
                        disambiguation: str | None, member_entity_ids: list[str]) -> int:
    """Apply the gated body to an existing wiki: update content + header
    fields, union new keyword members, bump revision."""
    new_kw = _keyword_ids_among(conn, member_entity_ids)
    with conn.cursor() as cur:
        cur.execute("UPDATE entities SET content=%s, summary=%s WHERE id=%s",
                    (new_body, summary, wiki_id))
        cur.execute(
            """UPDATE wikis_ext
                  SET disambiguation = COALESCE(%s, disambiguation),
                      member_keyword_ids = (
                          SELECT ARRAY(SELECT DISTINCT unnest(
                              member_keyword_ids || %s::uuid[]))),
                      revision = revision + 1,
                      last_synthesised_at = now()
                WHERE entity_id = %s
              RETURNING revision""",
            (disambiguation, new_kw, wiki_id),
        )
        return cur.fetchone()[0]


def fetch_wikis_for_merge(conn, wiki_ids: list[str]) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT e.id::text AS id, e.content, e.importance, w.canonical_name,
                      w.revision, w.member_keyword_ids::text[] AS member_keyword_ids,
                      w.retired_at
               FROM entities e JOIN wikis_ext w ON w.entity_id = e.id
               WHERE e.id = ANY(%s::uuid[]) AND e.entity_type='wiki'""",
            (wiki_ids,),
        )
        return [dict(r) for r in cur.fetchall()]


def soft_retire_wiki(conn, loser_id: str, canonical_id: str, note: str | None) -> None:
    """LLM-decided retirement, executed deterministically + reversibly: the
    loser drops out of ranking (importance 0) but still resolves; provenance
    is kept via duplicate_of / consolidated_into edges (which also self-clear
    the maintainer's dedup, since it is prompted to skip marked pairs)."""
    from braindb.services.activity_log import log_activity
    with conn.cursor() as cur:
        cur.execute("UPDATE entities SET importance = 0.0 WHERE id = %s", (loser_id,))
        cur.execute(
            "UPDATE wikis_ext SET retired_at = now(), redirect_to = %s WHERE entity_id = %s",
            (canonical_id, loser_id),
        )
        for rtype in ("duplicate_of", "consolidated_into"):
            cur.execute(
                """INSERT INTO relations
                   (from_entity_id, to_entity_id, relation_type, relevance_score, description)
                   VALUES (%s, %s, %s, 0.0, %s)
                   ON CONFLICT (from_entity_id, to_entity_id, relation_type) DO NOTHING""",
                (loser_id, canonical_id, rtype, (note or "merged")[:500]),
            )
    log_activity(conn, "wiki_merge", "wiki", canonical_id,
                 details={"retired": loser_id, "canonical": canonical_id, "note": note})


def extract_summary_disambig(body: str) -> tuple[str | None, str | None]:
    sm = SUMMARY_RE.search(body or "")
    dm = DISAMBIG_RE.search(body or "")
    return (sm.group(1).strip() if sm else None,
            dm.group(1).strip() if dm else None)


def list_jobs(conn, status: str | None, job_type: str | None, limit: int) -> list[dict]:
    conditions, params = [], []
    if status:
        conditions.append("status = %s")
        params.append(status)
    if job_type:
        conditions.append("job_type = %s")
        params.append(job_type)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT id, job_type, status, target_wiki_id,
                   entity_ids::text[] AS entity_ids, dedupe_key, rationale,
                   proposed_name, batch_id, created_at, assigned_at,
                   completed_at, attempts, last_error
            FROM wiki_job
            {where}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]
