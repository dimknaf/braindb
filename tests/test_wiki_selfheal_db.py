"""Self-healing and guard behaviour, tested against the real database.

Every test here pins a property that would NOT fail under a revert if it
were asserted on strings or private tuples (the earlier, weaker tests did
exactly that). These call the real functions and the real tools.

The contract under test
-----------------------

1. THE SELF-HEAL LOOP. A job past both the lease and the reclaim ceiling is
   un-claimable, and `assigned` counts as an ACTIVE job in
   `_orphan_conditions()` — so without a disposition its entity would be
   permanently barred from re-triage: silent, unbounded loss. `run_cron`
   must flip such rows to `failed` (the status that, by design, returns the
   entity to the orphan pool) and re-enqueue a fresh triage job for the
   entity — in the same tick.

2. THE RECLAIM CEILING, behaviourally: `claim_jobs` re-claims a
   lease-expired job below the ceiling and refuses one at it.

3. `update_entity` GUARDS, through the real tool (`on_invoke_tool` with a
   stub ctx): a wiki body cannot be overwritten, a fact still can, other
   fields still apply to wikis, and `content=""` never silently destroys a
   body. NOTE: the SDK converts tool exceptions into an error STRING, so
   every assertion here is on reply content, never on "no exception".

4. ONE CITATION PREDICATE. `uncited_members` splits un-cited members into
   `missing` (entity exists — real work) and `gone` (deleted since triage —
   can never be cited, must not wedge the job). The router's no-op gate and
   the `check_members_cited` tool both call it, so agreement is by
   construction; these tests pin the split itself and the tool's reporting.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Iterator

import psycopg2
import psycopg2.extras
import pytest

from braindb.services import wiki_jobs


# Same default as tests/conftest.py — the isolated stack from
# docker-compose.test.yml. An explicit DATABASE_URL in the environment wins
# (in CI it points at the workflow's Postgres service container).
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test"
)


class _Ctx:
    """Minimal ToolContext stub: the SDK's invocation path reads only
    `tool_name` (agents/tool.py `_on_invoke_tool_impl`)."""
    tool_name = "test"


def _invoke(tool, **kwargs) -> str:
    """Run a FunctionTool the way the SDK would, returning its string reply."""
    return asyncio.run(tool.on_invoke_tool(_Ctx(), json.dumps(kwargs)))


# ---------------------------------------------------------------- helpers --


def _insert_entity(conn, entity_type: str, label: str, *,
                   age_minutes: int = 0, notes: str | None = None) -> str:
    eid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source,
                                     importance, notes, created_at)
               VALUES (%s, %s, %s, %s, 'user-stated', 0.5, %s,
                       now() - make_interval(mins => %s))""",
            (str(eid), entity_type, f"_pytest_selfheal_{label}",
             [f"_pytest_selfheal_{label}"], notes, age_minutes),
        )
    return str(eid)


def _insert_wiki(conn, label: str, body: str) -> str:
    wid, kw_id = uuid.uuid4(), uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source, importance)
               VALUES (%s, 'keyword', %s, %s, 'agent-inference', 0.5)""",
            (str(kw_id), f"_pytest_selfheal_kw_{label}", [f"_pytest_selfheal_{label}"]),
        )
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source, importance)
               VALUES (%s, 'wiki', %s, %s, 'agent-inference', 0.5)""",
            (str(wid), body, [f"_pytest_selfheal_{label}"]),
        )
        cur.execute(
            """INSERT INTO wikis_ext (entity_id, canonical_name, language,
                                      member_keyword_ids, revision)
               VALUES (%s, %s, 'en', %s::uuid[], 1)""",
            (str(wid), f"PytestSelfheal_{label}", [str(kw_id)]),
        )
    return str(wid)


def _insert_job(conn, *, job_type: str, entity_ids: list[str],
                status: str = "pending", attempts: int = 0,
                assigned_age_minutes: int | None = None,
                target_wiki_id: str | None = None,
                dedupe_key: str | None = None) -> str:
    jid = uuid.uuid4()
    dedupe = dedupe_key or f"_pytest_selfheal_{job_type}_{uuid.uuid4().hex}"
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO wiki_job
               (id, job_type, status, target_wiki_id, entity_ids, dedupe_key,
                attempts, assigned_at, rationale)
               VALUES (%s, %s, %s, %s, %s::uuid[], %s, %s,
                       CASE WHEN %s::int IS NULL THEN NULL
                            ELSE now() - make_interval(mins => %s) END,
                       'pytest selfheal')""",
            (str(jid), job_type, status, target_wiki_id, entity_ids, dedupe,
             attempts, assigned_age_minutes, assigned_age_minutes or 0),
        )
    return str(jid)


def _job_row(conn, job_id: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM wiki_job WHERE id = %s", (job_id,))
        return dict(cur.fetchone())


def _cleanup(conn, *, entity_ids: list[str] = (), job_ids: list[str] = ()) -> None:
    with conn.cursor() as cur:
        if job_ids:
            cur.execute("DELETE FROM wiki_job WHERE id = ANY(%s::uuid[])", (list(job_ids),))
        cur.execute("DELETE FROM wiki_job WHERE rationale = 'pytest selfheal'")
        cur.execute("DELETE FROM wiki_job WHERE dedupe_key LIKE 'triage:%%' "
                    "AND entity_ids && %s::uuid[]",
                    (list(entity_ids) or ["00000000-0000-0000-0000-000000000000"],))
        if entity_ids:
            cur.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", (list(entity_ids),))
        cur.execute("DELETE FROM entities WHERE content LIKE '_pytest_selfheal_%%' "
                    "OR (entity_type='keyword' AND content LIKE '_pytest_selfheal_kw_%%')")


@pytest.fixture
def db() -> Iterator[psycopg2.extensions.connection]:
    c = psycopg2.connect(DB_URL)
    c.autocommit = True
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def _point_tools_at_test_db(monkeypatch):
    """The tools open their own connections via `get_conn()`, which reads
    `settings.database_url` per call — point it at the test database."""
    from braindb.config import settings
    monkeypatch.setattr(settings, "database_url", DB_URL)


# ------------------------------------------------- 1. the self-heal loop --


def test_wedged_job_is_failed_and_entity_retriaged_in_one_cron_tick(db):
    """THE self-heal test. Without run_cron's disposition, this job would
    be un-claimable forever and its entity barred from re-triage for good."""
    eid = _insert_entity(db, "fact", "wedge", age_minutes=90)  # settled
    jid = _insert_job(db, job_type="triage", entity_ids=[eid],
                      status="assigned",
                      attempts=wiki_jobs.ASSIGNED_MAX_RECLAIMS,
                      assigned_age_minutes=wiki_jobs.ASSIGNED_LEASE_MIN + 60,
                      dedupe_key=f"triage:{eid}")
    try:
        result = wiki_jobs.run_cron(db)
        row = _job_row(db, jid)
        assert row["status"] == "failed"
        assert "reclaim ceiling" in (row["last_error"] or "")
        assert result["assigned_expired_failed"] >= 1
        # Same tick: the entity is an orphan again and a FRESH pending
        # triage job exists (the partial dedupe index ignores failed rows).
        with db.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM wiki_job WHERE dedupe_key = %s "
                "AND status = 'pending'", (f"triage:{eid}",))
            assert cur.fetchone()[0] == 1
    finally:
        _cleanup(db, entity_ids=[eid], job_ids=[jid])


def test_healthy_assigned_job_is_left_alone_by_cron(db):
    """A job under the ceiling, or within its lease, is someone's live work."""
    eid = _insert_entity(db, "fact", "healthy", age_minutes=90)
    under_ceiling = _insert_job(db, job_type="triage", entity_ids=[eid],
                                status="assigned", attempts=1,
                                assigned_age_minutes=wiki_jobs.ASSIGNED_LEASE_MIN + 60)
    in_lease = _insert_job(db, job_type="triage", entity_ids=[eid],
                           status="assigned",
                           attempts=wiki_jobs.ASSIGNED_MAX_RECLAIMS,
                           assigned_age_minutes=1)
    try:
        wiki_jobs.run_cron(db)
        assert _job_row(db, under_ceiling)["status"] == "assigned"
        assert _job_row(db, in_lease)["status"] == "assigned"
    finally:
        _cleanup(db, entity_ids=[eid], job_ids=[under_ceiling, in_lease])


# ------------------------------------------ 2. the ceiling, behaviourally --


def test_claim_jobs_reclaims_below_ceiling_refuses_at_it(db):
    jid = _insert_job(db, job_type="attach", entity_ids=[],
                      status="assigned",
                      attempts=wiki_jobs.ASSIGNED_MAX_RECLAIMS - 1,
                      assigned_age_minutes=wiki_jobs.ASSIGNED_LEASE_MIN + 60)
    try:
        # Below the ceiling + lease expired -> reclaimable.
        assert wiki_jobs.claim_jobs(db, [jid]) == 1
        assert _job_row(db, jid)["attempts"] == wiki_jobs.ASSIGNED_MAX_RECLAIMS
        # Now AT the ceiling: expire the lease again and try to re-claim.
        with db.cursor() as cur:
            cur.execute(
                "UPDATE wiki_job SET assigned_at = now() - make_interval("
                "mins => %s) WHERE id = %s",
                (wiki_jobs.ASSIGNED_LEASE_MIN + 60, jid))
        assert wiki_jobs.claim_jobs(db, [jid]) == 0
    finally:
        _cleanup(db, job_ids=[jid])


# --------------------------------------- 3. update_entity, the real tool --


def test_update_entity_tool_refuses_wiki_body_and_leaves_db_untouched(db):
    from braindb.agent.tools import update_entity
    body = "<!-- section:overview -->\noriginal wiki body\n"
    wid = _insert_wiki(db, "guard", body)
    try:
        reply = _invoke(update_entity, entity_id=wid, content="OVERWRITTEN")
        assert "content ignored" in reply
        assert "edit_wiki_section" in reply
        with db.cursor() as cur:
            cur.execute("SELECT content FROM entities WHERE id = %s", (wid,))
            assert cur.fetchone()[0] == body
    finally:
        _cleanup(db, entity_ids=[wid])


def test_update_entity_tool_still_applies_other_fields_to_a_wiki(db):
    from braindb.agent.tools import update_entity
    wid = _insert_wiki(db, "fields", "<!-- section:overview -->\nbody\n")
    try:
        reply = _invoke(update_entity, entity_id=wid, notes="curator note",
                        importance=0.9)
        assert reply.startswith("Updated entity")
        with db.cursor() as cur:
            cur.execute("SELECT notes, importance FROM entities WHERE id = %s", (wid,))
            notes, importance = cur.fetchone()
        assert notes == "curator note"
        assert float(importance) == 0.9
    finally:
        _cleanup(db, entity_ids=[wid])


def test_update_entity_tool_still_updates_a_fact_body(db):
    from braindb.agent.tools import update_entity
    eid = _insert_entity(db, "fact", "editable")
    try:
        reply = _invoke(update_entity, entity_id=eid, content="corrected fact")
        assert reply.startswith("Updated entity")
        assert "ignored" not in reply
        with db.cursor() as cur:
            cur.execute("SELECT content FROM entities WHERE id = %s", (eid,))
            assert cur.fetchone()[0] == "corrected fact"
    finally:
        _cleanup(db, entity_ids=[eid])


def test_update_entity_tool_never_silently_blanks_a_body(db):
    """Observed live: content='' wiped a thought whose ref was already baked
    into a wiki. Blanking must be a warned no-op, not silent destruction."""
    from braindb.agent.tools import update_entity
    eid = _insert_entity(db, "thought", "precious", notes="keep me")
    try:
        reply = _invoke(update_entity, entity_id=eid, content="")
        assert "empty content ignored" in reply
        with db.cursor() as cur:
            cur.execute("SELECT content FROM entities WHERE id = %s", (eid,))
            assert cur.fetchone()[0] == "_pytest_selfheal_precious"
    finally:
        _cleanup(db, entity_ids=[eid])


# ------------------------------------------- 4. one citation predicate --


def test_uncited_members_splits_missing_from_gone(db):
    f_cited = _insert_entity(db, "fact", "cited")
    f_uncited = _insert_entity(db, "fact", "uncited")
    ghost = str(uuid.uuid4())  # never inserted — deleted-since-triage case
    body = f"claim [[ref:{f_cited}]]\n"
    try:
        missing, gone = wiki_jobs.uncited_members(
            db, body, [f_cited, f_uncited, ghost])
        assert missing == [f_uncited]   # real outstanding work
        assert gone == [ghost]          # can never be cited — must not wedge
    finally:
        _cleanup(db, entity_ids=[f_cited, f_uncited])


def test_uncited_members_all_cited_is_clean(db):
    f1 = _insert_entity(db, "fact", "done1")
    try:
        assert wiki_jobs.uncited_members(db, f"x [[ref:{f1}]]", [f1]) == ([], [])
    finally:
        _cleanup(db, entity_ids=[f1])


def test_check_members_cited_tool_reports_gone_distinctly(db):
    """The tool and the router share `uncited_members`, so agreement is by
    construction; what the tool adds is honest reporting — a deleted member
    must show as `gone`, not as outstanding work."""
    from braindb.agent.tools import check_members_cited
    f_cited = _insert_entity(db, "fact", "tool_cited")
    f_uncited = _insert_entity(db, "fact", "tool_uncited")
    ghost = str(uuid.uuid4())
    wid = _insert_wiki(db, "tool", f"<!-- section:overview -->\nx [[ref:{f_cited}]]\n")
    try:
        reply = _invoke(check_members_cited, wiki_id=wid,
                        entity_ids=[f_cited, f_uncited, ghost])
        assert "cited: 1/3" in reply
        not_cited_line = next(l for l in reply.splitlines()
                              if l.startswith("NOT_cited:"))
        gone_line = next(l for l in reply.splitlines() if l.startswith("gone"))
        assert f_uncited in not_cited_line and ghost not in not_cited_line
        assert ghost in gone_line
    finally:
        _cleanup(db, entity_ids=[f_cited, f_uncited, wid])
