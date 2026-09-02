"""`reconcile_summarises_additive` must never let CONTENT block BOOKKEEPING.

Exercises the function directly against the live Postgres instance, in the
style of `test_wiki_jobs_grouping.py`: seed a minimal wiki + entities, call
the function, assert, clean up in `try/finally`.

The contract under test
-----------------------

A wiki body cites entities inline as `[[ref:UUID]]`, and this function
mirrors each citation into a `wiki --summarises--> e` relation. It is the
step that makes a member stop being an orphan, and it runs in the SAME
transaction as `finish_jobs`.

So a citation whose entity does not exist is not a content problem — it is
a liveness problem. Inserting it raises a foreign-key violation, which
aborts the whole transaction and takes `finish_jobs` with it. The section
edits made during the agent run are already committed, so the page keeps
growing while its jobs are never closed and re-queue forever.

That is not hypothetical. A subagent without the section tools retyped a
52k-char body through `update_entity` to change one line and flipped one
hex digit of a cited UUID (`...-4e7c` -> `...-4f7c`). Every subsequent
write on that page returned 500, no job closed for three days, and the
`summarises` set froze at 114 while the body grew 44k -> 73.5k chars.

The fix under test: a dangling ref is SKIPPED and REPORTED, never raised.
Reporting matters — the caller spreads the result into the `wiki_write`
activity log, so this stays visible rather than silent.
"""
from __future__ import annotations

import os
import uuid
from typing import Iterator

import psycopg2
import pytest

from braindb.services import wiki_jobs


# Same default as tests/conftest.py — the isolated stack from
# docker-compose.test.yml. An explicit DATABASE_URL in the env wins.
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test"
)


# ---------------------------------------------------------------- helpers --


def _insert_wiki(conn, label: str) -> str:
    """Minimal wiki entity + its keyword + wikis_ext row (wikis_ext expects
    member_keyword_ids non-empty). Returns the wiki entity UUID as text."""
    wid, kw_id = uuid.uuid4(), uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source, importance)
               VALUES (%s, 'keyword', %s, %s, 'agent-inference', 0.5)""",
            (str(kw_id), f"_pytest_reconcile_kw_{label}", [f"_pytest_reconcile_{label}"]),
        )
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source, importance)
               VALUES (%s, 'wiki', %s, %s, 'agent-inference', 0.5)""",
            (str(wid), f"# Test wiki ({label})\n", [f"_pytest_reconcile_{label}"]),
        )
        cur.execute(
            """INSERT INTO wikis_ext (entity_id, canonical_name, language,
                                      member_keyword_ids, revision)
               VALUES (%s, %s, 'en', %s::uuid[], 1)""",
            (str(wid), f"PytestReconcile_{label}", [str(kw_id)]),
        )
    return str(wid)


def _insert_fact(conn, label: str) -> str:
    fid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO entities (id, entity_type, content, keywords, source, importance)
               VALUES (%s, 'fact', %s, %s, 'user-stated', 0.5)""",
            (str(fid), f"_pytest_reconcile_fact_{label}", [f"_pytest_reconcile_{label}"]),
        )
    return str(fid)


def _summarised_ids(conn, wiki_id: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_entity_id::text FROM relations "
            "WHERE from_entity_id = %s AND relation_type = 'summarises'",
            (wiki_id,),
        )
        return {r[0].lower() for r in cur.fetchall()}


def _cleanup(conn, entity_ids: list[str]) -> None:
    with conn.cursor() as cur:
        if entity_ids:
            cur.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", (entity_ids,))
        cur.execute(
            "DELETE FROM entities WHERE entity_type='keyword' "
            "AND content LIKE '_pytest_reconcile_kw_%'"
        )


def _body(*refs: str) -> str:
    lines = "\n".join(f"- claim [[ref:{r}]]" for r in refs)
    return f"<!-- section:references -->\n{lines}\n"


@pytest.fixture
def db() -> Iterator[psycopg2.extensions.connection]:
    """One autocommit psycopg2 connection per test, closed at teardown."""
    c = psycopg2.connect(DB_URL)
    c.autocommit = True
    try:
        yield c
    finally:
        c.close()


# ------------------------------------------------------------------ tests --


def test_live_refs_become_summarises_relations(db):
    """Baseline: the normal path is unchanged."""
    wid = _insert_wiki(db, "live")
    f1, f2 = _insert_fact(db, "live1"), _insert_fact(db, "live2")
    try:
        res = wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1, f2))
        assert res["relations_added"] == 2
        assert res["refs_skipped"] == []
        assert _summarised_ids(db, wid) == {f1.lower(), f2.lower()}
    finally:
        _cleanup(db, [wid, f1, f2])


def test_dangling_ref_does_not_raise(db):
    """THE regression. A FK violation here aborts the caller's transaction
    and `finish_jobs` never runs, so the jobs re-queue forever."""
    wid = _insert_wiki(db, "dangle")
    ghost = str(uuid.uuid4())  # never inserted
    try:
        res = wiki_jobs.reconcile_summarises_additive(db, wid, _body(ghost))
        assert res["relations_added"] == 0
        assert res["refs_skipped"] == [ghost.lower()]
    finally:
        _cleanup(db, [wid])


def test_one_dangling_ref_does_not_block_the_good_ones(db):
    """The corrupted page had 3 bad refs among ~300 good ones. All the
    valid citations must still be recorded."""
    wid = _insert_wiki(db, "mixed")
    f1, f2 = _insert_fact(db, "mixed1"), _insert_fact(db, "mixed2")
    ghost = str(uuid.uuid4())
    try:
        res = wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1, ghost, f2))
        assert res["relations_added"] == 2
        assert res["refs_skipped"] == [ghost.lower()]
        assert _summarised_ids(db, wid) == {f1.lower(), f2.lower()}
    finally:
        _cleanup(db, [wid, f1, f2])


def test_skipped_refs_are_reported_not_silent(db):
    """The caller spreads this dict into the `wiki_write` activity log, so
    a skipped ref stays visible — "never a silent bad write"."""
    wid = _insert_wiki(db, "report")
    g1, g2 = str(uuid.uuid4()), str(uuid.uuid4())
    try:
        res = wiki_jobs.reconcile_summarises_additive(db, wid, _body(g1, g2))
        assert sorted(res["refs_skipped"]) == sorted([g1.lower(), g2.lower()])
    finally:
        _cleanup(db, [wid])


def test_reconcile_remains_additive_only(db):
    """A pre-existing relation whose entity is no longer cited must NOT be
    removed — the LLM owns retraction via `delete_relation`."""
    wid = _insert_wiki(db, "additive")
    f1, f2 = _insert_fact(db, "add1"), _insert_fact(db, "add2")
    try:
        wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1, f2))
        # now re-run with a body citing only f1
        res = wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1))
        assert res["relations_added"] == 0
        assert res["relations_removed"] == 0
        assert _summarised_ids(db, wid) == {f1.lower(), f2.lower()}
    finally:
        _cleanup(db, [wid, f1, f2])


def test_rerun_is_idempotent(db):
    """Re-running on the same body adds nothing — the cron/retry path leans
    on this."""
    wid = _insert_wiki(db, "idem")
    f1 = _insert_fact(db, "idem1")
    try:
        first = wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1))
        second = wiki_jobs.reconcile_summarises_additive(db, wid, _body(f1))
        assert first["relations_added"] == 1
        assert second["relations_added"] == 0
        assert _summarised_ids(db, wid) == {f1.lower()}
    finally:
        _cleanup(db, [wid, f1])


def test_body_with_no_refs_is_a_clean_noop(db):
    wid = _insert_wiki(db, "norefs")
    try:
        res = wiki_jobs.reconcile_summarises_additive(
            db, wid, "<!-- section:overview -->\nprose with no citations\n")
        assert res == {"relations_added": 0, "relations_removed": 0,
                       "refs_skipped": []}
    finally:
        _cleanup(db, [wid])
