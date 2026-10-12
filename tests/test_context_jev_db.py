"""JEV's recall filter, tested against the real database.

`assemble_context(conn, req, need=...)` is the one place JEV changes recall.
The judge itself is stubbed — these pin the wiring, not the model:

- no need, or a judge that fails open (None): exactly the unfiltered result;
- with a need: only the items judged useful (P >= JEV_STRICTNESS) or left
  unjudged are returned, still in final_rank order, and only those are
  access-tracked;
- the judge reads each item's own text, never the recall preview.

Each test runs inside one transaction that is rolled back at teardown, so
nothing it inserts or access-tracks ever persists.
"""
from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

from braindb.schemas.search import ContextRequest
from braindb.services import context as context_mod
from braindb.services import jev

# Same default as tests/conftest.py — the isolated stack from
# docker-compose.test.yml. An explicit DATABASE_URL in the environment wins
# (in CI it points at the workflow's Postgres service container).
DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test"
)


@pytest.fixture
def conn():
    c = psycopg2.connect(DB_URL)
    yield c
    c.rollback()
    c.close()


@pytest.fixture
def facts(conn):
    """Four facts only this test's query can find, marked for the stub judge."""
    token = f"zqjev{uuid.uuid4().hex[:10]}"
    bodies = {"keep_a": f"{token} KEEP alpha", "drop": f"{token} DROP beta",
              "unjudged": f"{token} UNJUDGED gamma",
              "keep_long": f"{token} KEEP delta " + "x" * 2000}
    ids = {}
    with conn.cursor() as cur:
        for name, body in bodies.items():
            ids[name] = str(uuid.uuid4())
            cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                           VALUES (%s, 'fact', %s, %s, 'user-stated', 0.8)""",
                        (ids[name], body, [token]))
    return token, ids


def _stub_judge(calls):
    def judge(need, texts):
        calls.append(texts)
        return [None if "UNJUDGED" in t else 0.9 if "KEEP" in t else 0.1 for t in texts]
    return judge


def _ids(result):
    return [str(it.id) for it in result.items]


def _access_counts(conn, ids):
    with conn.cursor() as cur:
        cur.execute("SELECT id::text, access_count FROM entities WHERE id = ANY(%s::uuid[])", (ids,))
        return dict(cur.fetchall())


def test_without_a_need_or_when_jev_fails_open_recall_is_unchanged(conn, facts, monkeypatch):
    token, ids = facts
    req = ContextRequest(query=token)
    baseline = _ids(context_mod.assemble_context(conn, req))
    assert set(ids.values()) <= set(baseline)
    assert _ids(context_mod.assemble_context(conn, req, need="")) == baseline
    monkeypatch.setattr(jev, "judge", lambda need, texts: None)  # off, or failed open
    assert _ids(context_mod.assemble_context(conn, req, need="anything")) == baseline


def test_with_a_need_only_useful_or_unjudged_items_are_returned_and_tracked(conn, facts, monkeypatch):
    token, ids = facts
    calls: list = []
    monkeypatch.setattr(jev, "judge", _stub_judge(calls))
    monkeypatch.setattr(context_mod.settings, "jev_strictness", 0.5)
    req = ContextRequest(query=token)
    unfiltered = _ids(context_mod.assemble_context(conn, req))
    before = _access_counts(conn, list(ids.values()))

    result = context_mod.assemble_context(conn, req, need="what is kept?")

    kept = [ids["keep_a"], ids["unjudged"], ids["keep_long"]]
    assert set(_ids(result)) == set(kept)
    assert _ids(result) == [i for i in unfiltered if i in kept]       # final_rank order kept
    assert result.total_found == len(kept)
    after = _access_counts(conn, list(ids.values()))
    assert all(after[i] == before[i] + 1 for i in kept)
    assert after[ids["drop"]] == before[ids["drop"]]                  # dropped: not tracked
    assert not any("--truncated" in t for t in calls[0])              # own text, not the preview
