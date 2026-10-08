"""DB tests: JEV triage inside the real /wiki/maintain route.

Runs against a Postgres with the BrainDB schema (the CI workflow's service, or
the isolated test stack). The JEV decision and the maintainer agent are both
stubbed: this pins the ROUTE's contract, not the model —

- a seed JEV decides never reaches the agent; its own triage job closes and its
  suggestion is created exactly as for an agent decision, and the activity row
  records decided_by=jev;
- with JEV not serving the run, the agent decides every seed as before and the
  activity row is unchanged (no decided_by);
- in a unit JEV decides only in part, the agent decides only the other seeds
  (its answer for a JEV seed never counts), and a failed agent run fails only
  the seeds it was given: JEV's decisions still apply, and the response
  reports both;
- a seed's brief shortlists the wiki that summarises its evidence, never shows
  the seed as its own evidence, and asks recall without the always-on rules.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace

import psycopg2
import psycopg2.extras
import pytest

import braindb.routers.wiki as wiki_router
from braindb.agent.schemas import MaintainerClusterDecision, MaintainerClusterItem
from braindb.services import wiki_triage

DB_URL = os.getenv("DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test")


def _insert_wiki(cur, wiki: str, kw: str) -> None:
    """One live wiki, with its own member keyword."""
    cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                   VALUES (%s, 'keyword', '_pytest_triage_kw', %s, 'agent-inference', 0.5)""",
                (kw, ["_pytest_triage"]))
    cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                   VALUES (%s, 'wiki', '# Pytest Calder Labs', %s, 'agent-inference', 0.5)""",
                (wiki, ["_pytest_triage"]))
    cur.execute("""INSERT INTO wikis_ext (entity_id, canonical_name, language, member_keyword_ids, revision)
                   VALUES (%s, 'PytestTriage Calder Labs', 'en', %s::uuid[], 1)""", (wiki, [kw]))


def _insert_job(cur, job: str, entity: str) -> None:
    cur.execute("""INSERT INTO wiki_job (id, job_type, status, entity_ids, dedupe_key, rationale)
                   VALUES (%s, 'triage', 'pending', %s::uuid[], %s, 'pytest triage')""",
                (job, [entity], f"_pytest_triage_{job}"))


def _cleanup(conn, entities: list[str], jobs: list[str], wiki: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM wiki_job WHERE id = ANY(%s::uuid[]) OR entity_ids && %s::uuid[]",
                    (jobs, entities))
        cur.execute("DELETE FROM activity_log WHERE entity_id = ANY(%s::uuid[]) OR details::text LIKE ANY(%s)",
                    (entities, [f"%{j}%" for j in jobs]))
        cur.execute("DELETE FROM wikis_ext WHERE entity_id = %s", (wiki,))
        cur.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", (entities + [wiki],))


@pytest.fixture
def seeded():
    """An orphan fact with a pending triage job, and one live wiki."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    fact, wiki, kw, job = (str(uuid.uuid4()) for _ in range(4))
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance, created_at)
                       VALUES (%s, 'fact', '_pytest_triage fact about Calder Labs', %s, 'user-stated', 0.99,
                               now() - interval '2 hours')""", (fact, ["_pytest_triage"]))
        _insert_wiki(cur, wiki, kw)
        _insert_job(cur, job, fact)
    try:
        yield conn, {"fact": fact, "wiki": wiki, "job": job}
    finally:
        _cleanup(conn, [fact, kw], [job], wiki)
        conn.close()


@pytest.fixture
def cluster():
    """A two-seed unit: an orphan keyword (claimed first) and the orphan fact
    it tags (claimed with it), each with a pending triage job, and one wiki."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    seed_kw, fact, wiki, kw, job_kw, job_fact = (str(uuid.uuid4()) for _ in range(6))
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance, created_at)
                       VALUES (%s, 'keyword', '_pytest_triage_seed', %s, 'agent-inference', 1.0,
                               now() - interval '2 hours')""", (seed_kw, ["_pytest_triage"]))
        cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance, created_at)
                       VALUES (%s, 'fact', '_pytest_triage fact about Calder Labs', %s, 'user-stated', 0.5,
                               now() - interval '2 hours')""", (fact, ["_pytest_triage"]))
        cur.execute("""INSERT INTO relations (from_entity_id, to_entity_id, relation_type)
                       VALUES (%s, %s, 'tagged_with')""", (fact, seed_kw))
        _insert_wiki(cur, wiki, kw)
        _insert_job(cur, job_kw, seed_kw)
        _insert_job(cur, job_fact, fact)
    try:
        yield conn, {"seed_kw": seed_kw, "fact": fact, "wiki": wiki, "job_kw": job_kw, "job_fact": job_fact}
    finally:
        _cleanup(conn, [seed_kw, fact, kw], [job_kw, job_fact], wiki)
        conn.close()


def _rows(conn, ids):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT status FROM wiki_job WHERE id = %s", (ids["job"],))
        triage = cur.fetchone()["status"]
        cur.execute("""SELECT job_type, target_wiki_id::text AS target FROM wiki_job
                       WHERE entity_ids && %s::uuid[] AND id <> %s""", ([ids["fact"]], ids["job"]))
        suggestions = [dict(r) for r in cur.fetchall()]
        cur.execute("""SELECT details FROM activity_log WHERE operation = 'wiki_maintain' AND entity_id = %s""",
                    (ids["fact"],))
        activity = [r["details"] for r in cur.fetchall()]
    return triage, suggestions, activity


def _jev_attaches(seed: str, wiki: str):
    """A stub `wiki_triage.decide` that decides `seed` alone: attach to `wiki`."""
    def decide(seeds, cat):
        number = next(n for n, w in enumerate(cat, start=1) if w["id"] == wiki)
        return {seed: MaintainerClusterItem(entity_id=seed, action="attach", target_wiki_no=number,
                                            rationale="JEV triage: attach")}
    return decide


def test_a_seed_jev_decides_skips_the_agent_and_is_applied_as_usual(seeded, monkeypatch):
    conn, ids = seeded
    calls = []

    async def agent(prompt, *args, **kwargs):
        calls.append(prompt)  # the route catches an agent's errors: record, never raise
        return MaintainerClusterDecision(decisions=[])

    monkeypatch.setattr(wiki_router.wiki_triage, "wanted", lambda: True)
    monkeypatch.setattr(wiki_router.wiki_triage, "decide", _jev_attaches(ids["fact"], ids["wiki"]))
    monkeypatch.setattr(wiki_router, "run_typed", agent)
    result = asyncio.run(wiki_router.wiki_maintain())
    assert calls == []                                    # the maintainer agent never ran
    assert result["result"][ids["fact"]]["action"] == "attach"
    triage, suggestions, activity = _rows(conn, ids)
    assert triage == "done"
    assert suggestions == [{"job_type": "attach", "target": ids["wiki"]}]
    assert activity and activity[0]["decided_by"] == "jev"


def test_without_jev_the_agent_decides_as_before(seeded, monkeypatch):
    conn, ids = seeded
    calls = []

    async def agent(prompt, agent_obj, cls, **kwargs):
        calls.append(prompt)
        return MaintainerClusterDecision(decisions=[{"entity_id": ids["fact"], "action": "skip",
                                                     "rationale": "trivial"}])

    monkeypatch.setattr(wiki_router.wiki_triage, "wanted", lambda: False)
    monkeypatch.setattr(wiki_router, "run_typed", agent)
    result = asyncio.run(wiki_router.wiki_maintain())
    assert len(calls) == 1 and ids["fact"] in calls[0]
    assert result["result"][ids["fact"]]["action"] == "skip"
    triage, suggestions, activity = _rows(conn, ids)
    assert triage == "rejected" and suggestions == []
    assert activity and "decided_by" not in activity[0]


def test_a_failed_agent_run_fails_only_its_own_seeds(cluster, monkeypatch):
    conn, ids = cluster
    calls = []

    async def agent(prompt, *args, **kwargs):
        calls.append(prompt)
        raise RuntimeError("the model never submitted")

    monkeypatch.setattr(wiki_router.wiki_jobs, "WIKI_TRIAGE_CLUSTER", True)
    monkeypatch.setattr(wiki_router.wiki_triage, "wanted", lambda: True)
    monkeypatch.setattr(wiki_router.wiki_triage, "decide", _jev_attaches(ids["seed_kw"], ids["wiki"]))
    monkeypatch.setattr(wiki_router, "run_typed", agent)
    result = asyncio.run(wiki_router.wiki_maintain())
    assert len(calls) == 1 and ids["fact"] in calls[0] and ids["seed_kw"] not in calls[0]
    assert result["seeds"] == 2
    assert result["result"][ids["seed_kw"]]["action"] == "attach"
    assert result["result"][ids["fact"]]["action"] == "failed"
    with conn.cursor() as cur:
        cur.execute("SELECT id::text, status FROM wiki_job WHERE id = ANY(%s::uuid[])",
                    ([ids["job_kw"], ids["job_fact"]],))
        status = dict(cur.fetchall())
    assert status == {ids["job_kw"]: "done", ids["job_fact"]: "failed"}


def test_in_a_mixed_unit_the_agent_decides_only_its_own_seeds(cluster, monkeypatch):
    conn, ids = cluster
    calls = []

    async def agent(prompt, *args, **kwargs):
        calls.append(prompt)
        return MaintainerClusterDecision(decisions=[  # an answer for the JEV seed too: it must not count
            {"entity_id": ids["seed_kw"], "action": "skip", "rationale": "not mine"},
            {"entity_id": ids["fact"], "action": "skip", "rationale": "trivial"}])

    monkeypatch.setattr(wiki_router.wiki_jobs, "WIKI_TRIAGE_CLUSTER", True)
    monkeypatch.setattr(wiki_router.wiki_triage, "wanted", lambda: True)
    monkeypatch.setattr(wiki_router.wiki_triage, "decide", _jev_attaches(ids["seed_kw"], ids["wiki"]))
    monkeypatch.setattr(wiki_router, "run_typed", agent)
    result = asyncio.run(wiki_router.wiki_maintain())
    assert len(calls) == 1 and ids["fact"] in calls[0] and ids["seed_kw"] not in calls[0]
    assert result["result"][ids["seed_kw"]]["action"] == "attach"
    assert result["result"][ids["fact"]]["action"] == "skip"
    with conn.cursor() as cur:
        cur.execute("SELECT id::text, status FROM wiki_job WHERE id = ANY(%s::uuid[])",
                    ([ids["job_kw"], ids["job_fact"]],))
        status = dict(cur.fetchall())
        cur.execute("""SELECT entity_id::text, details FROM activity_log
                       WHERE operation = 'wiki_maintain' AND entity_id = ANY(%s::uuid[])""",
                    ([ids["seed_kw"], ids["fact"]],))
        activity = dict(cur.fetchall())
    assert status == {ids["job_kw"]: "done", ids["job_fact"]: "rejected"}
    assert activity[ids["seed_kw"]]["decided_by"] == "jev" and "decided_by" not in activity[ids["fact"]]


@pytest.fixture
def summarised():
    """A seed fact, an evidence fact, and a live wiki that summarises the evidence."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    seed, evidence, wiki, kw = (str(uuid.uuid4()) for _ in range(4))
    with conn.cursor() as cur:
        for eid in (seed, evidence):
            cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                           VALUES (%s, 'fact', '_pytest_triage fact', %s, 'user-stated', 0.5)""",
                        (eid, ["_pytest_triage"]))
        _insert_wiki(cur, wiki, kw)
        cur.execute("""INSERT INTO relations (from_entity_id, to_entity_id, relation_type)
                       VALUES (%s, %s, 'summarises')""", (wiki, evidence))
    try:
        yield conn, {"seed": seed, "evidence": evidence, "wiki": wiki}
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM wikis_ext WHERE entity_id = %s", (wiki,))
            cur.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", ([seed, evidence, wiki, kw],))
        conn.close()


def test_the_brief_shortlists_the_wiki_that_summarises_the_evidence(summarised, monkeypatch):
    conn, ids = summarised
    asked = []

    def recall(conn_, req, need=None):  # recall's ranking is not under test: its items are fixed
        asked.append(req)
        return SimpleNamespace(items=[
            SimpleNamespace(id=uuid.UUID(ids["seed"]), entity_type="fact", title=None, content="the seed itself"),
            SimpleNamespace(id=uuid.UUID(ids["evidence"]), entity_type="fact", title=None,
                            content="Calder Labs builds battery testing robots")])

    monkeypatch.setattr(wiki_triage, "assemble_context", recall)
    brief = {"id": ids["seed"], "content": "Calder Labs opened a factory", "keywords": ["factory"]}
    active = {ids["wiki"]: (5, {"canonical_name": "PytestTriage Calder Labs", "char_count": 900})}
    text, labels = wiki_triage._brief(conn, brief, active)
    assert labels == {"A": ids["wiki"]}                  # found only through the summarises relation
    assert asked[0].queries == ["Calder Labs opened a factory", "factory"]
    assert asked[0].include_always_on_rules is False
    assert "Calder Labs builds battery testing robots" in text and "the seed itself" not in text
    assert "\n[A] PytestTriage Calder Labs - " in text and text.startswith("SEED: Calder Labs opened a factory")
