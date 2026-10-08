"""DB tests: the JEV gate inside the real /wiki/write route.

Runs against a Postgres with the BrainDB schema (the CI workflow's service, or
the isolated test stack). The gate's decision and the writer agent are both
stubbed: this pins the ROUTE's contract, not the model —

- a gated job reaches the writer with only the kept facts: the research and
  identity steps are replaced by the gate's facts, a found fact the gate left
  out is not in the prompt, the writer runs without research tools, and the
  activity row records the gate;
- without a gate (off, no decision, or a failing gate) the writer's prompt is
  byte-identical to the pre-gate assembly and runs on the full writer;
- a handoff successor of a gated job runs on the gated writer too;
- on a long page the writer is shown only the sections its facts belong in,
  within a budget, and is told what it was not shown;
- a gated no-op is accepted like any other, with the gate on its row.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import psycopg2
import psycopg2.extras
import pytest

import braindb.routers.wiki as wiki_router
from braindb import custom_profile
from braindb.agent import agent as agent_mod
from braindb.agent.run_state import record_handoff
from braindb.agent.schemas import WikiWriteResult
from braindb.services import wiki_jobs
from braindb.services.wiki_gate import Gate

DB_URL = os.getenv("DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test")
HEADER = ("<!-- wiki:meta canonical_name=PytestGate Calder Labs language=en -->\n# PytestGate Calder Labs\n"
          "> **Summary:** battery testing robots maker.\n")
M1_TEXT, M2_TEXT = "_pytest_gate Calder Labs opened a factory", "_pytest_gate Calder Labs hired engineers"


def _setup(body_for):
    """A live wiki with two orphan member facts and their pending attach job
    (old enough to be claimed first). `body_for(m1, m2)` builds the page body."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    wiki, m1, m2, job = (str(uuid.uuid4()) for _ in range(4))
    body = body_for(m1, m2)
    with conn.cursor() as cur:
        for m, text in ((m1, M1_TEXT), (m2, M2_TEXT)):
            cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance, created_at)
                           VALUES (%s, 'fact', %s, %s, 'user-stated', 0.5, now() - interval '2 hours')""",
                        (m, text, ["_pytest_gate"]))
        cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                       VALUES (%s, 'wiki', %s, %s, 'agent-inference', 0.5)""", (wiki, body, ["_pytest_gate"]))
        cur.execute("""INSERT INTO wikis_ext (entity_id, canonical_name, language, member_keyword_ids, revision)
                       VALUES (%s, 'PytestGate Calder Labs', 'en', '{}'::uuid[], 1)""", (wiki,))
        cur.execute("""INSERT INTO wiki_job (id, job_type, status, target_wiki_id, entity_ids, dedupe_key, rationale, created_at)
                       VALUES (%s, 'attach', 'pending', %s, %s::uuid[], %s, 'pytest gate', now() - interval '10 years')""",
                    (job, wiki, [m1, m2], f"_pytest_gate_{job}"))
    return conn, {"wiki": wiki, "m1": m1, "m2": m2, "job": job, "body": body}


def _teardown(conn, ids):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM wiki_job WHERE id = %s OR target_wiki_id = %s OR entity_ids && %s::uuid[]",
                    (ids["job"], ids["wiki"], [ids["m1"], ids["m2"]]))
        cur.execute("DELETE FROM activity_log WHERE entity_id = %s", (ids["wiki"],))
        cur.execute("DELETE FROM wikis_ext WHERE entity_id = %s", (ids["wiki"],))
        cur.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", ([ids["wiki"], ids["m1"], ids["m2"]],))
    conn.close()


def _page(body_for):
    conn, ids = _setup(body_for)
    try:
        yield conn, ids
    finally:
        _teardown(conn, ids)


@pytest.fixture
def short_page():
    yield from _page(lambda m1, m2: HEADER + "<!-- section: overview -->\nBuilds robots.\n")


@pytest.fixture
def cited_page():
    """The page already cites both members (so a no-op is legitimate)."""
    yield from _page(lambda m1, m2: HEADER + f"<!-- section: overview -->\nFactory [[ref:{m1}]], hiring [[ref:{m2}]].\n")


@pytest.fixture
def long_page():
    yield from _page(lambda m1, m2: HEADER + "<!-- section: overview -->\n" + "OVERVIEW_ONLY_TEXT. " * 300
                     + "\n<!-- section: timeline -->\nTIMELINE_TEXT 2019 founded.\n"
                     + "<!-- section: finances -->\nFINANCES_TEXT revenue.\n")


def _gate(ids, **kwargs):
    members = [{"id": ids["m1"], "entity_type": "fact", "content": M1_TEXT, "keywords": []},
               {"id": ids["m2"], "entity_type": "fact", "content": M2_TEXT, "keywords": []}]
    return Gate(members, kwargs.pop("found", []), **kwargs)


def _write_row(conn, wiki):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT details FROM activity_log WHERE operation = 'wiki_write' AND entity_id = %s", (wiki,))
        return [r["details"] for r in cur.fetchall()]


def _agent_stub(calls, result):
    async def run_typed(prompt, agent, cls, **kwargs):
        calls.append((prompt, agent))
        if isinstance(result, Exception):
            raise result
        return result
    return run_typed


def _use_gate(monkeypatch, gate):
    monkeypatch.setattr(wiki_router.wiki_gate, "wanted", lambda: True)
    monkeypatch.setattr(wiki_router.wiki_gate, "prepare", lambda *a, **k: gate)


def test_a_gated_job_reaches_the_writer_with_only_the_kept_facts(short_page, monkeypatch):
    conn, ids = short_page
    found = {"id": str(uuid.uuid4()), "entity_type": "fact", "content": "_pytest_gate Calder Labs won a prize", "keywords": []}
    left_out = str(uuid.uuid4())
    calls = []
    _use_gate(monkeypatch, _gate(ids, found=[found], excluded=[left_out]))
    monkeypatch.setattr(wiki_router, "run_typed", _agent_stub(calls, WikiWriteResult(
        mode="attach", body=ids["body"] + f"Both [[ref:{ids['m1']}]] [[ref:{ids['m2']}]].\n")))
    result = asyncio.run(wiki_router.wiki_write())
    assert result["written"] == 1
    prompt, agent = calls[0]
    assert "Step 1 — Gather raw facts" not in prompt and "MANDATORY `delegate_to_subagent`" not in prompt
    assert "Facts found in memory about this subject" in prompt and found["id"] in prompt
    assert ids["m1"] in prompt and ids["m2"] in prompt and left_out not in prompt
    assert agent is agent_mod.get_gated_writer_agent()
    rows = _write_row(conn, ids["wiki"])
    assert rows and rows[0]["gate"]["excluded"] == [left_out] and rows[0]["gate"]["found"] == [found["id"]]


@pytest.mark.parametrize("page", ["short_page", "long_page"])
@pytest.mark.parametrize("gate_case", ["off", "no decision", "gate fails"])
def test_without_a_gate_the_writer_prompt_is_byte_identical(page, gate_case, request, monkeypatch):
    conn, ids = request.getfixturevalue(page)
    calls = []
    monkeypatch.setattr(wiki_router, "_now_line", lambda: "\n<NOW>\n")
    monkeypatch.setattr(wiki_router.wiki_gate, "wanted", lambda: gate_case != "off")
    if gate_case == "gate fails":
        def boom(*a, **k):
            raise RuntimeError("the gate broke")
        monkeypatch.setattr(wiki_router.wiki_gate, "prepare", boom)
    else:
        monkeypatch.setattr(wiki_router.wiki_gate, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(wiki_router, "run_typed", _agent_stub(calls, RuntimeError("stop after the prompt")))
    asyncio.run(wiki_router.wiki_write())
    members = wiki_jobs.fetch_members(conn, [ids["m1"], ids["m2"]])
    expected = (                                   # the assembly as it was before the gate existed
        custom_profile.replace_or_base(wiki_router._WRITER_PROMPT, "wiki_writer")
        .replace("%%MODE%%", "attach")
        .replace("%%CANONICAL%%", "PytestGate Calder Labs")
        .replace("%%WIKI_ID%%", ids["wiki"])
        .replace("%%MEMBERS%%", wiki_router._members_block(members))
        .replace("%%CURRENT_BODY%%", wiki_router._body_block_or_stub("attach", ids["wiki"], ids["body"]))
        .replace("%%RELATED_WIKIS%%", "(none — no neighbouring pages exist yet)")
        .replace("%%DUPLICATES%%", "(n/a)")
    ) + "\n<NOW>\n" + custom_profile.additions("wiki_writer")
    prompt, agent = calls[0]
    assert prompt == expected and agent is agent_mod.get_writer_agent()


def test_a_gated_job_s_successor_runs_on_the_gated_writer_too(short_page, monkeypatch):
    conn, ids = short_page
    calls = []

    async def run_typed(prompt, agent, cls, **kwargs):
        calls.append((prompt, agent))
        if len(calls) == 1:
            record_handoff("progress so far", "remaining work")   # the first run hands off
        return WikiWriteResult(mode="attach", body=ids["body"] + f"Both [[ref:{ids['m1']}]] [[ref:{ids['m2']}]].\n")

    _use_gate(monkeypatch, _gate(ids))
    monkeypatch.setattr(wiki_router, "run_typed", run_typed)
    asyncio.run(wiki_router.wiki_write())
    assert len(calls) == 2 and calls[1][1] is calls[0][1] is agent_mod.get_gated_writer_agent()
    assert "CONTINUATION" in calls[1][0] and "Step 1 — Gather raw facts" not in calls[1][0]


def test_a_long_page_shows_only_the_sections_its_facts_belong_in(long_page, monkeypatch):
    conn, ids = long_page
    calls = []
    _use_gate(monkeypatch, _gate(ids, sections={ids["m1"]: "timeline"}))
    monkeypatch.setattr(wiki_router, "run_typed", _agent_stub(calls, RuntimeError("stop after the prompt")))
    asyncio.run(wiki_router.wiki_write())
    prompt, _ = calls[0]
    assert f"- timeline <- {ids['m1']}" in prompt and "TIMELINE_TEXT 2019 founded." in prompt
    assert "OVERVIEW_ONLY_TEXT" not in prompt and "FINANCES_TEXT" not in prompt
    assert "- overview: " in prompt and "- finances: " in prompt          # the outline still names them
    assert f"Place these yourself, reading the sections you need: {ids['m2']}" in prompt
    assert "revision 1" in prompt and "the others stay as they are" not in prompt


def test_a_gated_no_op_is_accepted_with_the_gate_on_its_row(cited_page, monkeypatch):
    conn, ids = cited_page
    left_out = str(uuid.uuid4())
    _use_gate(monkeypatch, _gate(ids, excluded=[left_out]))
    monkeypatch.setattr(wiki_router, "run_typed", _agent_stub([], WikiWriteResult(mode="attach", body="")))
    result = asyncio.run(wiki_router.wiki_write())
    assert result.get("no_op") is True
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM wiki_job WHERE id = %s", (ids["job"],))
        assert cur.fetchone()[0] == "done"
    assert _write_row(conn, ids["wiki"])[0]["gate"]["excluded"] == [left_out]


def test_the_long_page_block_keeps_to_its_budget_and_says_what_it_left_out():
    big = "x" * 31000                                                   # over the budget and the density limit
    body = HEADER + f"<!-- section: overview -->\n{big}\n<!-- section: timeline -->\nTIMELINE_TEXT.\n"
    members = [{"id": "m1", "entity_type": "fact", "content": "a", "keywords": []},
               {"id": "m2", "entity_type": "fact", "content": "b", "keywords": []}]
    block = wiki_router._gated_body_block("w", 3, body, Gate(members, [], sections={"m1": "overview"}))
    assert big not in block and "Too long to show here; read before replacing: overview" in block
    assert "Place these yourself, reading the sections you need: m2" in block
    assert "TIMELINE_TEXT" not in block and "revision 3" in block
    assert "Those sections are shown in full below" not in block       # nothing was shown
    assert "ALREADY LONG" in block                                      # the density paragraph long pages get
