"""Unit tests for the wiki writer's JEV gate (braindb/services/wiki_gate.py).

DB-free and GPU-free: a fake OpenAI-compatible chat endpoint (conftest.py
`fake_chat`) answers by cues in the fact or member asked about. These pin
the contract the writer relies on:

- the decisions are taken INSIDE the gated writer's own first request: each
  is that request (its system prompt and tools) with the job through the
  facts so far, the question line ending it, thinking off; the writer's own
  request then continues what was kept;
- only the found facts judged about the subject stay: one judged no, or
  whose options hold little probability, is rolled back and the next fact
  follows what was kept; wikis, keywords, the job's members and the facts
  the page already cites are never offered;
- the members are never asked about and never send the job to the full
  writer;
- on a long page each member and kept fact gets its section (an unsure one
  is left to the writer); a page with more sections than letters is not
  placed;
- off unless enabled, never under a profile that replaces the writer's
  prompt; a failed probe turns it off; an error or the deadline gives None.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from braindb.services import jev, reader, wiki_gate
from braindb.services.wiki_sections import parse_sections

SYSTEM = "You are the BrainDB Wiki Writer. (gated, with worked examples)"
TOOLS = [{"type": "function", "function": {"name": "edit_wiki_section", "description": "Edit one section.",
                                           "parameters": {"type": "object", "properties": {}}}}]
_QUESTION_LINE = re.compile(r"\[(\w+)\] \| (about the subject \(yes or no\)|section):$")


class _Model:
    """Answers the fact or member a decision asks about by cues in it:
    identity — CUE_NO (or the probe's other Rowan) is no, CUE_LOWMASS holds
    little probability, anything else yes; placement — PLACE_<letter> or
    PLACE_NEW, CUE_LOWPLACE holds little probability. `probe_ok` False
    inverts the probe's answers."""

    def __init__(self):
        self.decisions, self.probe_ok = [], True

    def __call__(self, body: dict):
        anchor, question = _QUESTION_LINE.search(body["messages"][-1]["content"]).groups()
        content = body["messages"][-2]["content"]          # the job: the question is its own turn after it
        item = content.rsplit(f"[{anchor}] id:", 1)[-1].split("\n[", 1)[0]
        mass = 0.98
        if question.startswith("about"):
            kind, options = "identity", ["yes", "no"]
            want = "no" if ("CUE_NO" in item or "Rowan Pell" in item) else "yes"
            if not self.probe_ok and "Rowan" in item:
                want = "no" if want == "yes" else "yes"
            if "CUE_LOWMASS" in item:
                mass = 0.05
        else:
            kind, options = "placement", wiki_gate.LETTERS + ["new"]
            want = "new" if "PLACE_NEW" in item else next((c for c in wiki_gate.LETTERS if f"PLACE_{c}" in item), "A")
            if "CUE_LOWPLACE" in item:
                mass = 0.05
        self.decisions.append({"anchor": anchor, "kind": kind, "want": want, "body": body, "content": content,
                               "probe": "Rowan bakery" in content})
        alts = {o: mass * (0.97 if o == want else 0.01 / len(options)) for o in options}
        alts["<tool_call>"] = 1 - sum(alts.values())
        return [(want, alts)]


@pytest.fixture
def server(monkeypatch, fake_chat):
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "jev_hops", False)  # on by default; the gate with hops has its own test
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    model = _Model()
    fake = fake_chat(model)
    fake.model_cues, fake.decisions = model, model.decisions
    monkeypatch.setattr(wiki_gate, "get_conn", _NoConn)
    return fake


class _NoConn:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _entity(content, entity_type="fact"):
    return {"id": str(uuid.uuid4()), "entity_type": entity_type, "content": content, "keywords": []}


def _recall(monkeypatch, items):
    """Recall returns `items` (entity dicts) as cut previews; the full rows
    come from fetch_members. Records what recall was asked."""
    asked, rows = [], {i["id"]: i for i in items}

    def fake(conn, req, need=None):
        asked.append(req)
        return SimpleNamespace(items=[SimpleNamespace(id=uuid.UUID(i["id"]), entity_type=i["entity_type"],
                                                      content=i["content"][:20] + "--truncated-- get_entity(...)",
                                                      keywords=i["keywords"]) for i in items])

    monkeypatch.setattr(wiki_gate, "assemble_context", fake)
    monkeypatch.setattr(wiki_gate, "fetch_members", lambda conn, ids: [rows[i] for i in reversed(ids)])
    return asked


SHORT_PAGE = "<!-- wiki:meta canonical_name=Calder Labs keywords=calder;robots -->\n# Calder Labs\n> **Summary:** battery testing robots maker.\n"
LONG_PAGE = (SHORT_PAGE + "<!-- section: overview -->\nOverview text.\n<!-- section: timeline -->\n2019 founded.\n"
             "<!-- section: finances -->\nRevenue 41M.\n")


def _turn(members, page=SHORT_PAGE) -> reader.Turn:
    """The gated writer's first request as the router builds it: the job,
    with its members, as the user message."""
    head = (f"You maintain the page Calder Labs.\n### Seed member entities for this job\n"
            f"{wiki_gate.members_block(members)}\n### Current wiki body\n{page}\nThe time is now.")
    return reader.Turn([{"role": "system", "content": SYSTEM}, {"role": "user", "content": head.strip()}], TOOLS)


def _prepare(members, page=SHORT_PAGE, long_page=False):
    sections = parse_sections(page)[1] if long_page else []
    return wiki_gate.prepare(_turn(members, page), "Calder Labs", page, members, sections)


def _job(server):
    return [d for d in server.decisions if not d["probe"]]


def test_only_facts_about_the_subject_stay_and_a_left_out_fact_is_rolled_back(server, monkeypatch):
    m1 = _entity("Calder Labs opened a factory")
    cited = _entity("already on the page")
    f_yes = _entity("Calder Labs hired engineers in May, a long fact that recall only previews")
    f_no, f_low, f_yes2 = _entity("CUE_NO Calder Insurance raised premiums"), _entity("CUE_LOWMASS x"), _entity("Calder Labs won a prize")
    page = SHORT_PAGE + f"<!-- section: overview -->\nText [[ref:{cited['id']}]]\n"
    asked = _recall(monkeypatch, [_entity("a page", "wiki"), _entity("calder", "keyword"), m1, cited,
                                  f_yes, f_no, f_low, f_yes2])
    gate = _prepare([m1], page)
    assert [m["id"] for m in gate.members] == [m1["id"]]
    assert gate.found == [f_yes, f_yes2]                                # full rows, not recall's previews
    assert gate.excluded == [f_no["id"], f_low["id"]]
    asked_about = [d["anchor"] for d in _job(server)]
    assert asked_about == ["f1", "f2", "f3", "f4"]                       # only the found facts: never wiki,
    assert asked[0].queries == ["Calder Labs", "calder", "robots"]       # keyword, member or cited fact
    assert asked[0].include_always_on_rules is False
    # a fact left out is rolled back: the next one follows what was kept
    last = _job(server)[-1]["content"]
    assert f_yes["content"] in last and f_no["content"] not in last and "CUE_LOWMASS" not in last
    assert f_yes["content"] in gate.prompt and f_yes2["content"] in gate.prompt
    assert f_no["id"] not in gate.prompt and f_low["id"] not in gate.prompt
    assert "--truncated--" not in gate.prompt
    assert "SUBJECT: Calder Labs\nSUMMARY: battery testing robots maker." in gate.prompt
    assert gate.sections == {}


def test_each_decision_is_the_writer_s_own_first_request_with_the_question_opening_its_reply(server, monkeypatch):
    m1 = _entity("Calder Labs opened a factory")
    _recall(monkeypatch, [_entity("Calder Labs hired engineers"), _entity("CUE_NO other firm")])
    turn = reader.Turn(_turn([m1]).messages, TOOLS, {"chat_template_kwargs": {"reasoning_effort": "low"}})
    gate = wiki_gate.prepare(turn, "Calder Labs", SHORT_PAGE, [m1], [])
    for d in _job(server):
        b = d["body"]
        assert b["messages"][0] == turn.messages[0] and [m["role"] for m in b["messages"]] == ["system", "user", "assistant"]
        assert b["messages"][1]["content"].startswith(turn.messages[1]["content"])   # the job, then the facts
        assert b["messages"][2] == {"role": "assistant", "content": f"[{d['anchor']}] {wiki_gate.ID_LEAD}"}
        assert b["messages"][1]["content"].endswith("\nAnswer with one word: yes or no.")
        assert b["chat_template_kwargs"] == {"reasoning_effort": "low"}            # the writer's own settings
        assert b["continue_final_message"] is True and "reasoning_effort" not in b
        assert b["tools"] == TOOLS and b["tool_choice"] == "none" and b["max_tokens"] == 1
        # each continues what was kept; so does the writer's own first request
        so_far = d["content"].rsplit(f"\n\n[{d['anchor']}] id:", 1)[0]
        assert gate.prompt.startswith(so_far)


def test_a_server_that_does_not_continue_a_reply_gets_the_question_as_its_own_turn(server, monkeypatch):
    server.continues = False
    m1 = _entity("Calder Labs opened a factory")
    _recall(monkeypatch, [_entity("Calder Labs hired engineers"), _entity("CUE_NO other firm")])
    gate = _prepare([m1])
    assert gate is not None and [f["content"] for f in gate.found] == ["Calder Labs hired engineers"]
    for d in _job(server):
        b = d["body"]
        assert [m["role"] for m in b["messages"]] == ["system", "user", "user"]
        assert b["messages"][2]["content"] == f"Answer with one word: yes or no.\n[{d['anchor']}] {wiki_gate.ID_LEAD}"
        assert b["reasoning_effort"] == "none" and b["stop"] == ["\n"]


def test_the_members_are_never_asked_about_and_never_leave_the_job(server, monkeypatch):
    _recall(monkeypatch, [])
    members = [_entity("Calder Labs opened a factory"), _entity("CUE_NO Calder Insurance raised premiums")]
    gate = _prepare(members)
    assert gate is not None and gate.members == members
    assert _job(server) == []                                             # nothing to decide: no request
    assert members[1]["id"] in gate.prompt and gate.prompt.endswith("(none)")


def test_on_a_long_page_each_member_and_kept_fact_gets_its_section(server, monkeypatch):
    m1, m2, m3 = _entity("PLACE_B opened a factory"), _entity("PLACE_NEW chief executive resigned"), _entity("CUE_LOWPLACE x")
    f_kept, f_out = _entity("PLACE_C revenue up"), _entity("CUE_NO PLACE_A other firm")
    _recall(monkeypatch, [f_kept, f_out])
    gate = _prepare([m1, m2, m3], LONG_PAGE, long_page=True)
    # the unsure one is left to the writer; a found fact left out is never placed
    assert gate.sections == {m1["id"]: "timeline", m2["id"]: "new", f_kept["id"]: "finances"}
    placed = [d["anchor"] for d in _job(server) if d["kind"] == "placement"]
    assert sorted(placed) == ["f1", "m1", "m2", "m3"]


def test_a_page_with_more_sections_than_letters_is_not_placed(server, monkeypatch):
    page = SHORT_PAGE + "".join(f"<!-- section: s{i} -->\nText {i}.\n" for i in range(len(wiki_gate.LETTERS) + 1))
    _recall(monkeypatch, [])
    gate = _prepare([_entity("PLACE_B opened a factory")], page, long_page=True)
    assert gate is not None and gate.sections == {} and _job(server) == []
    assert all(line.startswith("- ") for line in wiki_gate.outline(parse_sections(page)[1]))  # no letters


def test_the_outline_is_lettered_with_each_section_s_first_words():
    lines = wiki_gate.outline(parse_sections(LONG_PAGE)[1])
    assert lines[0].startswith("A. overview: ") and lines[0].endswith(" - Overview text.")
    assert lines[2].startswith("C. finances: ") and lines[2].endswith(" - Revenue 41M.")


def test_the_deadline_leaves_the_job_to_the_full_writer(server, monkeypatch):
    monkeypatch.setattr(wiki_gate, "DEADLINE_S", -1)
    _recall(monkeypatch, [_entity("Calder Labs hired engineers")])
    assert _prepare([_entity("Calder Labs opened a factory")]) is None
    assert _job(server) == []


def test_a_failed_probe_switches_the_gate_off(server, monkeypatch):
    server.model_cues.probe_ok = False
    _recall(monkeypatch, [])
    assert _prepare([_entity("Calder Labs opened a factory")]) is None
    assert not wiki_gate.wanted()                                         # off until re-probed


def test_off_unless_enabled_and_on_any_profile_whose_server_passes_the_probe(server, monkeypatch):
    members = [_entity("Calder Labs opened a factory")]
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", False)
    assert not wiki_gate.wanted() and _prepare(members) is None
    assert server.bodies == []
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")         # a hosted gateway: the probe decides
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-secret-key")
    _recall(monkeypatch, [])
    assert wiki_gate.wanted() and _prepare(members) is not None


def test_off_under_a_profile_that_replaces_the_writer_prompt(server, monkeypatch, tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "wiki_writer.replace.md").write_text("Another contract", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "own")
    assert not wiki_gate.wanted()


def test_every_jev_request_counts_toward_one_in_flight_limit(monkeypatch, fake_chat):
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    def slow(body):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return [("yes", None)]

    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    fake_chat(slow)
    server = jev.Server()
    with ThreadPoolExecutor(max_workers=3 * jev.WORKERS) as pool:     # several features at once
        list(pool.map(lambda _: server.chat([{"role": "user", "content": "q"}]), range(3 * jev.WORKERS)))
    assert 1 < state["max"] <= jev.WORKERS


def test_the_gated_writer_has_only_the_page_tools_and_its_first_request_is_the_sdk_s():
    import asyncio

    from braindb.agent import agent as agent_mod
    gated = agent_mod.get_gated_writer_agent()
    assert {t.name for t in gated.tools} == {"read_wiki_outline", "read_wiki_section", "check_members_cited",
                                             "edit_wiki_section", "delete_wiki_section", "validate_wiki",
                                             "handoff_to_successor", "final_answer"}
    assert gated.instructions == agent_mod._GATED_WRITER_PROMPT and gated is not agent_mod.get_writer_agent()
    turn = asyncio.run(agent_mod.first_turn(gated, "THE JOB"))
    assert turn.messages == [{"content": agent_mod._GATED_WRITER_PROMPT, "role": "system"},
                             {"role": "user", "content": "THE JOB"}]
    assert sorted(t["function"]["name"] for t in turn.tools) == sorted(t.name for t in gated.tools)
