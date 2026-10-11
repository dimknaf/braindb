"""Unit tests for JEV hops (braindb/services/hops.py) and their wiring into
recall_memory and the wiki gate.

DB-free and GPU-free. A fake OpenAI-compatible chat endpoint (conftest.py
`fake_chat`) answers each neighbour by a cue in its text, and a fake graph
stands in for the relations. These pin the contract:

- recall_memory's tool schema is unchanged by its run context; with hops off,
  without a reading question, or for a call that does not lead the next
  turn, recall_memory answers exactly as before;
- hops judge INSIDE the agent's turn: each request is the agent's next
  request with the question line ending the tool output, thinking off;
- only the neighbours judged useful stay, and only they are followed to the
  next hop; nothing recall showed is removed; nothing kept gives recall's
  output as it was;
- a neighbour whose options hold little probability is rolled back;
- with hops on, the wiki gate offers the members' neighbours after recall's.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
from braindb.agent.hooks import CountdownHooks
from braindb.agent.run_state import (
    install_loop,
    install_usage,
    release_loop,
    release_usage,
    reset_reading,
    set_reading,
)
from braindb.services import hops, jev, reader, wiki_gate

QUESTION = "Where did Kel Aster train?"

# ---------------------------------------------------------------- the fake

_QUESTION_LINE = re.compile(r"\[([\w.]+)\] \| useful \(yes or no\):$")


class _Model:
    """Answers each neighbour (or the reader's probe piece) by cues in its
    text: USEFUL (or the probe's opening) is yes, anything else no, LOWMASS
    holds little probability."""

    def __init__(self):
        self.decisions = []

    def __call__(self, body: dict):
        content = body["messages"][-1]["content"]
        anchor = _QUESTION_LINE.search(content).group(1)
        item = content.rsplit(f"[{anchor}] ", 2)[-2] if content.count(f"[{anchor}] ") > 1 else content
        want = "yes" if ("USEFUL" in item or "opened the Rowan bakery" in item) else "no"
        mass = 0.05 if "LOWMASS" in item else 0.98
        other = "no" if want == "yes" else "yes"
        self.decisions.append({"anchor": anchor, "want": want, "body": body, "content": content,
                               "probe": "datasource probe" in content})
        return [(want, {want: mass * 0.97, other: mass * 0.01, "<tool_call>": 1 - mass * 0.98})]


# ------------------------------------------------------------ the memory

def _node(text: str, entity_type: str = "fact") -> dict:
    return {"id": str(uuid.uuid4()), "entity_type": entity_type, "title": None, "content": text, "keywords": []}


SHOWN = [_node("Kel Aster opened the Rowan bakery in 2018."), _node("The Rowan bakery sells rye bread.")]
N1 = _node("USEFUL Tarn Gazette: Kel Aster trained for four years at the Pell bakery school.", "datasource")
N2 = _node("The Tarn Street market runs on Saturdays.")
N3 = _node("USEFUL Kel Aster's apprenticeship ended in 2016.")
M1 = _node("USEFUL The Pell bakery school is in Brenno.")
X1 = _node("USEFUL a neighbour only of a dropped one: never offered")
GRAPH = {SHOWN[0]["id"]: [N1, N2], SHOWN[1]["id"]: [N3], N1["id"]: [M1], N2["id"]: [X1], N3["id"]: []}


def _neighbours(conn, ids, exclude, limit, types=None):
    out, seen = [], set()
    for k, i in enumerate(ids):
        for n in GRAPH.get(i, []):
            if n["id"] in exclude or n["id"] in seen or (types and n["entity_type"] not in types):
                continue
            seen.add(n["id"])
            out.append({**n, "relation_type": "elaborates", "outgoing": True, "via": i, "ord": k + 1, "weight": 0.5})
    return out[:limit]


def _recall_result(nodes=SHOWN):
    items = [SimpleNamespace(id=uuid.UUID(n["id"]), entity_type=n["entity_type"], final_rank=0.9 - k / 10,
                             source="user-stated", content=n["content"], keywords=[]) for k, n in enumerate(nodes)]
    return SimpleNamespace(total_found=len(items), items=items, always_on_rules=[])


@pytest.fixture
def server(monkeypatch, fake_chat):
    monkeypatch.setattr(jev.settings, "jev_hops", True)
    monkeypatch.setattr(jev.settings, "jev_reader", False)
    monkeypatch.setattr(reader, "serves", lambda: True)  # the prompt's probe has its own tests (test_reader.py)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    model = _Model()
    fake = fake_chat(model)
    fake.decisions = model.decisions
    monkeypatch.setattr(hops, "neighbours", _neighbours)
    monkeypatch.setattr(hops, "get_conn", contextlib.nullcontext)
    monkeypatch.setattr(tools_mod, "get_conn", contextlib.nullcontext)
    monkeypatch.setattr(tools_mod, "assemble_context", lambda conn, req, need=None: _recall_result())
    return fake


# ------------------------------------------------------ the agent's side

def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        requests=1, input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens=10, output_tokens_details=SimpleNamespace(reasoning_tokens=0))


def _call(call_id: str, name: str = "recall_memory") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps({"queries": ["Kel Aster"]})}


async def _turn(agent, calls: list[dict]) -> None:
    """One LLM call as the SDK reports it to the hooks; its response makes `calls`."""
    hooks = CountdownHooks(max_turns=20, threshold=0)
    await hooks.on_llm_start(None, agent, agent.instructions, [{"role": "user", "content": QUESTION}])
    await hooks.on_llm_end(None, agent, SimpleNamespace(usage=_usage(), to_input_items=lambda: calls))


async def _recall(call_id: str) -> str:
    args = json.dumps({"queries": ["Kel Aster"]})
    ctx = ToolContext(context=None, tool_name="recall_memory", tool_call_id=call_id, tool_arguments=args)
    return await tools_mod.recall_memory.on_invoke_tool(ctx, args)


def _in_a_run(calls: list[dict], call_id: str = "call_1", question: str = QUESTION):
    """recall_memory in a request with a reading question whose run records its loop."""
    async def go():
        reading_token, loop_token = set_reading(question), install_loop()
        ledger, usage_token = install_usage()
        try:
            await _turn(agent_mod.get_agent(), calls)
            return await _recall(call_id), ledger.as_dict()["jev"]
        finally:
            release_usage(usage_token)
            release_loop(loop_token)
            reset_reading(reading_token)
    return asyncio.run(go())


def _today() -> str:
    """recall_memory's answer without hops."""
    async def go():
        return await _recall("call_x")
    return asyncio.run(go())


# ================================================================= tests

def test_recall_memory_schema_is_unchanged_by_its_run_context():
    assert tools_mod.recall_memory.params_json_schema == {
        "additionalProperties": False,
        "properties": {
            "max_results": {"default": 30, "description": "Max items to return (1-100, default 30).",
                            "title": "Max Results", "type": "integer"},
            "queries": {"description": "List of search queries. Prefer 2-4 short focused queries\n"
                                       "over one long phrase. Include the bare keyword(s) of the\n"
                                       "subject you're investigating as standalone queries.",
                        "items": {"type": "string"}, "title": "Queries", "type": "array"}},
        "required": ["queries", "max_results"], "title": "recall_memory_args", "type": "object"}


@pytest.mark.parametrize("how", ["hops off", "no question", "not the first call"])
def test_recall_answers_as_before_unless_hops_serve_the_call(server, monkeypatch, how):
    today = _today()
    if how == "hops off":
        monkeypatch.setattr(jev.settings, "jev_hops", False)
        out, used = _in_a_run([_call("call_1")])
    elif how == "no question":
        out, used = _in_a_run([_call("call_1")], question="")
    else:
        out, used = _in_a_run([_call("call_0", "quick_search"), _call("call_1")])
    assert out == today and server.bodies == []
    assert used.get("calls", 0) == 0


def test_hops_keep_only_useful_neighbours_and_follow_only_those(server):
    today = _today()
    out, used = _in_a_run([_call("call_1")])
    assert out.startswith(today)                                    # nothing recall showed is removed
    hopped = out[len(today):]
    assert N1["id"] in hopped and N3["id"] in hopped and M1["id"] in hopped
    assert N2["id"] not in hopped                                   # judged no: rolled back
    asked = [d for d in server.decisions if not d["probe"]]
    assert [d["anchor"] for d in asked] == ["h1.1", "h1.2", "h1.3", "h2.1"]
    assert not any(X1["content"] in d["content"] for d in asked)    # a dropped one is never followed
    assert "Kept 3 of 4 neighbours offered" in out
    assert (used["calls"], used["failopen"]) == (1, 0) and used["requests"] == len(asked)


def test_each_hop_decision_is_the_agents_next_request_with_the_question_ending_the_output(server):
    out, _ = _in_a_run([_call("call_1")])
    asked = [d for d in server.decisions if not d["probe"]]
    for d in asked:
        b, tool = d["body"], d["body"]["messages"][-1]
        assert tool["role"] == "tool" and tool["content"].startswith("Found 2 items:")
        assert tool["content"].endswith(f"\n[{d['anchor']}] | useful (yes or no):")
        assert [m["role"] for m in b["messages"][:2]] == ["system", "user"]          # the agent's own turns
        assert b["messages"][0]["content"] == agent_mod.get_agent().instructions
        assert b["tools"] and b["tool_choice"] == "none" and b["reasoning_effort"] == "none"
        assert b["max_tokens"] == jev.ANSWER_TOKENS and b["stop"] == ["\n"]
        so_far = tool["content"].rsplit(f"[{d['anchor']}] ", 2)[0]
        assert out.startswith(so_far)                         # each continues what was kept


def test_an_item_recalls_cut_left_out_counts_as_not_shown(server, monkeypatch):
    # Recall's output is cut at MAX_OUTPUT_CHARS: an item past the cut was not
    # shown, so hops may offer it (here N1, recalled but cut away).
    monkeypatch.setattr(tools_mod, "assemble_context", lambda conn, req, need=None: _recall_result(SHOWN + [N1]))
    monkeypatch.setattr(tools_mod, "MAX_OUTPUT_CHARS", len(_today().split(f"  id: {N1['id']}")[0]) + 5)
    out, _ = _in_a_run([_call("call_1")])
    visible = out.split("\n\nNeighbours of the items above")[0]
    assert N1["id"] not in visible and N1["id"] in out          # offered and kept by hops


def test_nothing_kept_gives_recalls_output_as_it_was(server, monkeypatch):
    for n in (N1, N3):
        monkeypatch.setitem(n, "content", n["content"].replace("USEFUL", "background"))
    today = _today()
    out, used = _in_a_run([_call("call_1")])
    assert out == today and used["failopen"] == 0
    assert [d["anchor"] for d in server.decisions if not d["probe"]] == ["h1.1", "h1.2", "h1.3"]


def test_a_neighbour_whose_options_hold_little_probability_is_rolled_back(server, monkeypatch):
    monkeypatch.setitem(N1, "content", N1["content"] + " LOWMASS")
    out, _ = _in_a_run([_call("call_1")])
    assert N1["id"] not in out and N3["id"] in out and M1["id"] not in out   # M1 hung on N1


def test_a_failed_probe_gives_recalls_output_and_switches_hops_off(server):
    server.reply = lambda body: [("Sure.", None)]               # answers no option
    today = _today()
    out, used = _in_a_run([_call("call_1")])
    assert out == today and used["failopen"] == 1 and not hops.wanted()


def test_neighbours_most_shown_items_link_to_come_first_then_pages_and_documents():
    row = lambda i, links, t, ord_, w=0.5: {"id": i, "entity_type": t, "links": links, "ord": ord_, "weight": w}
    rows = [row("fact-strong", 1, "fact", 1, 0.9), row("page-3", 3, "wiki", 7), row("doc-3", 3, "datasource", 2),
            row("fact-3", 3, "fact", 1), row("doc-1", 1, "datasource", 9), row("fact-1-late", 1, "fact", 4, 0.9)]
    assert [r["id"] for r in hops._first(rows, 10)] == ["doc-3", "page-3", "fact-3", "doc-1", "fact-strong",
                                                        "fact-1-late"]
    assert [r["id"] for r in hops._first(rows, 2)] == ["doc-3", "page-3"]
    assert [r["id"] for r in hops._first(rows, 10, types=("fact",))] == ["fact-3", "fact-strong", "fact-1-late"]


def test_a_page_is_offered_by_its_name_summary_and_sections_a_document_with_what_links_it():
    page = {"id": "p", "entity_type": "wiki", "title": "Rowan bakery", "relation_type": "summarises",
            "outgoing": False, "via": "3f2a1b0c-0000", "links": 4,
            "content": ("<!-- wiki:meta canonical_name=Rowan bakery -->\n# Rowan bakery\n"
                        "> **Summary:** a bakery on Tarn Street.\n<!-- section: overview -->\nText.\n"
                        "<!-- section: breads -->\nRye.\n")}
    item = hops._item("h1.1", page)
    assert item.startswith("[h1.1] [wiki] summarises to 3f2a1b0c (and 3 more of the items above)\n")
    assert "page: Rowan bakery - a bakery on Tarn Street." in item and "sections: overview; breads" in item
    assert "wiki:meta" not in item and item.endswith("\n")
    doc = {"id": "d", "entity_type": "datasource", "title": "Tarn Gazette", "content": "Kel Aster trained at Pell.",
           "relation_type": "derived_from", "outgoing": True, "via": "5d4c3b2a-0000", "links": 1,
           "via_content": "Kel Aster opened the Rowan bakery."}
    item = hops._item("h1.2", doc)
    assert item.startswith("[h1.2] [datasource] derived_from from 5d4c3b2a\n")
    assert "content: Tarn Gazette Kel Aster trained at Pell." in item
    assert "linked from: Kel Aster opened the Rowan bakery." in item


def test_the_answering_agent_carries_the_hops_section_only_when_they_are_on(monkeypatch):
    monkeypatch.setattr(reader, "serves", lambda: True)  # a server that passes the probe
    for on in (True, False):
        monkeypatch.setattr(jev.settings, "jev_hops", on)
        monkeypatch.setattr(jev.settings, "jev_reader", False)
        prompt = agent_mod._reading_prompt()
        assert ("## Neighbours after a recall" in prompt) is on
        assert prompt == (f"{agent_mod.SYSTEM_PROMPT.rstrip()}\n\n{agent_mod._HOPS_SECTION}" if on
                          else agent_mod.SYSTEM_PROMPT)


def test_the_wiki_gate_offers_the_members_neighbours_after_recalls_with_hops_on(monkeypatch):
    member = _node("Calder Labs opened a factory")
    found, near = _node("Calder Labs hired engineers"), _node("Calder Labs' factory makes test robots")
    rows = {n["id"]: n for n in (found, near)}
    monkeypatch.setattr(wiki_gate, "assemble_context", lambda conn, req, need=None: SimpleNamespace(
        items=[SimpleNamespace(id=uuid.UUID(found["id"]), entity_type="fact")]))
    monkeypatch.setattr(wiki_gate, "fetch_members", lambda conn, ids: [rows[i] for i in ids if i in rows])
    asked = []

    def neighbours(conn, ids, exclude, limit, types=None):
        asked.append((ids, exclude, limit, types))
        return [{**near, "relation_type": "elaborates", "outgoing": True, "via": ids[0], "ord": 1, "weight": 0.5}]

    monkeypatch.setattr(hops, "neighbours", neighbours)
    page = "# Calder Labs\n> **Summary:** robots.\n"
    monkeypatch.setattr(jev.settings, "jev_hops", False)
    assert [f["id"] for f in wiki_gate._found(None, "Calder Labs", page, [member])] == [found["id"]]
    assert asked == []
    monkeypatch.setattr(jev.settings, "jev_hops", True)
    assert [f["id"] for f in wiki_gate._found(None, "Calder Labs", page, [member])] == [found["id"], near["id"]]
    ids, exclude, limit, types = asked[0]
    assert ids == [member["id"]] and found["id"] in exclude and member["id"].lower() in exclude
    assert limit == wiki_gate.PACK_HOPS and types == ("fact", "thought")
