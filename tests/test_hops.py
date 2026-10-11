"""Unit tests for JEV hops (braindb/services/hops.py) and their wiring into
recall_memory and the wiki gate.

DB-free and GPU-free. A fake vLLM endpoint renders turns with a template
shaped like the served model's (tool output trimmed and wrapped, a reasoning
line in the system block unless thinking is off, every assistant turn opened
with a think block), tokenizes one token per character with the special
markers and "\\n\\n" as single tokens, keeps a hybrid model's cache (only the
last whole block a request ended in), and answers each neighbour by a cue in
its text. A fake graph stands in for the relations. These pin the contract:

- recall_memory's tool schema is unchanged by its run context; with hops off,
  without a reading question, or for a call that does not lead the next
  turn, recall_memory answers exactly as before;
- hops judge INSIDE the agent's turn: each request is the agent's next turn
  up to the tool output, then one token with thinking off;
- only the neighbours judged useful stay, and only they are followed to the
  next hop; nothing recall showed is removed; nothing kept gives recall's
  output as it was;
- a neighbour whose options hold little probability is rolled back; a request
  the server renders differently falls back and leaves hops on;
- with hops on, the wiki gate offers the members' neighbours after recall's.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
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

BLOCK = 64
QUESTION = "Where did Kel Aster train?"

# ---------------------------------------------------------------- the fake

_SPECIAL = ["<|im_start|>", "<|im_end|>", "<tool_response>", "</tool_response>", "<think>", "</think>"]
_WORDS = [" yes", " Yes", " no", " No"]
_IDS = {w: i for i, w in enumerate(_SPECIAL + _WORDS + ["\n\n"], start=1)}


def _tok(text: str) -> list[int]:
    if text in _WORDS:
        return [_IDS[text]]
    ids, i = [], 0
    while i < len(text):
        mark = next((s for s in _SPECIAL + ["\n\n"] if text.startswith(s, i)), None)
        if mark:
            ids.append(_IDS[mark])
            i += len(mark)
        else:
            ids.append(1000 + ord(text[i]))
            i += 1
    return ids


def _text(ids: list[int]) -> str:
    back = {v: k for k, v in _IDS.items()}
    return "".join(back.get(i, chr(i - 1000) if i >= 1000 else "?") for i in ids)


def _render(messages, tools=None, kwargs=None, add_generation_prompt=False,
            continue_final_message=False) -> list[int]:
    kwargs = kwargs or {}
    thinking = kwargs.get("enable_thinking", True) is not False
    out = "<|im_start|>system\n"
    if thinking:
        out += f"Reasoning effort is set to {kwargs.get('reasoning_effort', 'xhigh')}.\n\n"
    if tools:
        out += "# Tools " + json.dumps(tools, sort_keys=True) + "\n\n"
    rest = messages
    if messages and messages[0]["role"] == "system":
        out += (messages[0]["content"] or "").strip()
        rest = messages[1:]
    out += "<|im_end|>\n"
    for i, m in enumerate(rest):
        content = (m.get("content") or "")
        content = content.strip() if isinstance(content, str) else json.dumps(content)
        last = i == len(rest) - 1
        if m["role"] == "user":
            out += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif m["role"] == "assistant":
            out += f"<|im_start|>assistant\n<think>\n{(m.get('reasoning_content') or '').strip()}\n</think>\n\n{content}"
            for call in m.get("tool_calls") or []:
                out += f"<tool_call>{call['function']['name']} {call['function']['arguments']}</tool_call>"
            if not (last and continue_final_message):
                out += "<|im_end|>\n"
        elif m["role"] == "tool":
            if i == 0 or rest[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += f"\n<tool_response>\n{content}\n</tool_response>"
            if last or rest[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")
    return _tok(out)


class _Cache:
    def __init__(self):
        self.points: list[tuple[int, ...]] = []

    def use(self, prompt: list[int], store: bool = True) -> int:
        cached = max((len(p) for p in self.points if tuple(prompt[:len(p)]) == p), default=0)
        if store and len(prompt) >= BLOCK:
            self.points.append(tuple(prompt[:len(prompt) // BLOCK * BLOCK]))
        return cached


class _Response:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class _FakeVLLM:
    """Answers each neighbour (or the reader's probe piece) by cues in its
    text: USEFUL (or the probe's opening) is yes, anything else no, LOWMASS
    holds little probability. `mismatch` makes the server render a hop's
    request differently from what was assembled."""

    def __init__(self, model: str):
        self.model, self.mismatch = model, False
        self.decisions, self.warms, self.cache = [], [], _Cache()

    def request(self, method, url, **kwargs):
        body = kwargs.get("json") or {}
        if url.endswith("/v1/models"):
            return _Response(200, {"data": [{"id": self.model, "max_model_len": 200_000}]})
        if url.endswith("/tokenize"):
            if "messages" in body:
                ids = _render(body["messages"], body.get("tools"), body.get("chat_template_kwargs"),
                              body.get("add_generation_prompt", False), body.get("continue_final_message", False))
                if self.mismatch and body["messages"][-1]["content"].startswith("[h"):
                    ids = ids + [999]
                return _Response(200, {"tokens": ids})
            return _Response(200, {"tokens": _tok(body["prompt"])})
        assert url.endswith("/v1/completions"), url
        prompt = body["prompt"]
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": self.cache.use(prompt)}}
        if "logprob_token_ids" not in body:
            self.warms.append(prompt)
            return _Response(200, {"choices": [{"text": "x"}], "usage": usage})
        text = _text(prompt)
        lead = text.rsplit("</think>\n\n", 1)[-1]
        anchor = re.match(r"\[([\w.]+)\]", lead).group(1)
        content = text.rsplit("<tool_response>\n", 1)[-1].split("\n</tool_response>", 1)[0]
        item = content.rsplit(f"[{anchor}] ", 1)[-1]
        want = "yes" if ("USEFUL" in item or "opened the Rowan bakery" in item) else "no"
        mass = 0.05 if "LOWMASS" in item else 0.98
        top = {f"token_id:{i}": math.log(mass * (0.97 if _WORDS[i - 7].strip().lower() == want else 0.01))
               for i in body["logprob_token_ids"]}
        self.decisions.append({"prompt": prompt, "anchor": anchor, "want": want, "lead": lead, "content": content,
                               "cached": usage["prompt_tokens_details"]["cached_tokens"],
                               "probe": "datasource probe" in content})
        return _Response(200, {"choices": [{"logprobs": {"top_logprobs": [top]}}], "usage": usage})


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
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_hops", True)
    monkeypatch.setattr(jev.settings, "jev_reader", False)
    monkeypatch.setattr(reader, "serves", lambda: True)  # the prompt's probe has its own tests (test_reader.py)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    fake = _FakeVLLM(jev.settings.resolved_agent_model)
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(jev, "BLOCK", BLOCK)
    monkeypatch.setattr(reader, "_state", {"server": None, "options": None, "window": 0, "consts": None,
                                           "leads": {}, "off_until": 0.0})
    monkeypatch.setattr(reader, "ROOM", 200)
    monkeypatch.setattr(hops, "_state", {"off_until": 0.0})
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
    assert out == today and server.decisions == [] and server.warms == []
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
    assert (used["calls"], used["failopen"]) == (1, 0) and used["requests"] == 1 + len(asked)


def test_each_hop_decision_is_the_agents_next_turn_and_one_token(server):
    _in_a_run([_call("call_1")])
    asked = [d for d in server.decisions if not d["probe"]]
    last = len(server.warms[0]) // BLOCK * BLOCK              # the warm request's checkpoint
    for d in asked:
        assert d["lead"] == f"[{d['anchor']}] | useful (yes or no):"
        assert _text(d["prompt"]).endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                                           + d["lead"])
        assert "Reasoning effort is set to" in _text(d["prompt"])  # the agent's own template arguments
        assert d["cached"] >= last                              # recall's output is computed once
        assert d["content"].startswith("Found 2 items:")


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


def test_a_request_the_server_renders_differently_falls_back_and_hops_stay_on(server):
    server.mismatch = True
    today = _today()
    out, used = _in_a_run([_call("call_1")])
    assert out == today and used["failopen"] == 1 and hops.wanted()


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
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
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
