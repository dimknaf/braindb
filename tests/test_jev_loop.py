"""Unit tests for in-loop JEV (braindb/agent/jev_loop.py and the sheet in
braindb/services/jev.py).

DB-free and GPU-free: retrieval is faked, and a fake vLLM chat endpoint
answers the sheets. These pin the contract in-loop JEV relies on:

- `recall_memory`'s tool schema is unchanged by its run-context parameter;
- `run_typed` records a run's conversation only when in-loop JEV serves the
  request, and the hooks record it as the model gets it, before any nudge;
- with no record, recall's output is exactly what it always was; with one,
  items are labelled, and an item an earlier turn's recall showed is named
  on one line instead of shown again;
- the sheet request IS the agent's next turn up to the recall's output, as
  the SDK's own model would send it, plus the opened answer turn; the
  answers reach the agent as one line that code writes;
- only the first call of a turn is judged, only whole entries are judged,
  and a recall whose sheet does not run goes out exactly as without JEV.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
from types import SimpleNamespace
from uuid import UUID

import openai.resources.chat.completions as chat_completions
import pytest
from agents.models.interface import ModelTracing
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
from braindb.agent import jev_loop
from braindb.agent.hooks import CountdownHooks
from braindb.agent.run_state import (
    get_loop,
    install_loop,
    install_usage,
    record_submit,
    release_loop,
    release_usage,
    reset_need,
    set_need,
)
from braindb.agent.schemas import AgentAnswer
from braindb.schemas.search import ContextResponse, SearchResultItem
from braindb.services import jev

NEED = "The date the Rowan bakery first opened."
QUESTION = [{"role": "user", "content": "When did the Rowan bakery open?"}]


def _item(n: int, content: str) -> SearchResultItem:
    return SearchResultItem(
        id=UUID(int=n), entity_type="fact", title=None, content=content, summary=None,
        keywords=[f"k{n}"], importance=0.5, source="user-stated", notes=None,
        search_score=0.5, effective_importance=0.5, final_rank=1.0 - n / 100)


class _Retrieval:
    """Stands in for `assemble_context`: each recall gets the next batch."""

    def __init__(self, *batches: list[SearchResultItem]):
        self.batches = list(batches)

    def __call__(self, conn, req, need=None):
        items = self.batches.pop(0)
        return ContextResponse(query=req.queries[0], items=items, total_found=len(items))


@contextlib.contextmanager
def _no_db():
    yield object()


class _Response:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


def _token(text: str) -> dict:
    return {"token": text, "logprob": 0.0, "top_logprobs": [{"token": text, "logprob": 0.0}]}


class _FakeChat:
    """A vLLM chat endpoint that answers sheets: yes with P 0.9 for an item
    whose entry holds GOOD (or the probe's yes item), else no with P(yes)
    0.002. `mode` breaks it: "no-logprobs", "shuffled" (keys out of label
    order), "inverted" (every answer flipped) or "refuse" (HTTP 400)."""

    def __init__(self, model: str):
        self.model, self.sheets, self.mode = model, [], "ok"

    def request(self, method, url, **kwargs):
        if url.endswith("/v1/models"):
            return _Response(200, {"data": [{"id": self.model}]})
        assert url.endswith("/v1/chat/completions"), url
        body = kwargs["json"]
        self.sheets.append(body)
        if self.mode == "refuse":
            return _Response(400)
        labels = body["response_format"]["json_schema"]["schema"]["required"]
        recall = [m for m in body["messages"] if m["role"] == "tool"][-1]["content"]
        entries = {m.group(1): m.group(0) for m in re.finditer(
            r"(?ms)^\[([A-Za-z])\] \[.*?(?=^\[[A-Za-z]\] \[|\Z)", recall)}
        answers, tokens = {}, [_token("{")]
        for label in reversed(labels) if self.mode == "shuffled" else labels:
            good = any(s in entries.get(label, "") for s in ("GOOD", "opened the Rowan bakery"))
            p = (0.9 if good else 0.002) if self.mode != "inverted" else (0.002 if good else 0.9)
            answers[label] = "yes" if p > 0.5 else "no"
            tokens += [_token('"'), _token(label), _token('":'), _token(' "'),
                       {"token": answers[label], "logprob": math.log(max(p, 1 - p)),
                        "top_logprobs": [{"token": "yes", "logprob": math.log(p)},
                                         {"token": "no", "logprob": math.log(1 - p)}]},
                       _token('",')]
        choice = {"message": {"role": "assistant", "content": None,
                              "reasoning_content": json.dumps(answers)},
                  "logprobs": None if self.mode == "no-logprobs" else {"content": tokens}}
        return _Response(200, {"choices": [choice], "usage": {
            "prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": 4200},
            "completion_tokens": len(tokens)}})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "jev_mode", "sheet")
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    fake = _FakeChat(model=jev.settings.resolved_agent_model)
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(jev, "_judge", None)
    monkeypatch.setattr(jev, "_off_until", 0.0)
    monkeypatch.setattr(tools_mod, "get_conn", _no_db)
    return fake


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        requests=1, input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens=10, output_tokens_details=SimpleNamespace(reasoning_tokens=0))


async def _turn(hooks: CountdownHooks, agent, items: list, call_ids: list[str]) -> list[dict]:
    """One LLM call, as the SDK reports it to the hooks, whose response calls
    recall_memory once per id. Returns the response's items."""
    await hooks.on_llm_start(None, agent, agent.instructions, list(items))
    calls = [{"type": "function_call", "call_id": c, "name": "recall_memory",
              "arguments": json.dumps({"queries": ["Rowan bakery"]})} for c in call_ids]
    await hooks.on_llm_end(None, agent, SimpleNamespace(usage=_usage(),
                                                        to_input_items=lambda: calls))
    return calls


async def _recall(call_id: str) -> str:
    args = json.dumps({"queries": ["Rowan bakery"]})
    ctx = ToolContext(context=None, tool_name="recall_memory", tool_call_id=call_id,
                      tool_arguments=args)
    return await tools_mod.recall_memory.on_invoke_tool(ctx, args)


class _Sent(Exception):
    pass


async def _sent_by_the_agent(agent, items: list, monkeypatch) -> dict:
    """The request the agent's own model makes for `items`: the SDK builds
    it, and the client call is captured instead of made."""
    captured = {}

    async def capture(self, *args, **kwargs):
        captured.update(kwargs)
        raise _Sent()

    monkeypatch.setattr(chat_completions.AsyncCompletions, "create", capture)
    with contextlib.suppress(_Sent):
        await agent.model.get_response(
            system_instructions=agent.instructions, input=items,
            model_settings=agent.model_settings,
            tools=await agent.get_all_tools(RunContextWrapper(context=None)),
            output_schema=None, handoffs=[], tracing=ModelTracing.DISABLED,
            previous_response_id=None, conversation_id=None, prompt=None)
    return captured


def _in_a_run(body, need: str = NEED):
    """Run `body(ledger)` as a request with `need` whose run records its loop."""
    async def go():
        need_token, loop_token = set_need(need), install_loop()
        ledger, usage_token = install_usage()
        try:
            return await body(ledger)
        finally:
            release_usage(usage_token)
            release_loop(loop_token)
            reset_need(need_token)
    return asyncio.run(go())


def _plain(items: list[SearchResultItem], monkeypatch) -> str:
    """Recall's output for `items` in a run that records no loop."""
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    return asyncio.run(_recall("call_plain"))


def _one_turn(call_ids: list[str], recall_id: str, agent=None):
    """A run's first turn calls recall_memory once per id; returns the body
    for `_in_a_run` that answers `recall_id` and reports the ledger."""
    async def body(ledger):
        await _turn(CountdownHooks(max_turns=20, threshold=0), agent or agent_mod.get_agent(),
                    QUESTION, call_ids)
        return await _recall(recall_id), ledger.as_dict()["jev"], get_loop()
    return body


# ====================================================================== #
# Wiring: schema, loop record, hooks                                      #
# ====================================================================== #

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
        "required": ["queries", "max_results"],
        "title": "recall_memory_args", "type": "object"}


@pytest.mark.parametrize("enabled,mode,profile,need,recorded", [
    (True, "sheet", "vllm_workstation_qwen", NEED, True),
    (True, "render", "deepinfra", NEED, True),             # render needs no model call
    (True, "sheet", "deepinfra", NEED, False),             # a sheet needs a self-hosted server
    (True, "filter", "vllm_workstation_qwen", NEED, False),
    (False, "sheet", "vllm_workstation_qwen", NEED, False),
    (True, "sheet", "vllm_workstation_qwen", "", False)])
def test_run_typed_records_the_loop_only_when_in_loop_jev_serves_the_request(
        monkeypatch, enabled, mode, profile, need, recorded):
    monkeypatch.setattr(jev.settings, "jev_enabled", enabled)
    monkeypatch.setattr(jev.settings, "jev_mode", mode)
    monkeypatch.setattr(jev.settings, "llm_profile", profile)
    seen = []

    async def fake_run(starting_agent, input, max_turns, **kwargs):
        seen.append(get_loop())
        record_submit(AgentAnswer(answer="a"))
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(agent_mod.Runner, "run", fake_run)

    async def go():
        token = set_need(need)
        try:
            await agent_mod.run_typed("q", SimpleNamespace(name="FakeAgent"), AgentAnswer, max_turns=5)
            return get_loop()
        finally:
            reset_need(token)

    assert asyncio.run(go()) is None                    # released with the run
    assert (seen[0] is not None) is recorded


def test_a_nested_run_records_its_own_loop_and_leaves_its_callers_alone(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "jev_mode", "render")
    seen = []

    async def fake_run(starting_agent, input, max_turns, **kwargs):
        seen.append(get_loop())
        if input == "outer":               # a subagent runs inside one of the outer run's tools
            seen[-1].shown["x"] = 1
            await agent_mod.run_typed("inner", SimpleNamespace(name="Sub"), AgentAnswer, max_turns=5)
            seen.append(get_loop())
        record_submit(AgentAnswer(answer=input))
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(agent_mod.Runner, "run", fake_run)

    async def go():
        token = set_need(NEED)
        try:
            return await agent_mod.run_typed("outer", SimpleNamespace(name="Main"), AgentAnswer,
                                             max_turns=5)
        finally:
            reset_need(token)

    assert asyncio.run(go()).answer == "outer"
    outer, inner, after = seen
    assert inner is not outer and inner.shown == {} and after is outer and outer.shown == {"x": 1}


def test_the_hooks_record_the_conversation_as_sent_before_any_nudge():
    hooks = CountdownHooks(max_turns=6, threshold=5)   # the turn nudge fires on the first call
    call = {"type": "function_call", "call_id": "c1", "name": "recall_memory", "arguments": "{}"}
    response = SimpleNamespace(usage=_usage(), to_input_items=lambda: [call])

    async def go():
        token = install_loop()
        ledger, usage_token = install_usage()
        try:
            sent = [{"role": "user", "content": "q"}]
            await hooks.on_llm_start(None, "the agent", "the system prompt", sent)
            await hooks.on_llm_end(None, "the agent", response)
            return get_loop(), sent, ledger.as_dict()["agent"]
        finally:
            release_usage(usage_token)
            release_loop(token)

    loop, sent, used = asyncio.run(go())
    assert len(sent) == 2 and "tool call" in sent[1]["content"]  # the nudge reached the model...
    assert loop.items == [{"role": "user", "content": "q"}]       # ...but not the record
    assert (loop.turn, loop.agent, loop.system) == (1, "the agent", "the system prompt")
    assert loop.response == [call] and used["input"] == 100       # usage still counted


# ====================================================================== #
# Recall rendering                                                        #
# ====================================================================== #

def test_without_a_loop_record_recall_output_is_unchanged(server, monkeypatch):
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval([_item(1, "GOOD one")]))
    token = set_need(NEED)  # JEV on in sheet mode with a need, but this run records no loop
    try:
        out = asyncio.run(_recall("call_1"))
    finally:
        reset_need(token)
    assert out == ("Found 1 items:\n[fact] rank=0.990 src=user-stated\n"
                   "  id: 00000000-0000-0000-0000-000000000001\n  content: GOOD one\n  keywords: k1")
    assert server.sheets == []


def test_render_mode_labels_items_and_names_items_shown_earlier_on_one_line(server, monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_mode", "render")
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(
        [_item(1, "one"), _item(2, "two")],     # turn 1
        [_item(2, "two"), _item(3, "three")],   # turn 2: item 2 again
        [_item(3, "three")]))                   # turn 2, a second call: item 3 again, same turn
    agent = agent_mod.get_agent()

    async def body(ledger):
        hooks = CountdownHooks(max_turns=20, threshold=0)
        await _turn(hooks, agent, QUESTION, ["call_1"])
        first = await _recall("call_1")
        await _turn(hooks, agent, QUESTION, ["call_2", "call_3"])
        return first, await _recall("call_2"), await _recall("call_3")

    first, second, third = _in_a_run(body)
    assert "\n[A] [fact]" in first and "\n[B] [fact]" in first
    assert second.startswith("Found 2 items:\nAlso found, already shown by an earlier recall above: "
                             "00000000-0000-0000-0000-000000000002 (fact)\n[A] [fact]")
    assert "content: two" not in second and "content: three" in second
    assert "content: three" in third    # shown by a call of the same turn: not "earlier"
    assert "Relevance check" not in first + second + third
    assert server.sheets == []          # render mode never calls the model


# ====================================================================== #
# The sheet                                                               #
# ====================================================================== #

def test_the_sheet_is_the_agents_next_turn_and_its_answers_become_one_line(server, monkeypatch):
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(
        [_item(1, "GOOD: opened in 2018"), _item(2, "sells rye bread"), _item(3, "GOOD: owner")]))
    agent = agent_mod.get_agent()

    async def body(ledger):
        hooks = CountdownHooks(max_turns=20, threshold=0)
        response = await _turn(hooks, agent, QUESTION, ["call_1"])
        out = await _recall("call_1")
        output = {"type": "function_call_output", "call_id": "call_1", "output": out}
        return out, await _sent_by_the_agent(agent, QUESTION + response + [output], monkeypatch), \
            ledger.as_dict()["jev"]

    out, sent, used = _in_a_run(body)
    recall, line = out.split("\n\nRelevance check of ")
    assert line == "[A]-[C]: likely useful for this request: [A], [C]; the rest look unrelated."
    probe, sheet = server.sheets                                  # the probe, then the sheet
    assert sheet["messages"][:-2] == sent["messages"][:-1]       # the agent's own next turn...
    assert sheet["messages"][-2] == {**sent["messages"][-1], "content": recall}  # ...to the recall
    assert sheet["messages"][-1] == {"role": "assistant",
                                     "content": f"{jev._NO_THINK}{jev.SHEET_LEAD} {NEED}\n"}
    assert sheet["tools"] == sent["tools"] and sheet["tool_choice"] == "none"
    assert probe["tool_choice"] == "none"                        # the probe is shaped the same
    assert sheet["continue_final_message"] is True and sheet["add_generation_prompt"] is False
    assert sheet["response_format"]["json_schema"]["schema"]["required"] == ["A", "B", "C"]
    assert "chat_template_kwargs" not in sheet                   # the agent sends none either
    assert (used["calls"], used["requests"], used["failopen"], used["unjudged"]) == (1, 1, 0, 0)
    assert (used["input"], used["cached"]) == (5000, 4200)


def test_a_recall_with_nothing_useful_says_so(server, monkeypatch):
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval([_item(1, "sells rye bread")]))
    out, _, _ = _in_a_run(_one_turn(["call_1"], "call_1"))
    assert out.endswith("\n\nRelevance check of [A]: none looks useful for this request.")


def test_only_whole_entries_are_judged_and_counted_as_shown(server, monkeypatch):
    items = [_item(n, "GOOD " + "x" * 1000) for n in range(1, 31)]  # too long for 8,000 chars
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    out, _, loop = _in_a_run(_one_turn(["call_1"], "call_1"))
    judged = server.sheets[-1]["response_format"]["json_schema"]["schema"]["required"]
    whole = [label for n, label in enumerate(jev_loop.LABELS[:30], start=1)
             if f"  keywords: k{n}\n" in out or out.endswith(f"  keywords: k{n}")]
    assert 0 < len(judged) < 30 and judged == whole
    assert sorted(loop.shown) == sorted(str(UUID(int=n)) for n in range(1, len(judged) + 1))


# ====================================================================== #
# A sheet that does not run leaves the recall exactly as without JEV      #
# ====================================================================== #

def test_a_recall_that_is_not_the_first_call_of_its_turn_is_not_judged(server, monkeypatch):
    items = [_item(1, "GOOD")]
    plain = _plain(items, monkeypatch)
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    out, used, _ = _in_a_run(_one_turn(["call_1", "call_2"], "call_2"))
    assert out == plain
    assert used["skipped"] == 1 and used["calls"] == 0 and server.sheets == []


@pytest.mark.parametrize("mode", ["no-logprobs", "inverted"])
def test_a_server_that_cannot_answer_the_probe_sheet_keeps_jev_off(server, monkeypatch, mode):
    items = [_item(1, "GOOD")]
    plain = _plain(items, monkeypatch)
    server.mode = mode
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    out, used, _ = _in_a_run(_one_turn(["call_1"], "call_1"))
    assert out == plain
    assert used["failopen"] == 1 and len(server.sheets) == 1     # the probe alone
    assert not jev.available()


@pytest.mark.parametrize("mode", ["shuffled", "refuse"])
def test_a_sheet_that_comes_back_unusable_leaves_the_recall_unchanged(server, monkeypatch, mode):
    assert jev.available(wait=5)                       # probed while the server behaves
    server.mode = mode
    items = [_item(1, "GOOD"), _item(2, "x")]
    plain = _plain(items, monkeypatch)
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    out, used, _ = _in_a_run(_one_turn(["call_1"], "call_1"))
    assert out == plain
    assert used["unjudged"] == 2 and used["failopen"] == 0
    assert jev.available()                             # one bad sheet does not switch JEV off


def test_a_sheet_that_cannot_be_built_leaves_the_recall_unchanged(server, monkeypatch):
    items = [_item(1, "GOOD")]
    plain = _plain(items, monkeypatch)
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval(items))
    no_model = SimpleNamespace(instructions="the system prompt")   # e.g. a changed SDK shape
    out, used, _ = _in_a_run(_one_turn(["call_1"], "call_1", agent=no_model))
    assert out == plain
    assert used["calls"] == 1 and used["failopen"] == 1 and server.sheets == []


def test_the_run_context_never_reaches_the_tool_log(server, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(tools_mod.settings, "agent_verbose", True)
    monkeypatch.setattr(tools_mod, "assemble_context", _Retrieval([_item(1, "GOOD")]))
    asyncio.run(_recall("call_1"))
    assert '"queries": ["Rowan bakery"]' in caplog.text and "ToolContext" not in caplog.text


