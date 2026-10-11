"""Unit tests for the JEV reader (braindb/services/reader.py) and its wiring
into get_entity, the hooks and run_typed.

DB-free and GPU-free. A fake OpenAI-compatible chat endpoint (conftest.py
`fake_chat`) answers each piece by cues in its text. These pin the contract
the reader relies on:

- get_entity's tool schema is unchanged by its run context; with the reader
  off, without a reading question, or for a later page, get_entity answers
  exactly as before;
- run_typed records a run's conversation only when the reader serves the
  request, and the hooks record it as sent, before any nudge;
- the reader reads INSIDE the agent's turn: each request is the agent's next
  request (the SDK's own messages and tools) with the question line ending
  the tool output, nothing else added; each continues what was kept, which
  the agent's next request then continues too;
- only useful pieces come back, with a note whose spans locate everything
  left out; a low-mass piece stays;
- only the first call of a turn is read, and anything that goes wrong gives
  today's output.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from types import SimpleNamespace

import httpx2
import openai.resources.chat.completions as chat_completions
import pytest
from agents.models.interface import ModelTracing
from agents.run_context import RunContextWrapper
from agents.tool_context import ToolContext

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
from braindb.agent.hooks import CountdownHooks
from braindb.agent.run_state import (
    get_loop,
    install_loop,
    install_usage,
    record_submit,
    release_loop,
    release_usage,
    reset_reading,
    set_reading,
)
from braindb.agent.schemas import AgentAnswer
from braindb.services import jev, reader
from conftest import FakeChat

QUESTION = "When did the Kestrel mill open?"
EID = "00000000-0000-0000-0000-00000000000a"
BODY = (
    "# Kestrel mill\n> **Summary:** notes on the Kestrel mill.\n"
    "<!-- section:history -->\n"
    + "Rain fell on the valley for most of that spring and the river rose. " * 3 + "\n\n"
    + "USEFUL opened in 2018 USEFUL\n\n"
    + "The weir upstream was repaired twice in those years by local crews. " * 3 + "\n"
    "<!-- section:menu -->\n"
    + "Rye loaves, oat rolls and seed bread are baked every morning early. " * 4 + "\n")
ROW = {"id": EID, "entity_type": "wiki", "title": "Kestrel mill", "content": BODY,
       "keywords": ["kestrel"], "notes": None, "embedding": [0.1], "search_vector": "x"}

# ---------------------------------------------------------------- the fake

_QUESTION_LINE = re.compile(r"\[(\d+\.\d+)\] \| (useful|enough to answer) \(yes or no\):$")


class _Model:
    """Answers each piece by cues in it (the text after the asked anchor's
    frame): USEFUL (or the probe's opening) is yes, anything else no;
    ENOUGH answers "enough to answer"; LOWMASS holds little probability;
    NANCUE answers NaN. `probe_ok` False inverts the probe's answers."""

    def __init__(self):
        self.decisions, self.probe_ok = [], True

    def __call__(self, body: dict):
        question, content = FakeChat.asked(body)
        anchor, kind = _QUESTION_LINE.search(question).groups()
        piece = content.rsplit(f"[{anchor}] (", 1)[-1]
        kind = "useful" if kind == "useful" else "enough"
        if kind == "enough":
            want = "yes" if "ENOUGH" in piece else "no"
        else:
            want = "yes" if ("USEFUL" in piece or "opened the Rowan bakery" in piece) else "no"
        if not self.probe_ok and "Rowan" in piece:
            want = "no" if want == "yes" else "yes"
        mass = 0.05 if "LOWMASS" in piece else 0.98
        other = "no" if want == "yes" else "yes"
        alts = {want: mass * 0.97, other: mass * 0.01, "<tool_call>": 1 - mass * 0.98}
        if "NANCUE" in piece:  # a broken forward pass: every logprob is NaN
            alts = {k: float("nan") for k in alts}
        self.decisions.append({"anchor": anchor, "kind": kind, "want": want, "body": body,
                               "probe": "datasource probe" in content})
        return [(want, alts)]


def _fresh_state() -> dict:
    return {"server": None, "off_until": 0.0}


@pytest.fixture
def server(monkeypatch, fake_chat):
    monkeypatch.setattr(jev.settings, "jev_reader", True)
    monkeypatch.setattr(jev.settings, "jev_hops", False)  # on by default; hops have their own tests
    monkeypatch.setattr(reader, "serves", lambda: True)     # the prompt's probe has its own tests
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    model = _Model()
    fake = fake_chat(model)
    fake.model_cues, fake.decisions = model, model.decisions
    monkeypatch.setattr(reader, "_state", _fresh_state())
    monkeypatch.setattr(reader, "PIECE_CHARS", 120)
    monkeypatch.setattr(reader, "MIN_PIECE_CHARS", 30)
    monkeypatch.setattr(tools_mod, "MAX_OUTPUT_CHARS", 400)
    monkeypatch.setattr(tools_mod, "get_conn", _db(ROW))
    return fake


# ------------------------------------------------------ the agent's side


class _Cursor:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        pass

    def fetchone(self):
        return dict(self.row)


def _db(row):
    @contextlib.contextmanager
    def get_conn():
        yield SimpleNamespace(cursor=lambda cursor_factory=None: _Cursor(row))
    return get_conn


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        requests=1, input_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens=10, output_tokens_details=SimpleNamespace(reasoning_tokens=0))


def _call(call_id: str, name: str = "get_entity", **args) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps(args or {"entity_id": EID})}


async def _turn(agent, calls: list[dict]) -> list[dict]:
    """One LLM call as the SDK reports it to the hooks; its response makes `calls`."""
    hooks = CountdownHooks(max_turns=20, threshold=0)
    await hooks.on_llm_start(None, agent, agent.instructions, [{"role": "user", "content": QUESTION}])
    await hooks.on_llm_end(None, agent, SimpleNamespace(usage=_usage(), to_input_items=lambda: calls))
    return calls


async def _get(call_id: str, **args) -> str:
    args = json.dumps({"entity_id": EID, **args})
    ctx = ToolContext(context=None, tool_name="get_entity", tool_call_id=call_id, tool_arguments=args)
    return await tools_mod.get_entity.on_invoke_tool(ctx, args)


def _in_a_run(body, question: str = QUESTION):
    """Run `body(ledger)` as a request with a reading question whose run records its loop."""
    async def go():
        reading_token, loop_token = set_reading(question), install_loop()
        ledger, usage_token = install_usage()
        try:
            return await body(ledger)
        finally:
            release_usage(usage_token)
            release_loop(loop_token)
            reset_reading(reading_token)
    return asyncio.run(go())


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


def _today(offset: int = 0, limit: int | None = None) -> str:
    """get_entity's answer without the reader."""
    d = {k: v for k, v in ROW.items() if k not in ("embedding", "search_vector")}
    if offset == 0 and limit is None:
        return tools_mod._truncate(json.dumps(d, default=str, indent=2))
    chunk, meta = tools_mod.slice_content(d["content"], offset, limit)
    return json.dumps({**d, "content": chunk, "content_meta": meta}, default=str, indent=2)


def _spans(note: str, label: str) -> list[tuple[int, int]]:
    line = next((x for x in note.splitlines() if x.startswith(label)), "")
    return [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+) \(", line)]


# ====================================================================== #
# Wiring                                                                  #
# ====================================================================== #

def test_get_entity_schema_is_unchanged_by_its_run_context():
    assert tools_mod.get_entity.params_json_schema == {
        "additionalProperties": False,
        "properties": {
            "entity_id": {"description": "UUID of the entity.", "title": "Entity Id", "type": "string"},
            "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                      "description": "max chars of this slice (clamped to the server slice max).\n"
                                     "   If offset and limit are both omitted, the full body is returned\n"
                                     "   (legacy behaviour, unchanged).",
                      "title": "Limit"},
            "offset": {"default": 0, "description": "start char of the content slice (default 0).",
                       "title": "Offset", "type": "integer"}},
        "required": ["entity_id", "offset", "limit"],
        "title": "get_entity_args", "type": "object"}


@pytest.mark.parametrize("enabled,off,question,recorded", [
    (True, False, QUESTION, True),
    (True, True, QUESTION, False),           # off until the server is probed again
    (False, False, QUESTION, False),
    (True, False, "", False)])               # no reading question: nothing to read for
def test_run_typed_records_the_loop_only_when_the_reader_serves_the_request(
        monkeypatch, enabled, off, question, recorded):
    import time

    monkeypatch.setattr(jev.settings, "jev_reader", enabled)
    monkeypatch.setattr(jev.settings, "jev_hops", False)  # on by default; hops have their own tests
    monkeypatch.setattr(reader, "_state", {"server": None, "off_until": time.monotonic() + 600 if off else 0.0})
    seen = []

    async def fake_run(starting_agent, input, max_turns, **kwargs):
        seen.append(get_loop())
        record_submit(AgentAnswer(answer="a"))
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(agent_mod.Runner, "run", fake_run)

    async def go():
        token = set_reading(question)
        try:
            await agent_mod.run_typed("q", SimpleNamespace(name="FakeAgent"), AgentAnswer, max_turns=5)
            return get_loop()
        finally:
            reset_reading(token)

    assert asyncio.run(go()) is None                    # released with the run
    assert (seen[0] is not None) is recorded


def test_a_nested_run_records_its_own_loop_and_leaves_its_callers_alone(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_reader", True)
    monkeypatch.setattr(reader, "_state", _fresh_state())
    seen = []

    async def fake_run(starting_agent, input, max_turns, **kwargs):
        seen.append(get_loop())
        if input == "outer":               # a subagent runs inside one of the outer run's tools
            seen[-1].items = ["outer item"]
            await agent_mod.run_typed("inner", SimpleNamespace(name="Sub"), AgentAnswer, max_turns=5)
            seen.append(get_loop())
        record_submit(AgentAnswer(answer=input))
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(agent_mod.Runner, "run", fake_run)

    async def go():
        token = set_reading(QUESTION)
        try:
            return await agent_mod.run_typed("outer", SimpleNamespace(name="Main"), AgentAnswer,
                                             max_turns=5)
        finally:
            reset_reading(token)

    assert asyncio.run(go()).answer == "outer"
    outer, inner, after = seen
    assert inner is not outer and inner.items == [] and after is outer
    assert outer.items == ["outer item"]


def test_the_hooks_record_the_conversation_as_sent_before_any_nudge():
    hooks = CountdownHooks(max_turns=6, threshold=5)   # the turn nudge fires on the first call
    call = _call("c1")
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
    assert (loop.agent, loop.system) == ("the agent", "the system prompt")
    assert loop.response == [call] and used["input"] == 100       # usage still counted


def test_the_reading_question_is_the_need_else_a_query_that_is_a_question():
    assert reader.reading_question("  When did it\n open?  ", "ignored") == "When did it open?"
    assert reader.reading_question(None, "When did it open?") == "When did it open?"
    assert reader.reading_question("", "Extract the facts. " * 40) == ""   # an instruction
    assert reader.reading_question("a<b", None) == "a‹b"               # never prompt structure


def test_only_the_first_call_of_a_turn_that_goes_on_is_read():
    one, two = _call("c1"), _call("c2")
    assert reader.first_call([one], "c1") and not reader.first_call([one], "c2")
    assert reader.first_call([one, two], "c1") and not reader.first_call([one, two], "c2")
    assert not reader.first_call([one, _call("c3", name="final_answer")], "c1")
    assert not reader.first_call([], "c1")


def test_parts_follow_wiki_sections_or_markdown_headings():
    wiki = reader._parts("wiki", BODY)
    assert [n for n, _ in wiki] == ["header", "history", "menu"]
    assert all(BODY[s:].startswith("<!-- section:" + n) for n, s in wiki[1:]) and wiki[0][1] == 0
    marked = reader._parts("wiki", "<!-- section:a -->\nx\n<!-- section:b -->\ny")
    assert marked == [("a", 0), ("b", 21)]
    doc = reader._parts("datasource", "intro\n# One\ntext\n## Two\nmore")
    assert doc == [("text", 0), ("One", 6), ("Two", 17)]
    assert reader._parts("datasource", "no headings at all") == [("text", 0)]


def test_pieces_end_at_the_strongest_boundary_near_their_size(monkeypatch):
    monkeypatch.setattr(reader, "PIECE_CHARS", 100)
    body = "a" * 80 + "\n\n" + "b" * 30 + ". " + "c" * 200
    cuts = reader._cuts(body, [("text", 0)])
    assert reader._cut(body, cuts, 0) == 82                  # the paragraph, past three quarters of a piece
    assert reader._cut("x" * 500, reader._cuts("x" * 500, [("text", 0)]), 0) == 100   # no boundary: the size
    assert reader._cut(body[:110], reader._cuts(body[:110], [("text", 0)]), 0) == 110  # the rest fits: all of it


# ====================================================================== #
# get_entity: unchanged unless the reader reads                           #
# ====================================================================== #

@pytest.mark.parametrize("how", ["off", "no question"])
def test_get_entity_answers_as_before_unless_the_reader_reads(server, monkeypatch, how):
    if how == "off":
        monkeypatch.setattr(jev.settings, "jev_reader", False)
    question = "" if how == "no question" else QUESTION

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), await _get("call_1", offset=0, limit=100), ledger.as_dict()["jev"]

    full, page, used = _in_a_run(body, question)
    assert full == _today() and page == _today(0, 100)
    assert used.get("calls", 0) == 0 and server.bodies == []


def test_a_later_page_is_never_read(server):
    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1", offset=100, limit=100)

    assert _in_a_run(body) == _today(100, 100) and server.bodies == []


# ====================================================================== #
# Reading inside the agent's turn                                         #
# ====================================================================== #

def test_a_long_read_keeps_only_useful_pieces_inside_the_agents_own_turn(server, monkeypatch):
    agent = agent_mod.get_agent()

    async def body(ledger):
        response = await _turn(agent, [_call("call_1")])
        out = await _get("call_1", offset=0, limit=400)    # a first page of a whole output starts it
        output = {"type": "function_call_output", "call_id": "call_1", "output": out}
        sent = await _sent_by_the_agent(agent, [{"role": "user", "content": QUESTION}] + response
                                        + [output], monkeypatch)
        return out, sent, ledger.as_dict()["jev"]

    out, sent, used = _in_a_run(body)

    # What the agent gets: the header, the useful piece, a note locating the rest.
    head, note = out.split("\nKept ", 1)
    assert head.startswith(f'wiki {EID} "Kestrel mill"') and f"Question: {QUESTION}" in head
    assert all(w in head for w in ("opened", "2018")) and "seed bread" not in head
    dropped = _spans("Kept " + note, "Not kept:")
    kept_chars = int(re.search(r"\((\d+) of (\d+) chars\)", note).group(1))
    assert all("USEFUL" not in BODY[a:b] for a, b in dropped)
    assert all(b1 <= a2 for (_, b1), (a2, _) in zip(dropped, dropped[1:]))  # in order, apart
    assert kept_chars + sum(b - a for a, b in dropped) == len(BODY)   # every char accounted for

    # Each request IS the agent's next request, as the SDK sends it, the tool
    # output so far as its last message, and the question opening the
    # model's reply, which the server continues: the answer is ONE token,
    # and the agent's own settings are unchanged (no thinking switch).
    asked = [d for d in server.decisions if not d["probe"]]
    reads = [d for d in asked if d["kind"] == "useful"]
    assert len(reads) >= 4
    assert [d["anchor"] for d in asked if d["kind"] == "enough"] == [d["anchor"] for d in reads[:-1] if d["want"] == "yes"]
    tool = {k: v for k, v in sent["messages"][-1].items() if k != "content"}
    for d in asked:
        b = d["body"]
        assert b["messages"][:-2] == sent["messages"][:-1]
        assert {k: v for k, v in b["messages"][-2].items() if k != "content"} == tool
        lead = f"[{d['anchor']}] " + (reader.LEAD if d["kind"] == "useful" else reader.ENOUGH_LEAD)
        assert b["messages"][-1] == {"role": "assistant", "content": lead}
        assert b["messages"][-2]["content"].endswith("\nAnswer with one word: yes or no.")
        assert b["continue_final_message"] is True and b["add_generation_prompt"] is False
        assert b["tools"] == sent["tools"] and b["tool_choice"] == "none" and "reasoning_effort" not in b
        assert b["max_tokens"] == 1 and b["temperature"] == 0

    # Each request continues what was kept, and so does the agent's next one:
    # its output begins with everything every decision was asked after.
    for d in reads:
        so_far = d["body"]["messages"][-2]["content"].rsplit(f"[{d['anchor']}] (", 1)[0]
        assert out.startswith(so_far)
    assert (used["calls"], used["requests"], used["failopen"]) == (1, len(asked), 0)


def test_a_server_that_does_not_continue_a_reply_is_asked_the_standard_way(server, monkeypatch):
    # The question then ends the tool output, nothing else added, thinking off
    # by the standard switch; the read keeps the same pieces.
    server.continues = False

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1")

    out = _in_a_run(body)
    assert all(w in out.split("\nKept ", 1)[0] for w in ("opened", "2018"))
    for d in [d for d in server.decisions if not d["probe"]]:
        b = d["body"]
        lead = f"[{d['anchor']}] " + (reader.LEAD if d["kind"] == "useful" else reader.ENOUGH_LEAD)
        assert b["messages"][-1]["role"] == "tool"
        assert b["messages"][-1]["content"].endswith("\nAnswer with one word: yes or no.\n" + lead)
        assert b["reasoning_effort"] == "none" and "continue_final_message" not in b
        assert b["max_tokens"] == jev.ANSWER_TOKENS and b["stop"] == ["\n"]


def _small_turn() -> reader.Turn:
    """A short conversation whose next turn reads ROW."""
    return reader.Turn(
        messages=[{"role": "user", "content": QUESTION},
                  {"role": "assistant", "content": None, "tool_calls": [
                      {"id": "c1", "type": "function", "function": {"name": "get_entity", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": ""}],
        tools=[])


def _read(turn: reader.Turn, row: dict = ROW) -> str | None:
    async def body(ledger):
        return await asyncio.to_thread(reader.read, turn, row, QUESTION)
    return _in_a_run(body)


def test_a_piece_whose_options_hold_little_probability_stays(server, monkeypatch):
    row = {**ROW, "content": BODY.replace("seed bread", "seed bread LOWMASS")}
    monkeypatch.setattr(tools_mod, "get_conn", _db(row))

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert "LOWMASS" in out.split("\nKept ", 1)[0] and used["unjudged"] >= 1


def test_reading_stops_once_what_was_kept_answers_the_question(server, monkeypatch):
    # The piece with the answer says it is enough: nothing after it is read.
    row = {**ROW, "content": BODY.replace("USEFUL opened in 2018 USEFUL", "USEFUL ENOUGH opened in 2018 USEFUL ENOUGH")}
    out = _read(_small_turn(), row)
    asked = [d for d in server.decisions if not d["probe"]]
    assert [d["kind"] for d in asked][-2:] == ["useful", "enough"] and asked[-1]["want"] == "yes"
    head, note = out.split("\nKept ", 1)
    assert "USEFUL ENOUGH" in head and "seed bread" not in head
    unread = _spans("Kept " + note, "Not read (enough for the question):")
    assert unread and unread[-1][1] == len(row["content"]) and "seed bread" in row["content"][unread[0][0]:]
    assert "Read any of these with get_entity(entity_id, offset, limit)." in note


def test_not_enough_or_too_little_probability_reads_on(server, monkeypatch):
    # "enough" answered no, or with little probability on its options: reading goes on to the end.
    for text in ("USEFUL opened in 2018 USEFUL", "USEFUL ENOUGH LOWMASS opened in 2018 USEFUL ENOUGH LOWMASS"):
        server.decisions.clear()
        out = _read(_small_turn(), {**ROW, "content": BODY.replace("USEFUL opened in 2018 USEFUL", text)})
        assert "Not read" not in out
        assert [d for d in server.decisions if d["kind"] == "enough" and not d["probe"]]


def test_reading_stops_where_the_agent_would_run_out_of_room(server, monkeypatch):
    assert _read(_small_turn()) is not None
    reads = [d for d in server.decisions if not d["probe"] and d["kind"] == "useful"]
    kept = next(i for i, d in enumerate(reads) if d["want"] == "yes")
    assert kept + 1 < len(reads)                           # a piece follows the kept one
    b = reads[kept]["body"]
    size = len(json.dumps(b["messages"]) + json.dumps(b.get("tools") or [])) // 4   # as the server counted it
    # A window just short of the next piece with the room the agent keeps: the reader stops there.
    monkeypatch.setattr(reader, "WINDOW_TOKENS", size + reader.ROOM + 1)
    server.decisions.clear()
    out = _read(_small_turn())
    unread = _spans(out[out.index("\nKept "):], "Not read (the context limit):")
    assert unread and unread[0][0] > 0 and unread[-1][1] == len(BODY)
    assert "opened" in out.split("\nKept ", 1)[0]       # what was kept before it stays


# ====================================================================== #
# Skips and fallbacks: today's output                                     #
# ====================================================================== #

@pytest.mark.parametrize("calls,call_id", [
    (["call_0", "call_1"], "call_1"),     # not the first call of its turn
    (["call_1", "final"], "call_1"),      # a turn that finalises has no next turn
    (["call_1", "handoff"], "call_1")])   # nor one that hands off
def test_a_call_whose_output_does_not_lead_the_next_turn_is_not_read(server, calls, call_id):
    async def body(ledger):
        names = {"final": "final_answer", "handoff": "handoff_to_successor"}
        await _turn(agent_mod.get_agent(), [_call(c, name=names.get(c, "get_entity")) for c in calls])
        return await _get(call_id), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and used["skipped"] == 1 and used["calls"] == 0


def test_a_failed_probe_keeps_the_reader_off(server):
    server.model_cues.probe_ok = False

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and (used["calls"], used["failopen"]) == (1, 1)
    assert not reader.wanted()


def test_a_turn_that_cannot_be_built_falls_back(server, monkeypatch):
    no_model = SimpleNamespace(instructions="the system prompt")   # e.g. a changed SDK shape

    async def body(ledger):
        await _turn(no_model, [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and (used["calls"], used["failopen"]) == (1, 1)


def test_the_run_context_never_reaches_the_tool_log(server, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(tools_mod.settings, "agent_verbose", True)
    asyncio.run(_get("call_1", offset=100, limit=100))
    assert f'"entity_id": "{EID}"' in caplog.text and "ToolContext" not in caplog.text


def test_a_first_page_of_any_size_reads_once_and_later_requests_are_as_before(server):
    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        first = await _get("call_1", offset=0, limit=100)         # a short first page starts the read
        await _turn(agent_mod.get_agent(), [_call("call_2")])
        span = await _get("call_2", offset=0, limit=100)          # then an exact span at 0 is as it is
        await _turn(agent_mod.get_agent(), [_call("call_3")])
        again = await _get("call_3")                              # and so is a full read
        return first, span, again, get_loop().read

    first, span, again, read = _in_a_run(body)
    assert first.startswith("wiki ") and read == {EID}
    assert span == _today(0, 100) and again == _today()


def test_a_read_that_reads_nothing_falls_back(server, monkeypatch):
    monkeypatch.setattr(reader, "DEADLINE_S", -1)   # out of time before the first piece

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and (used["calls"], used["failopen"]) == (1, 1) and reader.wanted()


def test_a_body_too_long_to_read_this_way_is_skipped(server, monkeypatch):
    monkeypatch.setattr(reader, "MAX_PIECES", 1)

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and used["skipped"] == 1 and used["calls"] == 0


def test_a_probe_that_cannot_finish_switches_the_reader_off(server, monkeypatch):
    def unreachable(body):
        raise httpx2.ConnectError("down")

    server.reply = unreachable

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert out == _today() and (used["calls"], used["failopen"]) == (1, 1)
    assert not reader.wanted()                     # off until it is probed again


def test_a_subagent_run_while_the_reader_is_off_never_writes_its_callers_record(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_reader", True)
    monkeypatch.setattr(jev.settings, "jev_hops", False)  # on by default; hops have their own tests
    monkeypatch.setattr(reader, "_state", _fresh_state())
    seen = []

    async def fake_run(starting_agent, input, max_turns, **kwargs):
        seen.append(get_loop())
        if input == "outer":
            monkeypatch.setattr(jev.settings, "jev_reader", False)   # turned off mid-request
            await agent_mod.run_typed("inner", SimpleNamespace(name="Sub"), AgentAnswer, max_turns=5)
            seen.append(get_loop())
        record_submit(AgentAnswer(answer=input))
        return SimpleNamespace(final_output="ok")

    monkeypatch.setattr(agent_mod.Runner, "run", fake_run)

    async def go():
        token = set_reading(QUESTION)
        try:
            return await agent_mod.run_typed("outer", SimpleNamespace(name="Main"), AgentAnswer,
                                             max_turns=5)
        finally:
            reset_reading(token)

    assert asyncio.run(go()).answer == "outer"
    outer, inner, after = seen
    assert outer is not None and inner is None and after is outer   # the subagent records nothing


@pytest.mark.parametrize("on,passes", [(False, True), (True, True), (True, False)])
def test_only_the_agents_that_read_carry_the_reading_section_and_only_when_it_serves(monkeypatch, on, passes):
    def probe():
        if not passes:
            raise jev.Unavailable("the model does not judge the probe pieces reliably")

    monkeypatch.setattr(jev.settings, "jev_reader", on)
    monkeypatch.setattr(jev.settings, "jev_hops", False)  # on by default; hops have their own tests
    monkeypatch.setattr(reader, "_state", _fresh_state())
    monkeypatch.setattr(reader, "ready", probe)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")   # any profile: the probe decides
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-secret-key")
    monkeypatch.setattr(agent_mod, "_cache", {})
    for agent in (agent_mod.get_agent(), agent_mod.get_subagent()):
        if on and passes:
            assert agent.instructions == (agent_mod.SYSTEM_PROMPT.rstrip() + "\n\n"
                                          + agent_mod._READER_SECTION)
        else:
            assert agent.instructions == agent_mod.SYSTEM_PROMPT       # byte-identical when off
    assert agent_mod.get_writer_agent().instructions == agent_mod.SYSTEM_PROMPT  # never the writer


def test_a_server_that_fails_the_probe_gets_the_plain_prompt_and_no_probe_per_request(monkeypatch):
    # The switches are on by default, and any configured server is asked:
    # the reading examples go in only once it has passed the reader's probe.
    monkeypatch.setattr(jev.settings, "jev_reader", True)
    monkeypatch.setattr(jev.settings, "jev_hops", True)
    monkeypatch.setattr(reader, "_state", _fresh_state())
    probes = []

    def fails():
        probes.append(1)
        raise jev.Unavailable("the server did not return the requested logprobs")

    monkeypatch.setattr(reader, "ready", fails)
    assert agent_mod._reading_prompt() == agent_mod.SYSTEM_PROMPT
    assert agent_mod._reading_prompt() == agent_mod.SYSTEM_PROMPT
    assert len(probes) == 1 and not reader.wanted()        # off until it is probed again


def test_the_reading_examples_go_in_once_the_server_passes_the_probe(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_reader", True)
    monkeypatch.setattr(jev.settings, "jev_hops", True)
    monkeypatch.setattr(reader, "_state", _fresh_state())
    monkeypatch.setattr(reader, "ready", lambda: None)
    assert agent_mod._reading_prompt() == "\n\n".join(
        [agent_mod.SYSTEM_PROMPT.rstrip(), agent_mod._READER_SECTION, agent_mod._HOPS_SECTION])


def test_the_reading_examples_say_to_answer_the_question_line_in_plain_text():
    # The agents reply only with tool calls, but a question line ending a
    # tool's output is answered in plain text: one word, the same for every model.
    for section in (agent_mod._READER_SECTION, agent_mod._HOPS_SECTION):
        assert "your reply is only its answer, yes or no, as plain text" in section


def test_a_piece_answered_with_nan_logprobs_stays_unjudged(server, monkeypatch):
    # NaN fails every "mass < MIN_MASS" comparison; it must count as no answer, never as "no".
    row = {**ROW, "content": BODY.replace("seed bread", "seed bread NANCUE")}
    monkeypatch.setattr(tools_mod, "get_conn", _db(row))

    async def body(ledger):
        await _turn(agent_mod.get_agent(), [_call("call_1")])
        return await _get("call_1"), ledger.as_dict()["jev"]

    out, used = _in_a_run(body)
    assert "NANCUE" in out.split("\nKept ", 1)[0] and used["unjudged"] >= 1
