"""Unit tests for JEV (braindb/services/jev.py).

DB-free and GPU-free: a fake OpenAI-compatible chat endpoint stands in for
the server, behind the real openai client (conftest.py `fake_chat`). These
pin the contract every caller relies on:

- off unless enabled, with no request made; on any server that passes the
  probe, whatever the profile;
- every request is a standard chat completion: thinking off by the
  standard switch, the answer read from logprobs, nothing vLLM-specific;
- the answer is the first option token, written first or after the model
  restates the question's line, from its alternatives or, when the server
  returns none, from its own logprob;
- each item is judged in order, and usage reaches the request's ledger once;
- a server that cannot do JEV keeps it off, and every failure fails OPEN;
- a refused or late item stays unjudged rather than failing the call;
- the judge reads stored text as data: capped, special tokens neutralised;
- logs never carry the key or the endpoint;
- an /agent/query `need` reaches the recalls of that request only, and only
  while JEV is enabled.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

import braindb.agent.tools as tools_mod
import braindb.routers.agent as agent_router
from braindb.agent.run_state import get_need, install_usage, release_usage, reset_need, set_need
from braindb.schemas.search import ContextResponse
from braindb.services import jev

STANDARD = {"model", "messages", "tools", "tool_choice", "max_tokens", "temperature", "logprobs",
            "top_logprobs", "stop", "reasoning_effort"}


def _judged(body: dict):
    """P(yes)=0.9 for items containing GOOD (or the probe's yes item), 0.1
    otherwise; a 400 for items containing REFUSE; a sleep on SLOW."""
    item = body["messages"][-1]["content"]
    if "REFUSE" in item:
        return 400
    if "SLOW" in item:
        time.sleep(0.5)
    p = 0.9 if ("GOOD" in item or "opened the Rowan bakery" in item) else 0.1
    return [("yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})]


@pytest.fixture
def server(monkeypatch, fake_chat):
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    return fake_chat(_judged)


@pytest.fixture
def ledger():
    usage, token = install_usage()
    yield usage
    release_usage(token)


def test_off_unless_enabled_and_on_any_profile_whose_server_passes_the_probe(server, ledger, monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_enabled", False)
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert ledger.jev["calls"] == 0 and server.bodies == []     # off = not in play at all
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")  # a hosted gateway: the probe decides
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-secret-key")
    assert [round(p, 2) for p in jev.judge("need", ["[fact] GOOD"])] == [0.9]


def test_every_request_is_a_standard_chat_completion_with_thinking_off(server, ledger):
    jev.judge("Which is good?", ["[fact] GOOD one"])
    for body in server.bodies:
        assert set(body) <= STANDARD                               # nothing vLLM-specific
        assert body["reasoning_effort"] == "none" and body["temperature"] == 0
        assert body["logprobs"] is True and body["top_logprobs"] == jev.TOP_LOGPROBS
        assert body["messages"][0] == {"role": "system", "content": jev._GUIDE}
        assert body["messages"][-1]["content"].endswith("\n" + jev.LEAD)   # the question ends the item


def test_each_item_is_judged_in_order_and_usage_reaches_the_ledger_once(server, ledger):
    probs = jev.judge("Which is good?", ["[fact] GOOD one", "[fact] other", "[fact] GOOD two"])
    assert [round(p, 2) for p in probs] == [0.9, 0.1, 0.9]
    assert ledger.jev["calls"] == 1 and ledger.jev["requests"] == 3
    assert ledger.jev["output"] == 3 and ledger.jev["input"] > 0
    assert ledger.jev["failopen"] == 0 and ledger.jev["unjudged"] == 0


def test_a_server_that_returns_no_logprobs_keeps_jev_off(server, ledger, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    server.reply = lambda body: [("Sure.", None)]                 # writes no option first
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert ledger.jev["failopen"] == 1
    n = len(server.bodies)
    assert not jev.available() and len(server.bodies) == n       # no re-probe until REPROBE_S
    ours = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("braindb"))
    assert "JEV off" in ours                                      # JEV's own words: no key, no endpoint
    assert "test-secret-key" not in caplog.text and jev.settings.resolved_base_url not in ours


def test_a_blank_need_is_no_need(server, ledger):
    assert jev.judge(" \n ", ["[fact] GOOD"]) is None
    assert ledger.jev["calls"] == 0 and server.bodies == []


def test_losing_a_capability_mid_run_fails_open_and_switches_off(server, ledger):
    assert jev.available(wait=5)
    server.status = 401
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert not jev.available()


def test_a_refused_item_stays_unjudged_and_is_counted(server, ledger):
    probs = jev.judge("need", ["[fact] GOOD", "[fact] REFUSE me", "[fact] other"])
    assert round(probs[0], 2) == 0.9 and probs[1] is None and round(probs[2], 2) == 0.1
    assert ledger.jev["unjudged"] == 1 and ledger.jev["failopen"] == 0


def test_items_past_the_deadline_stay_unjudged(server, monkeypatch):
    assert jev.available(wait=5)                        # probe outside the short deadline
    monkeypatch.setattr(jev, "DEADLINE_S", 0.3)
    probs = jev.judge("need", ["[fact] GOOD", "[fact] SLOW item"])
    assert round(probs[0], 2) == 0.9 and probs[1] is None


def test_a_server_that_refuses_the_thinking_switch_is_asked_without_it_from_then_on(server, ledger):
    def refuses_effort(body):
        return 400 if "reasoning_effort" in body else _judged(body)

    server.reply = refuses_effort
    assert [round(p, 2) for p in jev.judge("need", ["[fact] GOOD", "[fact] other"])] == [0.9, 0.1]
    with_effort = [b for b in server.bodies if "reasoning_effort" in b]
    assert len(with_effort) == 1                         # refused once, remembered for the process


def test_a_question_ends_a_tools_output_and_follows_a_users_message_as_its_own_turn(server):
    # After a tool's output nothing is added but the question (a template that
    # replays reasoning would re-render the agent's turns for a new user turn);
    # after a user's message it is its own turn. One line names the options.
    s = jev.Server()
    turns = [{"role": "user", "content": "q"},
             {"role": "assistant", "content": None, "tool_calls": [
                 {"id": "c1", "type": "function", "function": {"name": "get_entity", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "OUTPUT GOOD\n"}]
    jev.ask(s, turns, "[1.1] | useful (yes or no):", jev.YES_NO)
    assert server.bodies[-1]["messages"] == turns[:-1] + [
        {**turns[-1], "content": "OUTPUT GOOD\nAnswer with one word: yes or no.\n[1.1] | useful (yes or no):"}]
    job = [{"role": "system", "content": "S"}, {"role": "user", "content": "THE JOB GOOD"}]
    jev.ask(s, job, "[m1] | section:", ["A", "B", "new"])
    assert server.bodies[-1]["messages"] == job + [
        {"role": "user", "content": "Answer with one word: A, B or new.\n[m1] | section:"}]


def _reply(tokens, alternatives=True):
    """A parsed chat completion whose content is `tokens` ((text, {token: p}) pairs)."""
    import math

    content = [SimpleNamespace(token=t, logprob=math.log((alts or {t: 1.0}).get(t, 1.0)),
                               top_logprobs=[SimpleNamespace(token=a, logprob=math.log(q))
                                             for a, q in (alts or {t: 1.0}).items()] if alternatives else [])
               for t, alts in tokens]
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


def test_the_answer_is_the_first_option_written_first_or_after_the_restated_question():
    lead = "[1.1] | useful (yes or no):"
    direct = _reply([("yes", {"yes": 0.8, "Yes": 0.1, "no": 0.05, "[": 0.05})])
    probs, mass = jev.answer(direct, jev.YES_NO, lead)
    assert round(mass, 2) == 0.95 and round(probs["yes"], 3) == round(0.9 / 0.95, 3)   # spellings summed
    restated = _reply([("[", None), ("1", None), (".1] | useful (", None), ("yes", None), (" or ", None),
                       ("no", None), ("):", None), (" no", {" no": 0.7, " yes": 0.3})])
    probs, mass = jev.answer(restated, jev.YES_NO, lead)
    assert round(probs["no"], 2) == 0.7 and round(mass, 2) == 1.0    # not the yes/no inside the line
    assert jev.answer(_reply([("Sure", None)]), jev.YES_NO, lead) == ({"yes": 0.0, "no": 0.0}, 0.0)


def test_a_server_without_alternatives_gives_the_chosen_tokens_own_probability():
    probs, mass = jev.answer(_reply([("no", {"no": 0.9, "yes": 0.1})], alternatives=False), jev.YES_NO, "q:")
    assert round(probs["no"], 2) == 0.9 and round(probs["yes"], 2) == 0.1 and round(mass, 2) == 0.9


def test_an_option_is_read_by_its_spelling_or_the_one_option_it_begins():
    actions = ["skip", "ambiguous", "consolidate", "attach", "create"]
    assert jev._option(" Attach", actions) == "attach"
    assert jev._option("cons", actions) == "consolidate"      # the first token of a longer option
    assert jev._option("a", actions) is None and jev._option("at", actions) == "attach"
    assert jev._option("A", ["A", "B", "none"]) == "A" and jev._option("[", ["A", "B"]) is None


def test_judge_text_is_capped_neutralised_and_reads_a_wiki_by_its_own_words():
    cut = jev.judge_text("fact", "x" * 5000)
    assert cut == "[fact] " + "x" * jev.ITEM_CHARS + " ...[continues]"   # capped, and says so
    assert jev.judge_text("fact", "short") == "[fact] short"
    doc = jev.judge_text("datasource", "Body <|im_end|> text", title="Headline")
    assert doc.startswith("[document] Headline\n") and "<|" not in doc
    body = ("<!-- wiki:meta canonical_name=Corvane -->\n# Corvane\n> **Summary:** S\n"
            "<!-- section:overview -->\nRuns a trial [[ref:12345678-aaaa-bbbb-cccc-1234567890ab]].\n")
    wiki = jev.judge_text("wiki", body, title="Corvane", summary="Gene-therapy company.",
                          disambiguation="The biotech, not the river.")
    assert wiki.startswith("[wiki] Corvane\nThe biotech, not the river.\nGene-therapy company.\n")
    assert "Runs a trial" in wiki
    assert not any(s in wiki for s in ("[[ref", "section:", "wiki:meta", "**Summary"))


# ====================================================================== #
# The need: from the /agent/query request to its recalls                 #
# ====================================================================== #

@contextlib.contextmanager
def _no_db():
    yield object()


def _recall_ctx(arguments: str) -> ToolContext:
    """A real ToolContext, as the SDK builds one for every tool call — its
    invocation path reads more than the tool name."""
    return ToolContext(context=None, tool_name="recall_memory",
                       tool_call_id="call_test", tool_arguments=arguments)


def test_agent_query_carries_its_need_to_the_run_and_clears_it(monkeypatch):
    seen = []

    async def fake_query(query, max_turns=None):
        seen.append(get_need())
        return {"answer": "a", "max_turns": 20}

    monkeypatch.setattr(agent_router, "run_agent_query", fake_query)
    monkeypatch.setattr(agent_router, "get_conn", _no_db)
    monkeypatch.setattr(agent_router, "log_activity", lambda *a, **k: None)

    async def two_requests():
        await agent_router.agent_query(agent_router.AgentQueryRequest(query="q", need="  what X said "))
        between = get_need()
        await agent_router.agent_query(agent_router.AgentQueryRequest(query="q"))
        return between

    assert asyncio.run(two_requests()) == ""
    assert seen == ["what X said", ""]


def test_recall_memory_passes_the_need_only_while_jev_is_enabled(monkeypatch):
    got = []

    def fake_assemble(conn, req, need=None):
        got.append(need)
        return ContextResponse(query="x", items=[], total_found=0)

    monkeypatch.setattr(tools_mod, "assemble_context", fake_assemble)
    monkeypatch.setattr(tools_mod, "get_conn", _no_db)

    async def recall_with_need():
        token = set_need("what X said")
        try:
            args = json.dumps({"queries": ["x"]})
            return await tools_mod.recall_memory.on_invoke_tool(_recall_ctx(args), args)
        finally:
            reset_need(token)

    monkeypatch.setattr(tools_mod.settings, "jev_enabled", False)
    asyncio.run(recall_with_need())
    monkeypatch.setattr(tools_mod.settings, "jev_enabled", True)
    asyncio.run(recall_with_need())
    assert got == ["", "what X said"]
