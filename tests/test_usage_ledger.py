"""Unit tests for the per-request usage ledger.

DB-free and LLM-free. The ledger exists so that "what did this answer cost?"
has a measured answer: uncached input, output and reasoning tokens of every
LLM call a request made (subagents and retries included), plus its tool
calls. These tests pin the invariants that make those numbers trustworthy:

- LLM usage arrives through `CountdownHooks.on_llm_end` and never breaks a run;
- nested runs add into the ledger the outermost caller installed;
- tool calls are counted whether or not AGENT_VERBOSE is on;
- `/agent/query` records usage in its activity row — on failure too, without
  ever masking the original error — and never in the response.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
import braindb.routers.agent as agent_router
from braindb.agent.hooks import CountdownHooks
from braindb.agent.run_state import (
    add_usage,
    install_usage,
    release_usage,
)


def _response(input_tokens=1000, cached=800, output=50, reasoning=30):
    """The shape the SDK hands to on_llm_end (ModelResponse.usage)."""
    usage = SimpleNamespace(
        requests=1,
        input_tokens=input_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens=output,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    return SimpleNamespace(usage=usage)


def _hooks() -> CountdownHooks:
    return CountdownHooks(max_turns=20, threshold=0)


# ====================================================================== #
# The ledger itself                                                       #
# ====================================================================== #

def test_on_llm_end_adds_each_calls_usage_to_the_ledger():
    ledger, token = install_usage()
    try:
        asyncio.run(_hooks().on_llm_end(None, None, _response()))
        asyncio.run(_hooks().on_llm_end(None, None, _response(500, 0, 10, 0)))
        assert ledger.agent == {"requests": 2, "input": 1500, "cached": 800,
                                "output": 60, "reasoning": 30}
    finally:
        release_usage(token)


def test_on_llm_end_never_raises_on_an_unexpected_response_shape():
    """A future SDK shape change must cost us the numbers, never the run."""
    ledger, token = install_usage()
    try:
        asyncio.run(_hooks().on_llm_end(None, None, object()))
        assert ledger.agent["requests"] == 0
    finally:
        release_usage(token)


def test_nested_install_returns_the_outer_ledger_and_no_token():
    """Subagents and retries must add into the request's ledger, and only
    the outermost caller reports and releases it."""
    outer, token = install_usage()
    try:
        inner, inner_token = install_usage()
        assert inner is outer and inner_token is None
        add_usage("agent", output=7)
        assert outer.agent["output"] == 7
    finally:
        release_usage(token)


def test_add_usage_outside_a_request_is_a_no_op():
    add_usage("agent", output=5)  # nothing installed: must not raise


def test_weighted_is_uncached_input_plus_four_times_output():
    ledger, token = install_usage()
    try:
        ledger.add("agent", input=1000, cached=800, output=50)
        assert "weighted=400" in ledger.line()  # (1000 - 800) + 4 * 50
    finally:
        release_usage(token)


# ====================================================================== #
# Tool calls are counted, verbose or not                                  #
# ====================================================================== #

class _Ctx:
    tool_name = "delegate_to_subagent"


def test_tool_calls_are_counted_even_when_not_verbose(monkeypatch):
    from braindb.agent.schemas import SubagentResult

    async def fake_run_typed(task, agent, schema, max_turns=None):
        return SubagentResult(result="ok")

    monkeypatch.setattr(agent_mod, "run_typed", fake_run_typed)
    monkeypatch.setattr(agent_mod, "get_subagent", lambda: object())
    monkeypatch.setattr(tools_mod.settings, "agent_verbose", False)
    ledger, token = install_usage()
    try:
        asyncio.run(tools_mod.delegate_to_subagent.on_invoke_tool(
            _Ctx(), json.dumps({"task": "probe"})))
        assert ledger.tools == {"delegate_to_subagent": 1}
    finally:
        release_usage(token)


# ====================================================================== #
# /agent/query: usage in the activity row, never in the response          #
# ====================================================================== #

@pytest.fixture
def captured(monkeypatch):
    rows: list[dict] = []

    @contextlib.contextmanager
    def fake_conn():
        yield object()

    monkeypatch.setattr(agent_router, "get_conn", fake_conn)
    monkeypatch.setattr(agent_router, "log_activity",
                        lambda conn, op, details=None, **kw: rows.append({"op": op, **details}))
    return rows


def test_agent_query_records_usage_in_its_activity_row_not_in_the_response(monkeypatch, captured):
    async def fake_query(query, max_turns=None):
        add_usage("agent", requests=3, input=4000, cached=3000, output=900, reasoning=700)
        return {"answer": "a", "max_turns": 20}

    monkeypatch.setattr(agent_router, "run_agent_query", fake_query)
    result = asyncio.run(agent_router.agent_query(agent_router.AgentQueryRequest(query="q")))
    assert result == {"answer": "a", "max_turns": 20}
    assert captured[0]["op"] == "agent_query"
    assert captured[0]["usage"]["agent"]["output"] == 900
    add_usage("agent", output=1)  # the ledger was released: no-op, no leak into the next request


def test_failed_agent_query_records_usage_and_keeps_the_original_error(monkeypatch, captured):
    async def failing_query(query, max_turns=None):
        add_usage("agent", requests=1, output=50)
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_router, "run_agent_query", failing_query)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(agent_router.agent_query(agent_router.AgentQueryRequest(query="q")))
    assert exc.value.status_code == 500 and exc.value.detail == "Agent failed: boom"
    assert captured[0]["error"] == "RuntimeError"
    assert captured[0]["usage"]["agent"]["output"] == 50


def test_a_failing_activity_write_never_masks_the_agent_error(monkeypatch):
    async def failing_query(query, max_turns=None):
        raise RuntimeError("boom")

    @contextlib.contextmanager
    def broken_conn():
        raise OSError("database down")
        yield  # pragma: no cover

    monkeypatch.setattr(agent_router, "run_agent_query", failing_query)
    monkeypatch.setattr(agent_router, "get_conn", broken_conn)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(agent_router.agent_query(agent_router.AgentQueryRequest(query="q")))
    assert exc.value.detail == "Agent failed: boom"
