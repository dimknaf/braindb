"""Unit tests for the guards that keep the wiki writer loop terminating.

DB-free. Each test pins one invariant that a live 70-hour run proved was
NOT holding, so a regression here reproduces a failure we have actually
paid for:

- `_claimable()` bounds the LEASE-RECLAIM branch (behavioural proof lives
  in `test_wiki_selfheal_db.py`; the string tests here are cheap
  documentation of the predicate's shape).
- The subagent/writer TOOLSETS are asserted on the BUILT agents, so
  dropping `extra_tools=` from a factory fails these tests — asserting the
  private tuples alone would not.
- Delegation depth is per-run-context: two CONCURRENT delegations must both
  proceed (a process-global counter refuses the second — the exact live
  failure: two sibling delegations 1.8ms apart, the second rejected), while
  a NESTED delegation is still refused.
- `update_entity`'s protected-content map covers wikis (behavioural proof
  through the real tool is in `test_wiki_selfheal_db.py`).

These are guards, not behaviour: none of them judges or shapes content.
The citation predicate itself (`uncited_members`) is DB-backed and tested
in `test_wiki_selfheal_db.py` — the router and the `check_members_cited`
tool both call that single helper, so they cannot drift.
"""
from __future__ import annotations

import asyncio
import json

from agents.tool_context import ToolContext

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
from braindb.services.wiki_jobs import (
    ASSIGNED_LEASE_MIN,
    ASSIGNED_MAX_RECLAIMS,
    _claimable,
)


def _agent_tool_names(agent) -> set[str]:
    return {getattr(t, "name", None) for t in agent.tools}


# ====================================================================== #
# _claimable — the reclaim ceiling (shape; behaviour is DB-tested)        #
# ====================================================================== #

def test_claimable_bounds_the_reclaim_branch_by_attempts():
    """Without this the abandoned-run path is unbounded."""
    sql = _claimable()
    assert f"attempts < {ASSIGNED_MAX_RECLAIMS}" in sql


def test_claimable_still_admits_pending_unconditionally():
    """A fresh job must never be gated by the reclaim ceiling — the cap
    applies only to the branch nothing else bounds."""
    sql = _claimable()
    pending_clause, _, reclaim_clause = sql.partition(" OR ")
    assert "status = 'pending'" in pending_clause
    assert "attempts" not in pending_clause
    assert "attempts" in reclaim_clause


def test_claimable_still_requires_lease_expiry_to_reclaim():
    sql = _claimable()
    assert "assigned_at <" in sql
    assert f"mins => {ASSIGNED_LEASE_MIN}" in sql


def test_claimable_alias_qualifies_every_column():
    """The predicate is interpolated into queries that alias the table;
    an unqualified column there is a runtime SQL error."""
    sql = _claimable("j")
    for col in ("status", "attempts", "assigned_at"):
        assert f"j.{col}" in sql


def test_reclaim_ceiling_sits_above_the_graceful_cap():
    """`release_or_fail_jobs` uses max_attempts=3. The reclaim ceiling must
    be higher, or it would pre-empt normal failure handling."""
    assert ASSIGNED_MAX_RECLAIMS > 3


# ====================================================================== #
# update_entity — protected-content map                                   #
# ====================================================================== #

def test_content_readonly_covers_wiki_and_datasource():
    assert set(tools_mod._CONTENT_READONLY) == {"datasource", "wiki"}


def test_wiki_readonly_message_points_at_the_section_tools():
    """The model has to be told where to go, or it reaches for search_sql."""
    msg = tools_mod._CONTENT_READONLY["wiki"]
    assert "edit_wiki_section" in msg


def test_datasource_message_is_unchanged():
    """Pre-existing behaviour must not drift while generalising the guard."""
    assert tools_mod._CONTENT_READONLY["datasource"] == (
        "datasource bodies are read-only; use notes for analysis"
    )


# ====================================================================== #
# Toolsets — asserted on the BUILT agents, not the private tuples         #
# ====================================================================== #

def test_built_subagent_carries_the_wiki_read_tools():
    """Without these a subagent cannot inspect a page the safe way, and
    falls back to paging the raw body and re-emitting it through
    update_entity — which is how a cited UUID got corrupted live."""
    names = _agent_tool_names(agent_mod.get_subagent())
    assert {"read_wiki_outline", "read_wiki_section",
            "check_members_cited", "validate_wiki"} <= names


def test_built_subagent_has_no_wiki_write_tools():
    """One writer per wiki is what makes the revision CAS meaningful, and a
    subagent cannot hand off, so it has no business holding a revision."""
    names = _agent_tool_names(agent_mod.get_subagent())
    assert "edit_wiki_section" not in names
    assert "delete_wiki_section" not in names
    assert "handoff_to_successor" not in names


def test_built_writer_keeps_every_tool_it_had():
    """Regression guard: the writer's toolset may grow, never shrink."""
    names = _agent_tool_names(agent_mod.get_writer_agent())
    assert {"read_wiki_outline", "read_wiki_section", "check_members_cited",
            "edit_wiki_section", "delete_wiki_section", "validate_wiki",
            "handoff_to_successor"} <= names


def test_delegate_docstring_no_longer_promises_the_full_toolset():
    """The false promise is what made writers delegate edits a subagent
    could not perform."""
    doc = tools_mod.delegate_to_subagent.description or ""
    assert "all the same BrainDB tools" not in doc
    assert "read_wiki_section" in doc


# ====================================================================== #
# Delegation depth — per run-context, behaviourally                       #
# ====================================================================== #

def _ctx(arguments: str) -> ToolContext:
    """A real ToolContext, as the SDK builds one for every tool call — its
    invocation path reads more than the tool name."""
    return ToolContext(context=None, tool_name="delegate_to_subagent",
                       tool_call_id="call_test", tool_arguments=arguments)


def test_concurrent_delegations_proceed_and_nested_is_refused(monkeypatch):
    """THE behavioural test for the ContextVar fix. With the old
    process-global counter, the second CONCURRENT delegation was refused
    ("max delegation depth reached" — observed live, two sibling calls
    1.8ms apart); with a per-context depth both proceed, while a NESTED
    delegation inside a subagent is still bounded at depth 1.

    `run_typed` and `get_subagent` are patched at their import site
    (`braindb.agent.agent` — the tool imports them locally per call), so no
    LLM and no DB are touched.
    """
    from braindb.agent.schemas import SubagentResult

    state = {"active": 0, "max_active": 0, "nested_reply": None}

    async def call_delegate(task: str) -> str:
        args = json.dumps({"task": task})
        return await tools_mod.delegate_to_subagent.on_invoke_tool(_ctx(args), args)

    async def fake_run_typed(task, agent, schema, max_turns=None):
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.05)  # hold the slot so the two runs overlap
        if state["nested_reply"] is None:
            # A subagent trying to delegate further must be refused —
            # this runs inside the caller's depth-1 context.
            state["nested_reply"] = await call_delegate("nested probe")
        state["active"] -= 1
        return SubagentResult(result=f"ok:{task}")

    monkeypatch.setattr(agent_mod, "run_typed", fake_run_typed)
    monkeypatch.setattr(agent_mod, "get_subagent", lambda: object())

    async def main():
        return await asyncio.gather(call_delegate("A"), call_delegate("B"))

    r1, r2 = asyncio.run(main())
    assert "ok:A" in r1, r1
    assert "ok:B" in r2, r2
    # Genuine overlap — a global counter would have refused the second call
    # and max_active would never reach 2.
    assert state["max_active"] == 2
    assert "max delegation depth reached" in (state["nested_reply"] or "")


def test_delegation_depth_defaults_to_zero_and_resets():
    assert tools_mod._depth_var.get() == 0


def test_max_depth_still_one():
    """Bounded delegation is deliberate — the fix was the scope of the
    counter, not the limit."""
    assert tools_mod._MAX_DEPTH == 1


# ====================================================================== #
# Run tag — concurrent runs must be separable in the logs               #
# ====================================================================== #

def test_verbose_tool_lines_carry_the_current_run_tag(monkeypatch, caplog):
    """Audits of concurrent writer+maintainer+subagent logs previously had
    to attribute TOOL lines by argument fingerprint. `run_typed` sets a
    per-run tag; `_verbose` must print it."""
    import logging as _logging
    from braindb.agent.run_state import reset_run_tag, set_run_tag
    from braindb.agent.schemas import SubagentResult

    monkeypatch.setattr(tools_mod.settings, "agent_verbose", True)

    async def fake_run_typed(task, agent, schema, max_turns=None):
        return SubagentResult(result="ok")

    monkeypatch.setattr(agent_mod, "run_typed", fake_run_typed)
    monkeypatch.setattr(agent_mod, "get_subagent", lambda: object())

    token = set_run_tag("tag4242")
    try:
        with caplog.at_level(_logging.INFO, logger="braindb.agent.tools"):
            reply = asyncio.run(
                tools_mod.delegate_to_subagent.on_invoke_tool(
                    _ctx('{"task": "probe"}'), '{"task": "probe"}'))
        assert "ok" in reply
        tool_lines = [r.message for r in caplog.records
                      if "delegate_to_subagent" in r.getMessage()]
        assert tool_lines, "no TOOL lines captured"
        assert any("[tag4242]" in r.getMessage() for r in caplog.records)
    finally:
        reset_run_tag(token)


# ====================================================================== #
# Reasoning effort — wiki-scoped, never the general agent                 #
# ====================================================================== #

def test_reasoning_effort_default_is_a_no_op():
    """Blank default must send nothing, so behaviour is unchanged until an
    operator opts in."""
    from braindb.config import settings
    assert settings.agent_wiki_reasoning_effort == ""
    agent_mod._cache.clear()
    try:
        for factory in (agent_mod.get_agent, agent_mod.get_maintainer_agent,
                        agent_mod.get_writer_agent, agent_mod.get_subagent):
            ms = factory().model_settings
            assert ms.extra_body is None
            assert "reasoning_effort" not in ms.extra_args
    finally:
        agent_mod._cache.clear()


def test_context_window_errors_are_recognised_whatever_the_provider():
    """run_typed re-raises an over-the-window request at once (a retry would
    hit the same wall) and retries any other 400 once. The classifier must
    see vLLM's and the gateways' 'maximum context length' / 'maximum model
    length' messages and OpenAI's `context_length_exceeded` code."""
    from openai import BadRequestError

    class _Response:  # what the client's error reads from an HTTP response
        status_code, headers, request = 400, {}, None

    def err(message, code=None):
        return BadRequestError(message, response=_Response(), body={"code": code} if code else None)

    assert agent_mod._is_context_window(
        err("This model's maximum context length is 40960 tokens. However, you requested 51000 tokens."))
    assert agent_mod._is_context_window(err("request too large", code="context_length_exceeded"))
    assert agent_mod._is_context_window(
        err("The decoder prompt (length 51000) is longer than the maximum model length of 40960."))
    assert not agent_mod._is_context_window(err("Invalid JSON in tool call arguments"))


def test_agents_build_without_a_key_and_are_never_stored_on_openai(monkeypatch):
    """The OpenAI client refuses an empty key when it is built: a missing key
    must fail at the provider on the first request (as it always did), not
    break agent construction. OpenAI's own endpoint keeps chat completions by
    default, so it gets `store=False`; every other endpoint gets no `store`."""
    from braindb.config import settings
    monkeypatch.setattr(settings, "llm_profile", "deepinfra")
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    agent_mod._cache.clear()
    try:
        assert agent_mod.get_agent().model_settings.store is None
        monkeypatch.setattr(settings, "llm_profile", "openai_compatible")
        monkeypatch.setattr(settings, "openai_base_url", "")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent_mod._cache.clear()
        assert agent_mod.get_agent().model_settings.store is False
    finally:
        agent_mod._cache.clear()


def test_reasoning_effort_never_travels_as_a_plain_param():
    """THE regression guard for a live outage. Under LiteLLM a top-level
    `reasoning_effort` was lifted out of extra_args/extra_body and rejected by
    a provider allow-list — 59 triage jobs died this way. It must ride
    nested inside `chat_template_kwargs`, where vLLM reads template
    variables and nothing intercepts or filters it."""
    from braindb.config import settings
    monkey = settings.agent_wiki_reasoning_effort
    try:
        object.__setattr__(settings, "agent_wiki_reasoning_effort", "low")
    except Exception:
        settings.agent_wiki_reasoning_effort = "low"
    agent_mod._cache.clear()
    try:
        ms = agent_mod.get_writer_agent().model_settings
        assert ms.extra_body == {"chat_template_kwargs":
                                 {"reasoning_effort": "low"}}
        # a top-level reasoning_effort is a provider-checked parameter: NEITHER top level
        assert "reasoning_effort" not in ms.extra_body
        assert "reasoning_effort" not in ms.extra_args
    finally:
        settings.agent_wiki_reasoning_effort = monkey
        agent_mod._cache.clear()


def test_reasoning_effort_reaches_wiki_agents_but_not_the_general_agent(monkeypatch):
    """THE scoping guarantee. `get_agent()` is shared by /agent/query AND the
    ingest watcher, whose extraction runs are the most reasoning-dependent
    work in the stack — it must never inherit the wiki setting. This test
    fails if anyone later threads it there."""
    from braindb.config import settings
    monkeypatch.setattr(settings, "agent_wiki_reasoning_effort", "low")
    agent_mod._cache.clear()
    try:
        for factory in (agent_mod.get_maintainer_agent,
                        agent_mod.get_writer_agent, agent_mod.get_subagent):
            ms = factory().model_settings
            assert ms.extra_body == {"chat_template_kwargs":
                                     {"reasoning_effort": "low"}}, factory.__name__
            # the transport deadline must survive alongside it
            assert ms.extra_args.get("timeout") == settings.agent_request_timeout
        assert agent_mod.get_agent().model_settings.extra_body is None
    finally:
        agent_mod._cache.clear()
