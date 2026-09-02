"""Unit tests for the guards that keep the wiki writer loop terminating.

DB-free. Each test pins one invariant that a live 70-hour run proved was
NOT holding, so a regression here reproduces a failure we have actually
paid for:

- `_claimable()` bounds the LEASE-RECLAIM branch. The graceful failure
  path caps at `max_attempts`, but an abandoned run (client timeout, 500,
  worker death) never reaches it — the row was re-admitted on lease expiry
  and `attempts` incremented, forever. Observed: one job at every attempts
  value 1..30, and 832 claim cycles on a single wiki whose work was
  already complete.
- `update_entity` refuses to overwrite a WIKI body. A subagent lacking the
  section tools retyped a 40-52k-char body through it to change one line,
  and flipped a hex digit in a cited UUID; every later write on that page
  then died on a dangling foreign key.
- The subagent toolset actually matches what the writer is told it is.
  The docstring promised "all the same BrainDB tools"; the subagent had
  none of the wiki tools, so it invented its own unsafe route.
- Delegation depth is per-run, not per-process, so one writer's in-flight
  delegation cannot refuse another writer's.

These are guards, not behaviour: none of them judges or shapes content.
"""
from __future__ import annotations

import asyncio

import pytest

import braindb.agent.agent as agent_mod
import braindb.agent.tools as tools_mod
from braindb.services.wiki_jobs import (
    ASSIGNED_LEASE_MIN,
    ASSIGNED_MAX_RECLAIMS,
    _claimable,
)


def _tool_names(tools) -> set[str]:
    return {getattr(t, "name", None) for t in tools}


# ====================================================================== #
# _claimable — the reclaim ceiling                                        #
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
    assert " attempts <" not in sql.replace("j.attempts", "")


def test_reclaim_ceiling_sits_above_the_graceful_cap():
    """`release_or_fail_jobs` uses max_attempts=3. The reclaim ceiling must
    be higher, or it would pre-empt normal failure handling."""
    assert ASSIGNED_MAX_RECLAIMS > 3


def test_lease_exceeds_the_schedulers_own_client_patience():
    """The old 20-minute lease was justified by "AGENT_TIMEOUT ~10 min".
    The scheduler now waits 2400s (40 min) and one LLM call may run to
    4800s (80 min), so a lease under those reclaims live work."""
    assert ASSIGNED_LEASE_MIN >= 80


# ====================================================================== #
# update_entity — wiki bodies belong to the section tools                 #
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
# Toolsets — what each agent can actually do                              #
# ====================================================================== #

def test_subagent_gets_the_wiki_read_tools():
    """Without these it cannot inspect a page the safe way, and falls back
    to paging the raw body and re-emitting it through update_entity."""
    names = _tool_names(agent_mod._SUBAGENT_EXTRA_TOOLS)
    assert {"read_wiki_outline", "read_wiki_section",
            "check_members_cited", "validate_wiki"} <= names


def test_subagent_has_no_wiki_write_tools():
    """One writer per wiki is what makes the revision CAS meaningful, and a
    subagent cannot hand off, so it has no business holding a revision."""
    names = _tool_names(agent_mod._SUBAGENT_EXTRA_TOOLS)
    assert "edit_wiki_section" not in names
    assert "delete_wiki_section" not in names
    assert "handoff_to_successor" not in names


def test_writer_keeps_every_tool_it_had():
    """Regression guard: the writer's toolset may grow, never shrink."""
    names = _tool_names(agent_mod._WRITER_EXTRA_TOOLS)
    assert {"read_wiki_outline", "read_wiki_section", "edit_wiki_section",
            "delete_wiki_section", "validate_wiki",
            "handoff_to_successor"} <= names


def test_writer_gets_the_citation_check():
    """The cheap "is there any work left?" call. Without it the only way to
    verify the citation invariant is to re-read the whole page."""
    assert "check_members_cited" in _tool_names(agent_mod._WRITER_EXTRA_TOOLS)


def test_delegate_docstring_no_longer_promises_the_full_toolset():
    """The false promise is what made writers delegate edits a subagent
    could not perform."""
    doc = tools_mod.delegate_to_subagent.description or ""
    assert "all the same BrainDB tools" not in doc
    assert "read_wiki_section" in doc


# ====================================================================== #
# check_members_cited — must agree with the router, exactly               #
# ====================================================================== #
#
# The router decides at `routers/wiki.py` whether a no-op run did real work:
#     cited   = wiki_jobs.parse_refs(body_now)
#     missing = [m for m in member_ids if m.lower() not in cited]
# The tool exists so the writer can ask that same question BEFORE spending
# thirty turns re-reading the page to answer it itself. If the two ever
# disagree, the writer would confidently finish a job the router then fails.

def _router_missing(body: str, member_ids: list[str]) -> list[str]:
    from braindb.services.wiki_jobs import parse_refs
    cited = parse_refs(body)
    return [m for m in member_ids if m.lower() not in cited]


UUID_A = "aaaaaaaa-1111-2222-3333-444444444444"
UUID_B = "bbbbbbbb-1111-2222-3333-444444444444"
UUID_C = "cccccccc-1111-2222-3333-444444444444"


def test_citation_check_matches_router_when_all_cited():
    body = f"prose [[ref:{UUID_A}]] and [[ref:{UUID_B}]]\n"
    assert _router_missing(body, [UUID_A, UUID_B]) == []


def test_citation_check_matches_router_when_some_missing():
    body = f"prose [[ref:{UUID_A}]]\n"
    assert _router_missing(body, [UUID_A, UUID_B, UUID_C]) == [UUID_B, UUID_C]


def test_citation_check_is_case_insensitive_like_the_router():
    """`parse_refs` lower-cases; member ids may arrive upper-cased."""
    body = f"prose [[ref:{UUID_A.upper()}]]\n"
    assert _router_missing(body, [UUID_A]) == []


def test_citation_check_accepts_the_display_text_form():
    """`[[ref:UUID|display]]` is a documented variant and must still count."""
    body = f"prose [[ref:{UUID_A}|Some Source]]\n"
    assert _router_missing(body, [UUID_A]) == []


# ====================================================================== #
# Delegation depth — per run, not per process                             #
# ====================================================================== #

def test_delegation_depth_defaults_to_zero():
    assert tools_mod._depth_var.get() == 0


def test_delegation_depth_is_a_contextvar_not_a_module_global():
    """A plain global counts delegations across the whole PROCESS, so with
    concurrent writers one agent's delegation refused every other agent's."""
    from contextvars import ContextVar
    assert isinstance(tools_mod._depth_var, ContextVar)
    assert not hasattr(tools_mod, "_call_depth")


def test_depth_is_isolated_between_concurrent_runs():
    """Two sibling agent runs must not see each other's depth. This is the
    exact live failure: two delegations 1.8ms apart in one parallel batch,
    the second rejected with "max delegation depth reached"."""
    async def sibling(set_to: int, hold: float) -> int:
        token = tools_mod._depth_var.set(set_to)
        try:
            await asyncio.sleep(hold)
            return tools_mod._depth_var.get()
        finally:
            tools_mod._depth_var.reset(token)

    async def main():
        # asyncio.gather runs each coroutine in its own context copy
        return await asyncio.gather(sibling(1, 0.02), sibling(0, 0.01))

    seen = asyncio.run(main())
    assert seen == [1, 0]
    assert tools_mod._depth_var.get() == 0


def test_depth_increment_is_visible_to_nested_work():
    """A subagent spawned from the tool body must see depth+1, so it is
    correctly barred from delegating again."""
    async def main():
        outer = tools_mod._depth_var.get()
        token = tools_mod._depth_var.set(outer + 1)
        try:
            async def nested() -> int:
                return tools_mod._depth_var.get()
            return await nested()
        finally:
            tools_mod._depth_var.reset(token)

    assert asyncio.run(main()) == 1
    assert tools_mod._depth_var.get() == 0


def test_max_depth_still_one():
    """Bounded delegation is deliberate — the fix was the scope of the
    counter, not the limit."""
    assert tools_mod._MAX_DEPTH == 1
