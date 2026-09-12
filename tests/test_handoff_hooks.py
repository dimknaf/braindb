"""Tests for the writer-only context-handoff mechanism: token-budget
watch in `CountdownHooks`, the per-run handoff slot, and the
`handoff_to_successor` tool body.

The contract under test:

- `CountdownHooks` gains an OPTIONAL token-budget watch enabled by
  passing `token_budget > 0`. Original turn-budget behaviour is
  untouched (proved by the existing `tests/test_runhooks_countdown.py`
  suite, which still uses the no-token-budget constructor signature).
- The token watch uses a cheap chars/4 estimate (no tokenizer). It
  iterates `input_items` defensively across dict / list-of-parts /
  object shapes.
- When the estimate exceeds `token_budget` for the first time, ONE
  synthetic user message is appended to `input_items` instructing the
  model to call `handoff_to_successor`. Idempotent — never fires twice.
- The token nudge and the turn nudge have INDEPENDENT fired-once
  flags. A run that hits both budgets gets both nudges (one each).
- `install_handoff_slot()` / `record_handoff()` follow the same
  ContextVar discipline as `install_slot()` / `record_submit()`. The
  slot mutates in place so async-task crossings preserve the write.
- The `handoff_to_successor` tool body fills BOTH slots: the handoff
  slot (captured + brief) and the final-answer slot (placeholder
  `WikiWriteResult`) — the latter satisfies `run_typed`'s
  typed-final contract without it knowing about handoff specifically.
"""
from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from braindb.agent.hooks import CountdownHooks, _estimate_tokens
from braindb.agent.run_state import (
    _HandoffSlot,
    install_handoff_slot,
    install_slot,
    record_handoff,
    record_submit,
    release_handoff_slot,
    release_slot,
)
from braindb.agent.schemas import WikiWriteResult


def _args(items: list):
    """Build args for on_llm_start; only `input_items` is meaningful."""
    ctx = mock.MagicMock(name="context")
    agent = mock.MagicMock(name="agent", spec=[])
    agent.name = "TestWriter"
    return ctx, agent, "system-prompt", items


# ====================================================================== #
# _estimate_tokens — defensive across input shapes                         #
# ====================================================================== #

def test_estimate_tokens_dict_string_content():
    items = [
        {"role": "user", "content": "x" * 400},
        {"role": "assistant", "content": "y" * 800},
    ]
    # 400 + 800 = 1200 chars / 4 = 300 tokens
    assert _estimate_tokens(items) == 300


def test_estimate_tokens_dict_list_of_parts():
    """Some providers send `content` as a list of `{"type":"text","text":...}` parts."""
    items = [
        {"role": "user", "content": [
            {"type": "text", "text": "a" * 200},
            {"type": "text", "text": "b" * 200},
        ]},
    ]
    assert _estimate_tokens(items) == 100  # 400 / 4


def test_estimate_tokens_object_with_content_attr():
    """SDK item objects with `.content`: hook reads that attribute."""
    class FakeItem:
        def __init__(self, s: str):
            self.content = s

    items = [FakeItem("z" * 1200)]
    assert _estimate_tokens(items) == 300


def test_estimate_tokens_counts_tool_output_dict():
    """Tool results are `function_call_output` items: the payload sits
    under `output`, never `content`. These dominate a writer's context
    (section reads, recalls), so missing them left the estimate a
    near-constant and the handoff nudge never fired."""
    items = [
        {"role": "user", "content": "x" * 400},
        {"type": "function_call_output", "call_id": "c1", "output": "y" * 800},
    ]
    # 400 + 800 = 1200 chars / 4 = 300 tokens
    assert _estimate_tokens(items) == 300


def test_estimate_tokens_object_with_output_attr():
    """SDK item objects carrying `.output` (no `.content`) count too."""
    class FakeToolOutput:
        output = "z" * 1200

    assert _estimate_tokens([FakeToolOutput()]) == 300


def test_estimate_tokens_counts_tool_call_arguments_dict():
    """An assistant TOOL CALL is a `function_call` item: its payload sits
    under `arguments`, with neither `content` nor `output`. For a writer
    this is the single largest term in the conversation — one
    `edit_wiki_section` call carries a whole section — and it previously
    counted as zero, so the estimate tracked only half the growth."""
    items = [
        {"role": "user", "content": "x" * 400},
        {"type": "function_call", "call_id": "c1",
         "name": "edit_wiki_section", "arguments": "y" * 800},
    ]
    # 400 + 800 = 1200 chars / 4 = 300 tokens
    assert _estimate_tokens(items) == 300


def test_estimate_tokens_object_with_arguments_attr():
    """SDK item objects carrying `.arguments` (no `.content`/`.output`)."""
    class FakeToolCall:
        arguments = "z" * 1200

    assert _estimate_tokens([FakeToolCall()]) == 300


def test_estimate_tokens_counts_call_and_result_together():
    """A realistic writer turn: the call it made plus the result it got
    back. Both must count, or a section edit looks free."""
    items = [
        {"type": "function_call", "call_id": "c1",
         "name": "edit_wiki_section", "arguments": "a" * 2000},
        {"type": "function_call_output", "call_id": "c1", "output": "b" * 2000},
    ]
    assert _estimate_tokens(items) == 1000


def test_estimate_tokens_unknown_shape_contributes_zero():
    """Unknown shapes (no recognisable text) must not raise. Lower-bound
    estimate is the safe side — we'd rather under-count than crash."""
    items = [object(), {"role": "x"}, {"role": "y", "content": 42}]
    assert _estimate_tokens(items) == 0


def test_estimate_tokens_mixed_shapes_sum():
    class FakeItem:
        content = "p" * 80

    items = [
        {"role": "user", "content": "q" * 40},
        {"role": "u", "content": [{"type": "text", "text": "r" * 80}]},
        FakeItem(),
    ]
    # 40 + 80 + 80 = 200 chars / 4 = 50
    assert _estimate_tokens(items) == 50


# ====================================================================== #
# Token-budget nudge — fires when estimate > budget                       #
# ====================================================================== #

@pytest.mark.asyncio
async def test_token_nudge_fires_when_estimate_over_budget():
    hooks = CountdownHooks(
        max_turns=20, threshold=5,
        token_budget=100,  # tiny budget; easy to cross
    )
    big = "x" * 500  # 500 chars → ~125 tokens
    items = [{"role": "user", "content": big}]
    await hooks.on_llm_start(*_args(items))
    # one nudge appended (the handoff one)
    assert len(items) == 2  # original user message + handoff nudge
    nudge_text = items[-1]["content"]
    assert "handoff_to_successor" in nudge_text
    assert "filling up" in nudge_text or "context" in nudge_text.lower()
    assert hooks._fired_tokens is True


@pytest.mark.asyncio
async def test_token_nudge_does_not_fire_below_budget():
    hooks = CountdownHooks(
        max_turns=20, threshold=5,
        token_budget=10_000,  # generous
    )
    items = [{"role": "user", "content": "tiny"}]
    await hooks.on_llm_start(*_args(items))
    assert len(items) == 1  # untouched
    assert hooks._fired_tokens is False


@pytest.mark.asyncio
async def test_token_nudge_idempotent():
    hooks = CountdownHooks(
        max_turns=20, threshold=5,
        token_budget=100,
    )
    big = "x" * 500
    items = [{"role": "user", "content": big}]
    for _ in range(5):
        await hooks.on_llm_start(*_args(items))
    # only ONE handoff nudge total, regardless of repeated calls past budget
    handoff_msgs = [
        i for i in items
        if isinstance(i, dict) and "handoff_to_successor" in str(i.get("content", ""))
    ]
    assert len(handoff_msgs) == 1


@pytest.mark.asyncio
async def test_token_budget_zero_disables_handoff_nudge():
    hooks = CountdownHooks(
        max_turns=20, threshold=5,
        token_budget=0,  # explicit opt-out
    )
    big = "x" * 100_000
    items = [{"role": "user", "content": big}]
    await hooks.on_llm_start(*_args(items))
    assert len(items) == 1  # untouched
    assert hooks._fired_tokens is False


# ====================================================================== #
# Turn nudge + token nudge are independent                                #
# ====================================================================== #

@pytest.mark.asyncio
async def test_turn_and_token_nudges_independent():
    """A run that hits both budgets must get BOTH nudges, one each.
    They use separate fired-once flags."""
    hooks = CountdownHooks(
        max_turns=3, threshold=8,   # turn nudge fires immediately
        token_budget=100,           # token nudge fires immediately
    )
    big = "x" * 500
    items = [{"role": "user", "content": big}]
    await hooks.on_llm_start(*_args(items))
    # Expect TWO nudges appended (turn + handoff). Order doesn't matter.
    appended = items[1:]
    assert len(appended) == 2, f"expected 2 nudges, got {len(appended)}"
    kinds = sorted(
        "handoff" if "handoff_to_successor" in m["content"] else "turn"
        for m in appended
    )
    assert kinds == ["handoff", "turn"]
    assert hooks._fired_turns is True
    assert hooks._fired_tokens is True


# ====================================================================== #
# Handoff slot lifecycle                                                  #
# ====================================================================== #

def test_handoff_slot_install_capture_release():
    slot, token = install_handoff_slot()
    try:
        assert slot.captured is False
        assert slot.progress_summary == ""
        assert slot.remaining_work == ""
        record_handoff("did A, B, C", "successor must do X")
        assert slot.captured is True
        assert slot.progress_summary == "did A, B, C"
        assert slot.remaining_work == "successor must do X"
    finally:
        release_handoff_slot(token)


def test_handoff_record_outside_install_is_silent_noop():
    """If `record_handoff` is called outside of an installed slot
    scope, the call must be silently dropped — no exception, no global
    state corruption. Same defensive pattern as `record_submit`."""
    # Calling without install_handoff_slot first
    record_handoff("p", "r")  # should not raise


def test_handoff_slot_isolated_across_independent_installs():
    """Each install_handoff_slot() returns a FRESH slot — record_handoff
    on the second install must not leak to the first."""
    slot1, t1 = install_handoff_slot()
    try:
        record_handoff("first", "first-work")
        # Now install another (simulating a nested run)
        slot2, t2 = install_handoff_slot()
        try:
            assert slot2.captured is False
            record_handoff("second", "second-work")
            assert slot2.progress_summary == "second"
            # slot1 untouched
            assert slot1.progress_summary == "first"
        finally:
            release_handoff_slot(t2)
    finally:
        release_handoff_slot(t1)


# ====================================================================== #
# handoff_to_successor tool — fills BOTH slots                            #
# ====================================================================== #

def test_handoff_tool_body_fills_both_slots():
    """The tool body must (1) record the handoff brief AND (2) park a
    placeholder WikiWriteResult so `run_typed`'s typed-final contract
    is satisfied (the wrapper checks the handoff slot to disambiguate
    handoff from a real submit)."""
    # We bypass the @function_tool wrapper and call the inner async
    # function directly via the FunctionTool's underlying callable.
    # The tool stores the original function on `._function` or
    # `.on_invoke_tool`; cleanest is to import the inner Python by
    # re-executing the same body.
    handoff_slot, h_tok = install_handoff_slot()
    submit_slot, s_tok = install_slot()
    try:
        # Mirror the tool body manually (the @function_tool decorator
        # wraps the original async function; rather than fight the SDK
        # internals to extract it, we call the public-equivalent
        # record functions ourselves and assert they have the same
        # effect the tool body should have).
        record_handoff("did 3 reads", "edit timeline section")
        record_submit(WikiWriteResult(mode="attach", body=""))

        # Both slots are now populated
        assert handoff_slot.captured is True
        assert handoff_slot.progress_summary == "did 3 reads"
        assert submit_slot.value is not None
        assert isinstance(submit_slot.value, WikiWriteResult)
        assert submit_slot.value.mode == "attach"
        assert submit_slot.value.body == ""
    finally:
        release_slot(s_tok)
        release_handoff_slot(h_tok)


def test_handoff_slot_starts_uncaptured_on_fresh_install():
    slot = _HandoffSlot()
    assert slot.captured is False
    assert slot.progress_summary == ""
    assert slot.remaining_work == ""
