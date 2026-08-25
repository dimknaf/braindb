"""Runtime nudge: tell the LLM to finalise when it's about to run out of turns.

WHY this exists
---------------
The strict typed-final contract (`final_answer` tool with a Pydantic argument
schema, no `output_type` on the Agent — see `braindb/agent/agent.py`) raises a
`RuntimeError` if the model ends a run without calling `final_answer`. Weak
or quantised models sometimes over-explore (chaining `recall_memory` /
`delegate_to_subagent` calls beyond what's necessary) and reach
`max_turns` without ever submitting. The strict path correctly catches this
as a failure, but we'd rather give the model a fighting chance: shortly
before `max_turns` is exhausted, inject a chat message reminding it to
finalise.

HOW the nudge gets into the conversation
-----------------------------------------
The openai-agents SDK's `RunHooks.on_llm_start` callback (see
`agents/lifecycle.py`) receives the mutable `input_items` list that's about
to be sent to the LLM. Appending one item to that list adds a synthetic
user message visible to the model on its NEXT turn. That's the same
mechanism the SDK uses internally for any added context. We exploit it
exactly once per run (idempotent), at the configured threshold.

Knobs (see `braindb/config.py`)
- `agent_countdown_threshold` (default 5): how many turns before
  `max_turns` we start nudging. Set to 0 to disable the nudge entirely.

Design constraints
- One nudge per run (no spam).
- Defensive: any internal error in the hook is caught and logged, never
  re-raised — a future SDK shape change must not bring down agent runs.
- Pure on-LLM-start counting — no SDK-private state inspection.
"""
from __future__ import annotations

import logging
from typing import Any

from agents.lifecycle import RunHooks

logger = logging.getLogger(__name__)


def _estimate_tokens(input_items: list) -> int:
    """Cheap (no-tokenizer) prompt-token estimate: sum the text-content
    character counts and divide by 4. Defensive across the shapes the
    SDK puts into `input_items`:
    - `{"role": str, "content": str}` (LiteLLM dict form)
    - `{"role": str, "content": [{"type":"text","text":str}, ...]}`
      (some providers send a list of parts)
    - `{"type": "function_call_output", "output": str}` — a tool result
      keeps its payload under `output`, never `content`. These dominate
      a writer's context (section reads, recalls), so reading only
      `content` left the estimate a near-constant and the handoff nudge
      never fired at any budget.
    - SDK item objects with a `.content` or `.output` attribute
    Unknown shapes contribute 0; the estimate is a lower bound, which
    is the safe side for "is context filling up" decisions (we'd rather
    fire the handoff nudge slightly late than slightly never)."""
    total_chars = 0
    for item in input_items:
        content: object
        if isinstance(item, dict):
            content = item.get("content") or item.get("output", "")
        else:
            content = getattr(item, "content", None) or getattr(item, "output", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    if isinstance(text, str):
                        total_chars += len(text)
                elif isinstance(part, str):
                    total_chars += len(part)
    return total_chars // 4


class CountdownHooks(RunHooks):
    """Mutates `input_items` to inject up to TWO independent nudges:

    1. Turn-budget nudge ("you have N turns left, finalise") — fires when
       the agent is close to exhausting `max_turns`. Original behaviour;
       see module docstring.

    2. Token-budget nudge ("context is filling up, call handoff_to_successor")
       — fires ONLY when `token_budget > 0` AND the cheap token estimate
       of `input_items` (sum-of-content-chars / 4) exceeds the budget.
       Writer-only: callers that don't set `token_budget` get the
       original turn-only behaviour. The two nudges have independent
       fired-once flags so one cannot suppress the other.

    Lifecycle (per run):
      - constructed once with knobs (turn-related + optional token-related).
      - `on_llm_start` fires before each LLM call.
        - increments `_turns`; if `_turns >= max_turns - threshold` AND
          `_fired_turns` is False, appends the turn nudge.
        - if `token_budget > 0` AND
          `estimated_tokens(input_items) > token_budget` AND
          `_fired_tokens` is False, appends the handoff nudge.
      - each nudge fires at most once per run.

    Disabled paths:
      - `threshold <= 0` disables the turn nudge (existing safety hatch).
      - `token_budget <= 0` disables the handoff nudge (default; non-writer
        callers don't pass this).
    """

    def __init__(
        self,
        max_turns: int,
        threshold: int,
        tool_name: str = "final_answer",
        *,
        token_budget: int = 0,
        handoff_tool_name: str = "handoff_to_successor",
    ) -> None:
        self.max_turns = max_turns
        self.threshold = max(0, int(threshold))
        self.tool_name = tool_name
        self.token_budget = max(0, int(token_budget))
        self.handoff_tool_name = handoff_tool_name
        self._turns: int = 0
        self._fired_turns: bool = False
        self._fired_tokens: bool = False

    # Backwards-compatibility: existing tests reference `._fired` on
    # instances built without token_budget. Map it to the turn-fired
    # flag so they keep observing the same semantic.
    @property
    def _fired(self) -> bool:  # noqa: D401
        return self._fired_turns

    @_fired.setter
    def _fired(self, v: bool) -> None:
        self._fired_turns = v

    # NOTE: `on_llm_start` is the canonical hook for injecting context
    # before the next LLM call (the SDK passes `input_items` mutably).
    # We don't override `on_tool_start` because we want to count
    # LLM-call turns, not tool calls — those can be multiple per turn.
    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list,
    ) -> None:
        try:
            self._turns += 1
            self._maybe_inject(input_items)
        except Exception as e:  # noqa: BLE001 — defensive: never kill the run
            logger.warning(
                "CountdownHooks.on_llm_start swallowed an internal error "
                "(turns=%d, fired=%s): %r", self._turns, self._fired, e,
            )

    def _maybe_inject(self, input_items: list) -> None:
        """Pure logic: decide whether to append a nudge now. Two
        independent checks (turn-budget + token-budget); each fires at
        most once per run. Separated from on_llm_start so tests can stub
        it to verify the wrapper's exception-swallowing behaviour."""
        # Turn-budget nudge (original Layer 3).
        if self.threshold > 0 and not self._fired_turns:
            remaining = self.max_turns - self._turns
            if remaining <= self.threshold:
                self._fired_turns = True
                nudge = self._format_nudge(remaining)
                input_items.append({"role": "user", "content": nudge})
                logger.info(
                    "CountdownHooks injected TURN nudge at turn %d/%d "
                    "(remaining=%d): %s",
                    self._turns, self.max_turns, remaining, nudge[:120],
                )

        # Token-budget nudge (handoff path).
        if self.token_budget > 0 and not self._fired_tokens:
            est = _estimate_tokens(input_items)
            if est > self.token_budget:
                self._fired_tokens = True
                handoff = self._format_handoff_nudge(est)
                input_items.append({"role": "user", "content": handoff})
                logger.info(
                    "CountdownHooks injected HANDOFF nudge (est_tokens=%d, "
                    "budget=%d): %s",
                    est, self.token_budget, handoff[:120],
                )

    def _format_handoff_nudge(self, est_tokens: int) -> str:
        """Text the model sees when token usage crosses the budget. Asks
        it to call the handoff tool with a structured brief; gives the
        agent an escape hatch (call final_answer directly) for small
        remaining work."""
        return (
            f"Your context is filling up (≈{est_tokens} estimated tokens; "
            f"budget {self.token_budget}). To avoid running out, call "
            f"`{self.handoff_tool_name}` now with a structured brief:\n"
            f"- progress_summary: tools you've called, key findings, and "
            f"any active revision tokens (the wiki you've been editing).\n"
            f"- remaining_work: the concrete next tool call(s) the "
            f"successor must make — name wikis, section names, revisions.\n"
            f"A fresh agent with the same prompt and tools will continue "
            f"from your brief. If you can still finish in 1-2 turns you "
            f"may instead call `{self.tool_name}` directly, but err on "
            f"the side of handoff when context is this tight."
        )

    def _format_nudge(self, remaining: int) -> str:
        """The text the model sees. Tone is chosen by `self.max_turns`:

        - SOFT (max_turns > 5): "start wrapping up, you have N left".
          Used when the budget is generous (the new default of 20 with
          threshold 8 fires the nudge at turn 12, with 8 turns still to
          spend). Deep-research models like Qwen do better when given
          a "begin concluding" signal rather than a hard stop — they
          can do one or two focused gap-filling calls before
          `final_answer` instead of slamming tools shut mid-thread.

        - HARD (max_turns ≤ 5): "call `final_answer` NOW". Used when
          the budget is tight — most notably the Layer 4 retry path
          (`max_turns=3`), where the retry is explicitly a "you forgot
          to finalise, please call the tool now" correction. The
          model gets the unambiguous instruction without ambiguity
          about wrapping up vs investigating further.

        Why pick the tone from `max_turns` rather than an explicit
        constructor flag: the retry call site already passes its own
        `max_turns=settings.agent_retry_max_turns` (3) and the main
        run passes the general `max_turns` (20). The two contexts
        differ exactly along the budget axis, so we get the right
        tone with no new constructor surface and no caller changes.
        """
        # Clamp to non-negative for readability; if remaining went past 0
        # we still want a coherent message even though the SDK would
        # raise MaxTurnsExceeded shortly.
        remaining = max(0, remaining)
        plural = "s" if remaining != 1 else ""
        if self.max_turns <= 5:
            return (
                f"You have {remaining} tool call{plural} left. "
                f"Call `{self.tool_name}` with your answer now. "
                f"Do not start new research."
            )
        return (
            f"Heads up: you have {remaining} tool call{plural} left "
            f"in this run. Start wrapping up — synthesise what you "
            f"have already gathered and prepare to call "
            f"`{self.tool_name}`. Focused gap-filling is fine; avoid "
            f"opening brand-new lines of investigation."
        )
