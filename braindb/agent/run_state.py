"""
Per-run side-channel for the agent's final structured payload.

Why this exists: `Agent(output_type=<PydanticModel>)` makes the SDK pass
`response_format: json_schema` on EVERY LLM call (not just the final
one), which steers weaker models to satisfy the schema on turn 1 and
skip tools entirely. We therefore build agents WITHOUT `output_type` so
intermediate turns are free — but then `StopAtTools` would `str()`-coerce
the stop-tool's return into `result.final_output`, and we'd lose the
typed instance.

This module is the bridge: each `submit_*` tool body parks the
SDK-validated payload via `record_submit(payload)`; `run_typed` reads it
back via `slot.value` after `Runner.run` returns.

## Why a mutable slot, not just `ContextVar[Any]`

ContextVar values are inherited by reference into child asyncio Tasks,
but `.set()` inside a child Task does NOT propagate up to the parent.
The openai-agents SDK runs tool bodies (including parallel-tool batches)
inside such child Tasks, so a naive `last_submit.set(payload)` in the
tool body is invisible to the surrounding `run_typed`. Putting a mutable
container in the ContextVar instead — and mutating its `.value` from the
tool — works across that boundary because every Task sees the same
object reference. The standard `set(slot) + reset(token)` lifecycle in
`run_typed` keeps nested runs (parent → `delegate_to_subagent` →
subagent) isolated: each level uses its own `_Slot`.
"""
from contextvars import ContextVar
from typing import Any


class _Slot:
    """One-shot holder for the validated payload of a single agent run."""
    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value: Any = None


# Default None — `run_typed` always installs its own slot before awaiting
# `Runner.run`. A `None` here at submit time means "called outside a
# run_typed scope" and is just silently dropped (no slot to write to).
_slot_var: ContextVar["_Slot | None"] = ContextVar(
    "braindb_last_submit_slot", default=None,
)


def install_slot() -> tuple[_Slot, object]:
    """Used by `run_typed` to start a run. Returns `(slot, token)`; pass
    `token` to `release_slot` in a `finally:` to restore the previous
    context (so nested runs are isolated)."""
    slot = _Slot()
    token = _slot_var.set(slot)
    return slot, token


def release_slot(token: object) -> None:
    """Restore the previous slot (call in `finally:` after `install_slot`)."""
    _slot_var.reset(token)  # type: ignore[arg-type]


# Run tag — a short id `run_typed` sets for the duration of ONE agent run,
# read DOWNWARD by tool logging (`_verbose` in tools.py) inside the SDK's
# child Tasks, so concurrent runs' TOOL lines are separable in `docker
# logs`. Same read-down pattern as the delegation-depth ContextVar in
# tools.py: no mutable slot needed, nothing ever writes it back up. A
# nested run (delegate_to_subagent -> run_typed) sets its own tag; the
# delegate call itself is logged under the parent's tag, which is the
# correlation point. (Audits of concurrent writer+maintainer+subagent
# logs previously had to attribute calls by argument fingerprint.)
_run_tag_var: ContextVar[str] = ContextVar("braindb_run_tag", default="")


def set_run_tag(tag: str) -> object:
    """Set the current run's log tag; returns the reset token."""
    return _run_tag_var.set(tag)


def reset_run_tag(token: object) -> None:
    _run_tag_var.reset(token)  # type: ignore[arg-type]


def get_run_tag() -> str:
    """The current run's log tag, or "" outside any run."""
    return _run_tag_var.get()


def record_submit(payload: Any) -> None:
    """Called from inside every `submit_*` tool body. The SDK has already
    validated `payload` against the tool's Pydantic argument schema, so
    the value parked here is the typed final answer by construction.

    Mutates the slot in place (does NOT call `ContextVar.set(...)`) — see
    module docstring for why."""
    slot = _slot_var.get()
    if slot is not None:
        slot.value = payload


# ====================================================================== #
# Handoff side-channel (writer-only)                                      #
# ====================================================================== #
#
# Parallels the final-answer slot above. The writer's `handoff_to_successor`
# tool parks its brief here; the run wrapper in `routers/wiki.py` reads it
# after `run_typed` returns and decides whether to spawn a successor. Lives
# in run_state.py (not in a writer-specific module) so the slot lifecycle
# uses the same ContextVar discipline — install in the wrapper, mutate in
# the tool body, isolated across nested runs.


class _HandoffSlot:
    """One-shot holder for the writer's handoff brief. Distinct from
    `_Slot` because the wrapper inspects two independent fields
    (progress + remaining) rather than a single typed payload."""
    __slots__ = ("captured", "progress_summary", "remaining_work")

    def __init__(self) -> None:
        self.captured: bool = False
        self.progress_summary: str = ""
        self.remaining_work: str = ""


_handoff_slot_var: ContextVar["_HandoffSlot | None"] = ContextVar(
    "braindb_handoff_slot", default=None,
)


def install_handoff_slot() -> tuple[_HandoffSlot, object]:
    """Used by the writer's run wrapper to start a run that may end via
    handoff. Returns `(slot, token)`; pass `token` to `release_handoff_slot`
    in a `finally:`."""
    slot = _HandoffSlot()
    token = _handoff_slot_var.set(slot)
    return slot, token


def release_handoff_slot(token: object) -> None:
    _handoff_slot_var.reset(token)  # type: ignore[arg-type]


def record_handoff(progress_summary: str, remaining_work: str) -> None:
    """Called from the `handoff_to_successor` tool body. Mutates the slot
    in place (same reason as `record_submit`)."""
    slot = _handoff_slot_var.get()
    if slot is not None:
        slot.captured = True
        slot.progress_summary = progress_summary
        slot.remaining_work = remaining_work
