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
import threading
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


# ====================================================================== #
# Usage ledger — what one request cost, in tokens and tool calls          #
# ====================================================================== #
#
# Nothing recorded token usage before this: cost questions ("how many
# uncached tokens and output tokens did that answer take?") could only be
# guessed. The ledger is installed ONCE per top-level request (the
# /agent/query router, or `run_typed` for wiki runs); nested runs
# (subagents, the final_answer retry, the BadRequest retry) find the
# installed ledger and add into it, so a request's cost includes every run
# it spawned. Same discipline as the slots above: a mutable object in a
# ContextVar, mutated in place. It carries a lock because, unlike a submit
# slot, it is written concurrently — parallel tool calls run as separate
# Tasks, and work offloaded with `asyncio.to_thread` runs on other threads.
#
# `weighted` = uncached input + 4 x output: cached input costs ~nothing on a
# self-hosted server, and output tokens dominate both time and cost.


class _Usage:
    """Token and tool counters for one top-level request."""
    __slots__ = ("lock", "agent", "jev", "tools", "tags")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.agent = {"requests": 0, "input": 0, "cached": 0, "output": 0, "reasoning": 0}
        self.jev = {"calls": 0, "requests": 0, "input": 0, "cached": 0, "output": 0,
                    "failopen": 0, "unjudged": 0}
        self.tools: dict[str, int] = {}
        self.tags: list[str] = []

    def add(self, kind: str, **counts: int) -> None:
        with self.lock:
            bucket = getattr(self, kind)
            for k, v in counts.items():
                bucket[k] = bucket.get(k, 0) + int(v or 0)

    def note_tool(self, name: str) -> None:
        with self.lock:
            self.tools[name] = self.tools.get(name, 0) + 1

    def add_tag(self, tag: str) -> None:
        """Record a run tag, so log lines of every run this request spawned
        (subagents included) can be found from the request's ledger."""
        with self.lock:
            self.tags.append(tag)

    def as_dict(self) -> dict:
        with self.lock:
            return {"agent": dict(self.agent), "jev": dict(self.jev),
                    "tools": dict(self.tools), "tags": list(self.tags)}

    def line(self) -> str:
        """One log line: the agent's LLM usage, JEV's, and tool calls."""
        d = self.as_dict()
        a, j = d["agent"], d["jev"]
        w_agent = a["input"] - a["cached"] + 4 * a["output"]
        w_jev = j["input"] - j["cached"] + 4 * j["output"]
        tools = ",".join(f"{k}={v}" for k, v in sorted(d["tools"].items())) or "-"
        return (f"agent req={a['requests']} in={a['input']} cached={a['cached']} "
                f"out={a['output']} reasoning={a['reasoning']} weighted={w_agent} | "
                f"jev calls={j['calls']} req={j['requests']} in={j['input']} "
                f"cached={j['cached']} out={j['output']} failopen={j['failopen']} "
                f"weighted={w_jev} | tools {tools}")


_usage_var: ContextVar["_Usage | None"] = ContextVar("braindb_usage", default=None)


def install_usage() -> tuple[_Usage, object | None]:
    """Start counting for a top-level request. Returns `(ledger, token)`;
    when a ledger is already installed (a nested run), returns THAT ledger
    and `None`, so the outermost caller alone reports and releases it."""
    existing = _usage_var.get()
    if existing is not None:
        return existing, None
    ledger = _Usage()
    return ledger, _usage_var.set(ledger)


def release_usage(token: object | None) -> None:
    if token is not None:
        _usage_var.reset(token)  # type: ignore[arg-type]


def add_usage(kind: str, **counts: int) -> None:
    """Add counts to the current request's ledger; a no-op outside one."""
    ledger = _usage_var.get()
    if ledger is not None:
        ledger.add(kind, **counts)


def note_tool(name: str) -> None:
    ledger = _usage_var.get()
    if ledger is not None:
        ledger.note_tool(name)
