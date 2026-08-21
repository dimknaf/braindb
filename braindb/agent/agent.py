"""
BrainDB internal agent — builder and runners.

Convention (absolute): every agent run finishes via the `final_answer`
trick, and that tool's argument is ALWAYS a typed Pydantic model. The LLM
never emits loose / free-form output we then scrape.

There is one agent per purpose, differing only by which typed
`submit_*` variant it carries (all named "final_answer" so prompts and
`StopAtTools(["final_answer"])` stay generic). The structured contract
lives on the **tool argument schema** (`@function_tool` + Pydantic),
which is what the user wanted: validated final answer, free middle
turns. We deliberately do NOT set `output_type` on the Agent — that flag
makes the SDK pass `response_format: json_schema` on every LLM call,
which steers weaker models to satisfy the schema on turn 1 and never
call any tool (the regression we are fixing).

How we still recover the typed payload: each `submit_*` tool body parks
its already-validated `payload` into `braindb.agent.run_state.last_submit`
(a ContextVar). `run_typed` reads it back after `Runner.run` returns.
asyncio's per-Task context isolation makes nested/parallel runs safe.
"""
import json
import logging
from pathlib import Path
from typing import TypeVar

from agents import Agent, ModelSettings, Runner, StopAtTools, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from litellm import BadRequestError, ContextWindowExceededError
from pydantic import BaseModel

from braindb.agent.hooks import CountdownHooks
from braindb.agent.run_state import install_slot, release_slot
from braindb.agent.schemas import (
    AgentAnswer,
    MaintainerDecision,
    SubagentResult,
    WikiWriteResult,
)
from braindb.agent.tools import (
    create_relation,
    delegate_to_subagent,
    delete_entity,
    delete_relation,
    delete_wiki_section,
    edit_wiki_section,
    generate_embeddings,
    get_entity,
    get_stats,
    handoff_to_successor,
    ingest_file,
    list_entities,
    quick_search,
    read_wiki_outline,
    read_wiki_section,
    recall_memory,
    save_fact,
    save_rule,
    save_source,
    save_thought,
    search_sql,
    submit_answer,
    submit_maintainer,
    submit_subagent,
    submit_wiki,
    update_entity,
    validate_wiki,
    view_entity_relations,
    view_log,
    view_tree,
)
from braindb.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_prompt.md").read_text(encoding="utf-8")

# Every tool except the final submit (that one is typed per purpose).
_BASE_TOOLS = [
    recall_memory,
    quick_search,
    save_fact,
    save_thought,
    save_source,
    save_rule,
    ingest_file,
    get_entity,
    list_entities,
    update_entity,
    delete_entity,
    create_relation,
    view_entity_relations,
    delete_relation,
    view_tree,
    search_sql,
    view_log,
    get_stats,
    generate_embeddings,
    delegate_to_subagent,
]

T = TypeVar("T")


def _expected_shape_hint(expected_cls: type[BaseModel]) -> str:
    """Render a literal JSON-call shape for the `final_answer` tool, derived
    from the Pydantic model the LLM must submit.

    Weak/quantised models routinely emit the wrong WRAPPER on retry: either
    they call `final_answer(<inner_dict>)` (missing the outer `payload`
    key) or `final_answer({"payload": <broken_dict>})` (missing required
    keys inside). The generic "call final_answer NOW" correction did not
    fix this on Gemma-31B (verified live: subagent retry kept emitting the
    same shape errors). Giving the model a literal JSON template that
    matches the @function_tool argument schema closes that gap — the LLM
    sees the exact key names and the outer wrapping it has to produce.

    Example output for `SubagentResult`:
        {"payload": {"result": "<your concise summary>"}}

    For `MaintainerDecision` (skip action):
        {"payload": {"action": "skip", "rationale": "<short justification>"}}

    Only REQUIRED fields are filled with placeholders; optional/nullable
    fields are omitted so the LLM doesn't fabricate values for them. The
    helper handles enums (uses the first allowed value as the placeholder)
    so the example is always actually-valid against the schema.
    """
    schema = expected_cls.model_json_schema()
    required = schema.get("required", [])
    props = schema.get("properties", {})

    def placeholder(field_name: str, field_schema: dict) -> str | int | list | dict:
        # Literal/Enum: use the first allowed value so the example validates.
        enum = field_schema.get("enum")
        if enum:
            return enum[0]
        t = field_schema.get("type")
        if t == "integer":
            return 1
        if t == "number":
            return 0.0
        if t == "boolean":
            return False
        if t == "array":
            return []
        if t == "object":
            return {}
        # default: string
        return f"<{field_name}>"

    example_payload = {
        name: placeholder(name, props.get(name, {})) for name in required
    }
    return json.dumps({"payload": example_payload})


def _model() -> LitellmModel:
    return LitellmModel(
        model=settings.resolved_agent_model,
        api_key=settings.resolved_api_key,
        base_url=settings.resolved_base_url,
    )


def _build(
    name: str,
    submit_tool,
    extra_tools: tuple = (),
    extra_stop_tools: tuple[str, ...] = (),
) -> Agent:
    """Build an agent. NOTE: no `output_type` — see module docstring. The
    structured contract lives on `submit_tool`'s argument schema, not on
    the agent.

    `extra_tools` lets a specific agent (currently only the writer) carry
    role-specific tools (the wiki section-edit tools + handoff) without
    polluting `_BASE_TOOLS` shared by all agents.

    `extra_stop_tools` adds extra stop-tool names beyond `final_answer`.
    The writer adds `handoff_to_successor` here so the run halts cleanly
    when handoff is called instead of continuing wastefully.
    """
    set_tracing_disabled(disabled=True)
    agent = Agent(
        name=name,
        instructions=SYSTEM_PROMPT,
        model=_model(),
        # `extra_args` is forwarded verbatim into the LiteLLM call, so
        # `timeout` lands as `kwargs["timeout"]` — which LiteLLM resolves
        # ahead of its 600s fallback. This is a TRANSPORT deadline only: it
        # never enters the request body and cannot steer the model, unlike
        # `output_type` / `tool_choice` (see the module docstring — those
        # stay unset deliberately). Without it a long wiki write is
        # abandoned client-side at 600s while the server is still working.
        model_settings=ModelSettings(
            extra_args={"timeout": settings.agent_request_timeout},
        ),
        tools=[*_BASE_TOOLS, *extra_tools, submit_tool],
        tool_use_behavior=StopAtTools(
            stop_at_tool_names=["final_answer", *extra_stop_tools],
        ),
    )
    logger.info(
        "Agent built: %s (model=%s) — free middle turns, typed final_answer",
        name, settings.resolved_agent_model,
    )
    return agent


_cache: dict[str, Agent] = {}


def _cached(
    key: str,
    name: str,
    submit_tool,
    extra_tools: tuple = (),
    extra_stop_tools: tuple[str, ...] = (),
) -> Agent:
    a = _cache.get(key)
    if a is None:
        a = _build(
            name, submit_tool,
            extra_tools=extra_tools,
            extra_stop_tools=extra_stop_tools,
        )
        _cache[key] = a
    return a


# Writer-only tools: section read/edit/delete + grammar validation +
# context-handoff. The writer rewrites whole wiki bodies today; section
# tools let it edit one at a time, and `handoff_to_successor` lets it
# bail to a fresh agent when context approaches the wall. See
# braindb/services/wiki_sections.py + plan
# `feat/wikis-and-maintainer-agent-read-write-tools`.
_WRITER_EXTRA_TOOLS = (
    read_wiki_outline,
    read_wiki_section,
    edit_wiki_section,
    delete_wiki_section,
    validate_wiki,
    handoff_to_successor,
)
_WRITER_EXTRA_STOP_TOOLS = ("handoff_to_successor",)


def get_agent() -> Agent:
    """Default agent: general recall/save (public /agent/query)."""
    return _cached("answer", "BrainDB Memory Agent", submit_answer)


def get_maintainer_agent() -> Agent:
    return _cached("maintainer", "BrainDB Wiki Maintainer", submit_maintainer)


def get_writer_agent() -> Agent:
    return _cached(
        "writer", "BrainDB Wiki Writer", submit_wiki,
        extra_tools=_WRITER_EXTRA_TOOLS,
        extra_stop_tools=_WRITER_EXTRA_STOP_TOOLS,
    )


def get_subagent() -> Agent:
    return _cached("subagent", "BrainDB Subagent", submit_subagent)


def create_braindb_agent() -> Agent:
    """Backward-compat alias — the default (general) agent."""
    return get_agent()


async def run_typed(
    query: str,
    agent: Agent,
    expected_cls: type[T],
    max_turns: int | None = None,
    *,
    token_budget: int = 0,
    _bad_request_retried: bool = False,
) -> T:
    """Run a typed agent and return the validated Pydantic instance it
    submitted. The instance is guaranteed-valid because the SDK validates
    the LLM's `final_answer` call args against `expected_cls` BEFORE the
    tool body runs (via `@function_tool`'s strict JSON schema).

    Raises `RuntimeError` if the run ends without `final_answer` firing
    (e.g. `max_turns` exhausted) — surfaces a real model failure instead
    of silently returning bad data. Routers handle this like any other
    agent error: log + release the job lease + 5xx.

    `token_budget` (writer-only, opt-in): when > 0, enables the handoff
    nudge in `CountdownHooks` — at the first LLM call where the cheap
    token estimate of the conversation exceeds this budget, one
    synthetic user message instructs the model to call
    `handoff_to_successor`. The successor-respawn loop lives in the
    caller (see `braindb/routers/wiki.py`).
    """
    turns = max_turns or settings.agent_max_turns
    slot, token = install_slot()
    # Layer-3 nudge: when the run is about to exhaust `max_turns`, the hook
    # appends a synthetic "you have N turns left, finalise via final_answer"
    # user message to the conversation. One nudge per run; disabled when
    # `agent_countdown_threshold == 0`. See braindb/agent/hooks.py.
    # When `token_budget > 0` (writer path) the same hook also watches
    # estimated prompt tokens and injects ONE handoff nudge at the first
    # call where the estimate crosses the budget. Independent fired-once
    # flag from the turn nudge.
    hooks = CountdownHooks(
        max_turns=turns,
        threshold=settings.agent_countdown_threshold,
        tool_name="final_answer",
        token_budget=token_budget,
        handoff_tool_name="handoff_to_successor",
    )
    try:
        logger.info("Running typed query (%s): %s", agent.name, query[:160])
        result = await Runner.run(
            starting_agent=agent, input=query, max_turns=turns, hooks=hooks,
        )
        payload = slot.value
        if isinstance(payload, expected_cls):
            return payload

        # The first attempt ended without `final_answer` firing. Most
        # commonly the model emitted plain prose (a "fast finisher" /
        # forgetter) — strict mode would raise here. But before giving
        # up, Layer 4 gives the model exactly one chance to fix it:
        # append a user-role correction message to the conversation it
        # already produced (`result.to_input_list()`) and re-invoke
        # `Runner.run` with a small budget. The correction is unambiguous
        # — "you ended without `final_answer`, call it now". No parsing
        # of the prose, no fallback that pretends success; we use the
        # SDK's own conversation mechanism to tell the model what it did
        # wrong, then either it complies on the retry (HTTP 200) or we
        # raise (still strict).
        if settings.agent_retry_on_missing_final:
            logger.info(
                "%s ended without final_answer; retrying once with correction",
                agent.name,
            )
            # Build a literal JSON-shape hint from `expected_cls` so the
            # LLM gets an unambiguous template — not just "call it now",
            # but "call it like THIS". Verified live: Gemma subagents
            # retry without this hint by emitting payload-as-string or
            # missing-required-key variants that fail the @function_tool
            # validator and trigger the same error in a loop.
            shape_hint = _expected_shape_hint(expected_cls)
            correction = {
                "role": "user",
                "content": (
                    "Your previous response ended WITHOUT a successful "
                    "`final_answer` call (or `final_answer` was called "
                    "with the wrong JSON shape and rejected by the tool "
                    "validator). The work you did is preserved, but the "
                    "run is INVALID until you finalise.\n\n"
                    "Call `final_answer` NOW. The tool expects EXACTLY "
                    "one argument named `payload`, whose value is a JSON "
                    "object with the required keys. The literal shape "
                    f"you MUST send is:\n\n  {shape_hint}\n\n"
                    "Replace each <placeholder> with your real value. "
                    "Do NOT omit the outer `payload` key. Do NOT wrap "
                    "the payload as a string. Issue ONLY the tool call, "
                    "no prose, no further research."
                ),
            }
            retry_input = result.to_input_list() + [correction]
            retry_hooks = CountdownHooks(
                max_turns=settings.agent_retry_max_turns,
                threshold=settings.agent_countdown_threshold,
                tool_name="final_answer",
            )
            await Runner.run(
                starting_agent=agent,
                input=retry_input,
                max_turns=settings.agent_retry_max_turns,
                hooks=retry_hooks,
            )
            payload = slot.value
            if isinstance(payload, expected_cls):
                logger.info(
                    "%s recovered via final_answer-retry (correction worked)",
                    agent.name,
                )
                return payload

            # Retry also failed: model truly refuses the typed-final
            # contract even when told explicitly what to do. That's a
            # genuine model-discipline failure — raise loudly.
            raise RuntimeError(
                f"{agent.name} did not call final_answer even after a "
                f"correction retry — model refuses the typed-final "
                f"contract. Last final_output: "
                f"{str(getattr(result, 'final_output', ''))[:200]}"
            )

        # Retry disabled (opt-out via settings): preserve the original
        # strict-raise behaviour.
        raise RuntimeError(
            f"{agent.name} did not call final_answer with a "
            f"{expected_cls.__name__} (got {type(payload).__name__}). "
            f"The run terminated without the typed final tool firing — "
            f"the model likely ended with plain prose."
        )
    except ContextWindowExceededError:
        # The conversation is already over the model's window. A retry
        # without input truncation would just hit the same wall, so we
        # re-raise and let the router fail the job cleanly. Real fix is
        # upstream: keep prompts/tool-results small enough that the
        # handoff threshold catches us first. See routers/wiki.py's
        # `_body_block_or_stub` for the prompt-side mitigation.
        raise
    except BadRequestError as e:
        # Quantised models (Qwen AWQ-INT4) occasionally emit malformed
        # JSON in tool-call args; the OpenAI client raises BadRequestError
        # before the tool body runs. One fresh attempt usually recovers.
        # Bounded to depth 1 via the `_bad_request_retried` flag —
        # recursion uses run_typed itself rather than duplicating the
        # setup. The current slot is released by `finally`; the recursive
        # call installs its own.
        if _bad_request_retried:
            raise
        logger.warning(
            "%s: BadRequestError on first attempt (%s); "
            "retrying once with a fresh run", agent.name, str(e)[:160],
        )
        return await run_typed(
            query, agent, expected_cls,
            max_turns=max_turns, token_budget=token_budget,
            _bad_request_retried=True,
        )
    finally:
        release_slot(token)


async def run_agent_query(query: str, max_turns: int | None = None) -> dict:
    """General recall/save path (public /agent/query, and the ingest watcher
    over HTTP). The model finishes via `final_answer(payload: AgentAnswer)`;
    the response shape stays `{"answer","max_turns"}` for backward
    compatibility."""
    turns = max_turns or settings.agent_max_turns
    payload: AgentAnswer = await run_typed(query, get_agent(), AgentAnswer, max_turns=turns)
    return {"answer": payload.answer, "max_turns": turns}
