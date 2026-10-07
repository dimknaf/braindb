"""
In-loop JEV: the agent's own model marks which items of a recall are useful
for the request's need, inside the agent's own conversation.

Filter mode (services/jev.py `judge`) asks a separate judge about every
recalled item, with a prompt of its own: the judge reads every item once and
the agent reads the kept ones again, and neither read is cached for the
other. Here the question is asked where the agent already is. The request is
the agent's next turn up to the recall's output — system prompt, tools,
conversation and recall output, built from the conversation the hooks record
(run_state.py) with the SDK's own converter, exactly as the agent's next call
will send them — so the server can reuse the agent's cached prefix for it,
and the agent's next turn the request's. One short constrained answer then
judges every labelled item.

The agent sees the result as one line that code writes from the
probabilities ("likely useful for this request: [A], [C]"); generated text
never enters the conversation. Retrieval is untouched: every item is still
shown, and an item an earlier turn's recall showed is named on one line
instead of shown again.

Only the first call of a turn is judged: the next turn shows that call's
output first, so its prompt is complete up to there when the recall returns.
A later call's is not (the outputs before it are still running), so it is
skipped and counted. A recall whose sheet is skipped or fails goes out
exactly as it would without JEV.
"""
from __future__ import annotations

import asyncio
import logging
import string
from typing import Any, Callable

from braindb.agent.run_state import add_usage, get_need
from braindb.config import settings
from braindb.services import jev

logger = logging.getLogger(__name__)

# Item labels, one token each on the measured model. A recall's items past
# the 52nd go unlabelled and unjudged.
LABELS = string.ascii_uppercase + string.ascii_lowercase

# P(useful) from which an item is named likely useful. Chosen on dev news
# needs: at 0.01 the sheet named 98% of the essential items while naming 30%
# of all items (held-out needs: 94% and 25%).
USEFUL = 0.01


def wanted() -> bool:
    """Whether in-loop JEV serves the current request, so `run_typed`
    records the run's conversation. A sheet needs a self-hosted server."""
    return (settings.jev_enabled and bool(get_need())
            and (settings.jev_mode == "render"
                 or (settings.jev_mode == "sheet" and settings.self_hosted)))


async def annotate(ctx: Any, loop: Any, header: str, entries: list[tuple[str, str, str]],
                   rules: list[str], plain: str, fit: Callable[[str], str]) -> str:
    """A recall's output in a run that records its loop. `entries` are the
    items as (entity id, entity type, rendered entry), `plain` the output
    without JEV, and `fit` caps the output's length."""
    fresh, before = [], []
    for entity_id, entity_type, entry in entries:
        if loop.shown.get(entity_id, loop.turn) < loop.turn:  # an earlier turn showed it
            before.append(f"{entity_id} ({entity_type})")
        else:
            fresh.append((entity_id, f"[{LABELS[len(fresh)]}] {entry}"
                          if len(fresh) < len(LABELS) else entry))
    lines = [header]
    if before:
        lines.append("Also found, already shown by an earlier recall above: " + ", ".join(before))
    text = fit("\n".join(lines + [entry for _, entry in fresh] + rules))
    # An entry the length cap cut is neither judged nor counted as shown.
    visible = [LABELS[k] for k, (_, entry) in enumerate(fresh[:len(LABELS)])
               if f"\n{entry}" in text]
    line = ""
    if settings.jev_mode == "sheet" and visible:
        line = await _verdict(ctx, loop, text, visible)
        if line is None:  # no judgement: the recall goes out exactly as without JEV
            text, line = plain, ""
            fresh = [(entity_id, entry) for entity_id, _, entry in entries]
    for entity_id, entry in fresh:
        if f"\n{entry}" in text:
            loop.shown.setdefault(entity_id, loop.turn)
    return text + line


async def _verdict(ctx: Any, loop: Any, text: str, labels: list[str]) -> str | None:
    """The sheet's line for the recall `text` whose visible items carry
    `labels`, or None when the sheet was skipped or failed."""
    calls = [it.get("call_id") for it in loop.response
             if isinstance(it, dict) and it.get("type") == "function_call"]
    if not calls or calls[0] != ctx.tool_call_id:
        add_usage("jev", skipped=1)
        return None
    try:
        messages, tools, options = await _request(ctx, loop, text)
        probs = await asyncio.to_thread(jev.sheet, messages, tools, labels, get_need(), options)
    except Exception as e:  # noqa: BLE001 - e.g. an SDK change: the recall goes out as without JEV
        logger.warning("JEV sheet failed: %r", e)
        add_usage("jev", calls=1, failopen=1)
        return None
    if probs is None:
        return None
    span = f"[{labels[0]}]-[{labels[-1]}]" if len(labels) > 1 else f"[{labels[0]}]"
    useful = [f"[{label}]" for label in labels if probs[label] >= USEFUL]
    if not useful:
        return f"\n\nRelevance check of {span}: none looks useful for this request."
    rest = "; the rest look unrelated" if len(useful) < len(labels) else ""
    return f"\n\nRelevance check of {span}: likely useful for this request: {', '.join(useful)}{rest}."


async def _request(ctx: Any, loop: Any, output: str) -> tuple[list[dict], list[dict], dict | None]:
    """The agent's next-turn messages up to this call's output, its tools,
    and its chat-template arguments, built the way the SDK's chat model
    builds its own request."""
    from agents.models.chatcmpl_converter import Converter
    from agents.util._json import _to_dump_compatible

    agent = loop.agent
    model = agent.model
    items = loop.items + loop.response + [
        {"type": "function_call_output", "call_id": ctx.tool_call_id, "output": output}]
    messages = Converter.items_to_messages(
        items, model=model.model, base_url=str(model._client.base_url),
        should_replay_reasoning_content=model.should_replay_reasoning_content,
        strict_feature_validation=model._strict_feature_validation)
    if loop.system:
        messages.insert(0, {"content": loop.system, "role": "system"})
    tools = [Converter.tool_to_openai(t) for t in await agent.get_all_tools(ctx)]
    template = (agent.model_settings.extra_body or {}).get("chat_template_kwargs")
    return (_to_dump_compatible(messages), _to_dump_compatible(tools),
            {"chat_template_kwargs": template} if template else None)
