"""
Agent endpoint — POST /api/v1/agent/query

External callers (Claude Code, other tools) send a natural language query;
the BrainDB agent (OpenAI Agents SDK, provider per LLM_PROFILE) handles
recall/save/relate via its internal tools and returns a summary.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from braindb.agent.agent import run_agent_query
from braindb.agent.run_state import (
    install_usage,
    release_usage,
    reset_need,
    reset_reading,
    set_need,
    set_reading,
)
from braindb.db import get_conn
from braindb.services import hops, reader
from braindb.services.activity_log import log_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    # Bumped 10000 -> 40000 to accommodate the ingest watcher's per-chunk
    # extraction prompts at the larger CHUNK_WORDS=1200 size (chunk text
    # ~6 KB + boilerplate + cross-fact-relation instructions ~7.5 KB total,
    # well under the new cap). Sized so that even at 40K input chars
    # (~10K tokens) the LLM has plenty of headroom: Qwen 27B runs at
    # max_model_len=40960, so input + system prompt + tool defs +
    # ~35-turn tool-call iteration leaves ~30% of the window for output.
    query: str = Field(..., min_length=1, max_length=40000)
    max_turns: int | None = Field(default=None, ge=1, le=60)
    # Optional: what the caller is trying to find out, in a sentence. The
    # JEV reader reads long bodies for it and JEV hops judge recall's
    # neighbours for it (both on by default, on a server that passes their
    # probe); with JEV_ENABLED, the agent's recalls keep only the items the
    # model judges useful for it. Ignored otherwise.
    need: str | None = Field(default=None, max_length=2000)


@router.post("/query")
async def agent_query(body: AgentQueryRequest):
    """Run a natural-language query through the BrainDB agent.

    When AGENT_VERBOSE=true is set in the server environment, every tool call
    is logged to stdout and visible via `docker logs braindb_api`.
    """
    # One usage ledger per request: every run it spawns (subagents, retries)
    # adds into it. Recorded with the activity row — on failure too, or a
    # failing request would look free — and never in the response, whose
    # shape stays {"answer", "max_turns"}.
    usage, usage_token = install_usage()
    need_token = set_need((body.need or "").strip())
    # What the JEV reader reads long bodies for, and hops judge recall's
    # neighbours for: the need, else a short query (services/reader.py).
    # Empty while both are off.
    reading_token = set_reading(
        reader.reading_question(body.need, body.query) if reader.wanted() or hops.wanted() else "")
    try:
        result = await run_agent_query(body.query, max_turns=body.max_turns)
        with get_conn() as conn:
            log_activity(conn, "agent_query", details={
                "query": body.query[:500],
                "max_turns": result.get("max_turns"),
                "usage": usage.as_dict(),
            })
        return result
    except Exception as e:
        logger.exception("Agent query failed")
        try:
            with get_conn() as conn:
                log_activity(conn, "agent_query", details={
                    "query": body.query[:500],
                    "error": type(e).__name__,
                    "usage": usage.as_dict(),
                })
        except Exception:  # noqa: BLE001 — never mask the original failure
            logger.warning("could not record the failed agent query's usage")
        raise HTTPException(500, f"Agent failed: {e}")
    finally:
        logger.info("Usage (agent_query): %s", usage.line())
        release_usage(usage_token)
        reset_reading(reading_token)
        reset_need(need_token)
