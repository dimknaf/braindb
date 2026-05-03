"""
Agent endpoint — POST /api/v1/agent/query

External callers (Claude Code, other tools) send a natural language query;
the BrainDB agent (LiteLLM + NVIDIA NIM) handles recall/save/relate via
its internal tools and returns a summary.
"""
import logging
from typing import Any

from agents.exceptions import MaxTurnsExceeded
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from braindb.agent.agent import run_agent_query
from braindb.agent.fast_path import try_fast_path
from braindb.config import settings
from braindb.services.activity_log import log_activity_in_new_transaction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    max_turns: int | None = Field(default=None, ge=1, le=60)


def _log_agent_query(query: str, details: dict[str, Any]) -> None:
    payload = {"query": query[:500], **details}
    log_activity_in_new_transaction("agent_query", details=payload)


@router.post("/query")
async def agent_query(body: AgentQueryRequest):
    """Run a natural-language query through the BrainDB agent.

    When AGENT_VERBOSE=true is set in the server environment, every tool call
    is logged to stdout and visible via `docker logs braindb_api`.
    """
    turns = body.max_turns or settings.agent_max_turns
    fast_path_result = try_fast_path(body.query)
    if fast_path_result is not None:
        _log_agent_query(body.query, {
            "max_turns": 0,
            "status": fast_path_result.get("status"),
        })
        return fast_path_result

    try:
        result = await run_agent_query(body.query, max_turns=body.max_turns)
        result.setdefault("status", "ok")
        _log_agent_query(body.query, {
            "max_turns": result.get("max_turns"),
            "status": result.get("status"),
        })
        return result
    except MaxTurnsExceeded as e:
        logger.warning("Agent query exceeded max_turns=%s: %s", turns, e)
        result = {
            "answer": f"Agent exceeded max_turns={turns} before calling submit_result.",
            "max_turns": turns,
            "turns_used": turns,
            "status": "max_turns_exceeded",
        }
        _log_agent_query(body.query, {
            "max_turns": turns,
            "turns_used": turns,
            "status": "max_turns_exceeded",
        })
        return result
    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(500, f"Agent failed: {e}")
