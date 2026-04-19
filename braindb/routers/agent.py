"""
Agent endpoint — POST /api/v1/agent/query

External callers (Claude Code, other tools) send a natural language query;
the BrainDB agent (LiteLLM + NVIDIA NIM) handles recall/save/relate via
its internal tools and returns a summary.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from braindb.agent.agent import run_agent_query
from braindb.db import get_conn
from braindb.services.activity_log import log_activity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    max_turns: int | None = Field(default=None, ge=1, le=60)


@router.post("/query")
async def agent_query(body: AgentQueryRequest):
    """Run a natural-language query through the BrainDB agent.

    When AGENT_VERBOSE=true is set in the server environment, every tool call
    is logged to stdout and visible via `docker logs braindb_api`.
    """
    try:
        result = await run_agent_query(body.query, max_turns=body.max_turns)
        with get_conn() as conn:
            log_activity(conn, "agent_query", details={
                "query": body.query[:500],
                "max_turns": result.get("max_turns"),
            })
        return result
    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(500, f"Agent failed: {e}")
