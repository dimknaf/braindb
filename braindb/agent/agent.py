"""
BrainDB internal agent — builder and runner.

Mirrors the pattern in fa-automation/tasks/linkedin_research/agent.py:
- create_braindb_agent() wires model + tools + instructions
- run_agent_query() is the async Runner.run() wrapper
- Singleton pattern so the agent is built once and reused
"""
import logging
from pathlib import Path

from agents import Agent, ModelSettings, Runner, StopAtTools, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from braindb.agent.tools import (
    create_relation,
    delegate_to_subagent,
    delete_entity,
    delete_relation,
    generate_embeddings,
    get_entity,
    get_stats,
    ingest_file,
    list_entities,
    quick_search,
    recall_memory,
    save_fact,
    save_rule,
    save_source,
    save_thought,
    search_sql,
    submit_result,
    update_entity,
    view_entity_relations,
    view_log,
    view_tree,
)
from braindb.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system_prompt.md").read_text(encoding="utf-8")

_agent: Agent | None = None


def create_braindb_agent() -> Agent:
    """Build the BrainDB agent. Provider selected via settings.llm_profile."""
    model = LitellmModel(
        model=settings.resolved_agent_model,
        api_key=settings.resolved_api_key,
    )
    set_tracing_disabled(disabled=True)

    agent = Agent(
        name="BrainDB Memory Agent",
        instructions=SYSTEM_PROMPT,
        model=model,
        model_settings=ModelSettings(),
        tools=[
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
            submit_result,
        ],
        tool_use_behavior=StopAtTools(stop_at_tool_names=["submit_result"]),
    )
    logger.info("BrainDB agent created with model: %s", settings.resolved_agent_model)
    return agent


def get_agent() -> Agent:
    """Get the singleton agent instance — built on first call."""
    global _agent
    if _agent is None:
        _agent = create_braindb_agent()
    return _agent


async def run_agent_query(query: str, max_turns: int | None = None) -> dict:
    """Run a query through the agent loop. Returns the final answer + metadata.

    When `settings.agent_verbose` is True, every tool call is logged to stdout
    via the standard logger (visible in `docker logs braindb_api`).
    """
    agent = get_agent()
    turns = max_turns or settings.agent_max_turns
    logger.info("Running agent query: %s", query[:200])
    result = await Runner.run(
        starting_agent=agent,
        input=query,
        max_turns=turns,
    )
    return {
        "answer": str(result.final_output),
        "max_turns": turns,
    }
