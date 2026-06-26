"""
setup_agent.py — provision the BimpeAI agent for the agent-driven BrainDB memory
integration.

Creates (idempotently) a Workflow whose system prompt teaches the agent to use
BrainDB as its memory by emitting `[[MEMORY: ...]]` commands (the bridge fulfils
them). Binds an Agent to it. Removes any leftover Knowledge Base so the agent has
no stale static memory — it relies entirely on live BrainDB lookups.

No tunnel: this is a normal outbound SDK call. Run with this folder's venv:
    integrations/bimpeai/.venv/Scripts/python  integrations/bimpeai/setup_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from bimpeai import BimpeAI

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"

WORKFLOW_NAME = "BrainDB Memory Companion"
AGENT_NAME = "BrainDB Memory Companion"

# The whole behaviour lives here. The agent has ONE way to touch memory: a
# [[MEMORY: ...]] command. The bridge sees it, runs it through BrainDB's own
# agent, and posts back [[MEMORY_RESULT: ...]]. The agent then answers.
SYSTEM_PROMPT = """You are a warm, concise personal MEMORY COMPANION on WhatsApp and voice. You remember things for one person and recall them on demand. You have a reliable long-term memory. You NEVER say you "can't access", "don't have", or "can't retrieve" information - you always can, through your memory.

# YOUR MEMORY IS ASYNCHRONOUS - this is the single most important rule
You do NOT know stored information yourself. You look it up by writing a command on its own line:
[[MEMORY: <plain-English instruction>]]
The lookup TAKES TIME. You will NOT have the answer in the same message - the result comes back a little later as a separate message:
[[MEMORY_RESULT: <answer>]]
So, in the very message where you write [[MEMORY: ...]], you have received NO answer yet. Therefore you MUST NOT answer, confirm, deny, or say "nothing found" in that message. You may ONLY say you're checking and ask the user to bear with you. You give the answer ONLY in a later message, after a [[MEMORY_RESULT: ...]] has actually arrived.

# SAVE - the user shares something to remember
Briefly acknowledge, say you're saving it (it takes a moment), emit the command, then STOP and wait.
  User: "I drive a Mitsubishi Colt."
  You: "Lovely - let me save that. It takes a moment to file away, so bear with me. [[MEMORY: remember that the user drives a Mitsubishi Colt]]"
  ...later: [[MEMORY_RESULT: Saved.]]
  You: "Done - I've saved that you drive a Mitsubishi Colt."

# RECALL - the user asks something
Say you're checking (it takes a moment), emit the command, then STOP and wait. Do NOT guess and do NOT say you don't have it.
  User: "What car do I drive?"
  You: "Let me check my memory - it takes a moment, so bear with me and I'll be right back. [[MEMORY: what car does the user drive?]]"
  ...later: [[MEMORY_RESULT: The user drives a Mitsubishi Colt.]]
  You: "You drive a Mitsubishi Colt."

# WRONG - never do any of these
  "I'm sorry, I don't have that information."  <- WRONG: you never emitted a command / never got a result. Emit [[MEMORY: ...]] and wait.
  "It seems nothing is stored yet."            <- WRONG unless a [[MEMORY_RESULT]] actually told you so.
  "[[MEMORY: what car...]] You don't have a car saved."  <- WRONG: you cannot answer in the same message as the command.

# While you're still waiting (e.g. the user asks "got it yet?")
Just reassure: "Still checking - it can take a moment, bear with me." Do NOT conclude there's nothing. Keep waiting for [[MEMORY_RESULT]].

# When [[MEMORY_RESULT: X]] arrives
Tell the user X in natural, second-person words ("you ..."). Only if X itself says nothing was found may you say you don't have that saved yet, and offer to remember it.

# Style
Short, warm, human. Emit each command ONCE. Never show the user the bracketed tags or explain the mechanics."""


def find_by_name(page_iter, name: str):
    for item in page_iter:
        if (getattr(item, "name", "") or "").strip().lower() == name.strip().lower():
            return item
    return None


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def upsert_env(updates: dict[str, str]) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.is_file() else []
    seen, out = set(), []
    for line in lines:
        hit = next((k for k in updates if line.startswith(f"{k}=")), None)
        if hit:
            out.append(f"{hit}={updates[hit]}"); seen.add(hit)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    env = load_env()
    api_key = env.get("BIMPEAI_API_KEY", "").strip()
    if not api_key:
        sys.exit("BIMPEAI_API_KEY missing in integrations/bimpeai/.env")
    client = BimpeAI(api_key=api_key)

    # Workflow -----------------------------------------------------------
    wf = find_by_name(client.workflows.list(scope="owned"), WORKFLOW_NAME)
    if wf is None:
        wf = client.workflows.create(
            name=WORKFLOW_NAME, system_prompt=SYSTEM_PROMPT,
            description="Agent-driven memory companion backed by a local BrainDB.",
            category="Productivity",
        )
        print(f"[+] created workflow  {wf.id}")
    else:
        client.workflows.update(wf.id, system_prompt=SYSTEM_PROMPT)
        print(f"[=] workflow {wf.id} (prompt updated)")

    # Agent --------------------------------------------------------------
    agent = find_by_name(client.agents.list(search=AGENT_NAME), AGENT_NAME)
    if agent is None:
        agent_id = client.agents.create(
            workflow_id=wf.id, name=AGENT_NAME,
            description="Remembers what you tell it and recalls it on demand, via BrainDB.",
            persona="friendly",
        ).id
        print(f"[+] created agent     {agent_id}")
    else:
        agent_id = agent.id
        if getattr(agent, "workflow_id", None) != wf.id:
            client.agents.update(agent_id, workflow_id=wf.id)
        print(f"[=] agent {agent_id}")

    # Remove any leftover KB (no static memory; the agent uses live BrainDB) ----
    try:
        for kb in client.agents.knowledge_bases.list(agent_id):
            client.agents.knowledge_bases.delete(agent_id, kb.id)
            print(f"[-] deleted stale KB  {kb.id}")
    except Exception as e:
        print(f"    (KB cleanup skipped: {e})")

    upsert_env({"BIMPEAI_AGENT_ID": agent_id})

    print("\n=== DONE ===")
    print(f"  workflow_id : {wf.id}")
    print(f"  agent_id    : {agent_id}  (written to .env)")
    try:
        tc = client.agents.get_test_code(agent_id)
        wa = getattr(getattr(tc, "channels", None), "whatsapp", None)
        if wa:
            print(f"  WhatsApp test: {getattr(wa,'url','')}  (send: {getattr(wa,'start_message','')})")
    except Exception as e:
        print(f"  (test code unavailable: {e})")
    print("\nSee it: BimpeAI Dashboard -> Workflows / Agents -> 'BrainDB Memory Companion'.")
    print("Next: start bridge.py, then talk to the agent.")


if __name__ == "__main__":
    main()
