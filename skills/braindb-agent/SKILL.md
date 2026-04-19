---
name: braindb-agent
description: Persistent memory across sessions via the BrainDB agent. Use at conversation start and whenever you need to recall what you know about the user or save new information to long-term memory.
allowed-tools: Bash Read
---

## BrainDB Memory Agent

BrainDB has its own internal agent (LiteLLM + NVIDIA NIM) that handles all memory operations. You don't call individual endpoints — you ask the agent in plain English via one endpoint: `POST http://localhost:8000/api/v1/agent/query`.

### Health check:
!`curl -sf http://localhost:8000/health > /dev/null 2>&1 && echo "OK" || echo "BRAINDB_DOWN"`

**If the output contains `BRAINDB_DOWN`**, the memory database is not running. Do this:

1. **Ask the user**: "BrainDB isn't running. Do you want me to start it for you?"
2. **Find the braindb repo** — look for a directory that has ALL of these:
   - `docker-compose.yml` at the root
   - `braindb/main.py`
   - `pyproject.toml` with `name = "braindb"`
   Search in: current dir, parent dirs (up to 3 levels), common locations like `~/source/repos/**/braindb/`.
3. **Start it**: `cd <braindb-path> && docker compose up -d`
4. **Cache the path**: `echo "<braindb-path>" > ~/.claude/skills/braindb-agent/.repo_path`
5. **Wait for it**: poll `curl -sf http://localhost:8000/health` for up to 30 seconds.
6. If healthy, proceed. If the user declines or start fails, proceed WITHOUT memory.

---

## RECALL — at conversation start, and whenever you need context

Ask the agent in natural language. It handles keyword formulation, multi-query search, graph traversal, and summarization.

> **Encoding note**: When constructing the curl JSON body, use ASCII characters only — plain hyphens (`-`), straight quotes (`"`, `'`), no em-dashes (`—`) or smart quotes. On Windows shells these get mangled in the JSON body and the server returns 400 Bad Request.

```bash
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me who the user is - role, expertise, preferences, recent projects."}'
```

For topic-specific recall:
```bash
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What do you know about the user React frontend experience?"}'
```

**Read the `answer` field** from the response. Use the summary to inform your response. Never paste raw JSON into the conversation.

---

## SAVE — after learning something new

Describe what you learned in natural language. The agent decides the entity type, picks keywords, generates embeddings, creates relations.

```bash
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Save: the user just told me they prefer simple code over abstractions. Source: user-stated. Connect to existing preference entities."}'
```

**Be proactive**: save user profile info, expertise, preferences, decisions, inferences you make about their working style. When in doubt, save it.

---

## Example queries

| Situation | Query to send to the agent |
|-----------|---------------------------|
| Start of conversation | `"Tell me who the user is - role, expertise, preferences, recent projects."` |
| User mentions a topic | `"What do you know about the user ML experience and AI projects?"` |
| User shares a fact | `"Save: user is working on the IR pipeline multilingual extraction. Connect to existing IR entities."` |
| User gives a preference | `"Save as rule: always prefer simple code over abstractions. Source: user-stated. Category: behavior."` |
| User asks about past work | `"What has the user shipped recently? Check facts with source=user-stated from the last month."` |
| Need to find duplicates | `"Find near-duplicate entities in memory."` |
| Explore the graph | `"What are the densest topics in memory? Which entities have the most connections?"` |

---

## Delegation — ask the agent to spawn a subagent for focused work

To keep the agent's main context clean and your conversation uncluttered, you can explicitly tell it to delegate heavy tasks. Just include "use a subagent" or "delegate this" in the query — the agent has a `delegate_to_subagent` tool that runs a fresh agent instance in its own context and returns only a summary.

### When to delegate

- Deep searches or graph exploration
- Duplicate detection, orphan cleanup, bulk relation work
- Any task that would produce a lot of intermediate tool output

### Examples

```bash
# Find duplicates without seeing intermediate results
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Delegate to a subagent: find near-duplicate facts and return top 10 pairs with their IDs."}'

# Deep search on a topic
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Use a subagent: explore every entity tagged with investor-relations and return a clean summary of what is there."}'

# Create relations in bulk
curl -s -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Spawn a subagent: find orphaned user-profile facts and create tagged_with links to relevant keyword entities."}'
```

Delegation is 1 level deep — subagents cannot spawn more subagents.

---

## Verbose mode — watch the agent work in real time

Set `AGENT_VERBOSE=true` in the server's `.env` (default is `false`). When enabled, every tool call the agent makes is logged to stdout with args and result preview. Watch it live:

```bash
docker logs braindb_api -f
```

The HTTP response itself is unchanged (just `{"answer": "..."}`). Logs go to the server stdout only — clean separation between the response payload and operational logging.

---

## Error handling

- If the agent call fails (connection refused, 500, timeout): proceed WITHOUT memory. Don't retry, don't block the conversation.
- If the answer mentions an ERROR: the agent tried but some tool failed. Carry on — use whatever partial information came back.
- Agent calls can take 5-30 seconds (LLM + multi-turn loop). Subagent calls can take 30-90 seconds. That's normal.
