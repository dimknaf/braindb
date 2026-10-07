---
name: braindb-agent
description: Persistent memory across sessions via the BrainDB agent. Use at conversation start and whenever you need to recall what you know about the user or save new information to long-term memory.
allowed-tools: Bash Read
---

## BrainDB Memory Agent

BrainDB has its own internal agent (OpenAI Agents SDK with pluggable provider via `LLM_PROFILE`; defaults to DeepInfra's `google/gemma-4-31B-it`) that handles all memory operations. You don't call individual endpoints — you ask the agent in plain English via one endpoint: `POST http://localhost:8000/api/v1/agent/query`.

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

## TOOL PRIORITY (read this first)

The agent has a clear order of tools it should reach for. When you phrase a
request, lean into the sophisticated tools — don't ask it to "run SQL" for
anything to do with recall or understanding.

1. **Query-driven recall** — *"what do we know about X?"* → the agent calls
   `/memory/context` (keyword-mediated fuzzy + embedding + graph + ranking,
   with diversity quotas). The default for ALL discovery and understanding.
2. **Entity-driven neighbourhood** — the agent's `view_tree` returns a
   nested JSON tree (root keyed by `entity_type`, `children` arrays per
   node, 1-N hops out, keyword + retired-wiki noise filtered by default,
   `_truncated` marker if more remain). Especially useful when an entity
   ID is already in hand — often sharper than another query about the
   same entity. On hub entities pass `max_depth=3` for narrative chains.
3. **Multi-step investigation** — *"investigate / disambiguate / resolve X"*
   → the agent delegates to a subagent. Keeps the main context clean.
4. **Direct lookups** — `view_entity_relations`, `get_entity`, `list_entities`
   for narrow questions.
5. **`search_sql` ⚠ exception only** — for explicit aggregates (counts,
   GROUP BY, activity-log joins). Never for finding / understanding /
   "what's related to" — those are jobs for the tools above.

If you're tempted to phrase a request as *"run a SQL query that finds…"* for
*finding* or *understanding* something, stop — that's the recall or tree
path's job. Ask in plain English.

**Wikis** are first-class memory entities curated by an internal maintainer +
writer pipeline. The agent surfaces them through recall automatically when
relevant — you don't have to ask for them explicitly, and you don't have to
trigger anything to make new ones. Saving facts with the right keywords is
enough; the scheduler runs maintain → write on its 60s tick and the wikis
materialise on their own.

Internally the agent now researches from **short previews** and reads a full
body only by id (paging large ones, or delegating big documents to a
subagent), so its context stays clean — just ask in natural language ("read
and summarise datasource X"); it handles the chunking itself.

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

### Proactive save — but ASK the user first

The pattern is **RECALL → ASK → SAVE**:

1. When the user shares something that *might* be worth remembering (a name,
   role, project, preference, decision, your own inference about them), RECALL
   first via the agent to check if it's already known.
2. If it's **net-new**, **ASK the user**:

   > "I haven't seen this before — should I save it to BrainDB? I'd file it
   > as a [fact / thought / rule] tagged with [keywords]."

3. Only on a 'yes', issue the save request to the agent.

Don't pre-save without confirmation. The user has the final say on what
becomes long-term memory. User-confirmed memory is higher-signal and lets
the user catch judgement-call mistakes early.

**Exception**: when the user explicitly framed it as a rule ("from now on,
always X"; "never do Y"), save it without an extra confirmation — they
already said it — but surface the action: "Saving that as a rule."

#### What's worth flagging to the user

- Identity / role / company (one-time setup info)
- Strong preferences or working-style rules
- Project / topic context the user just disclosed
- Decisions the user explicitly made
- Useful URLs or references the user shared
- Your own inferences about the user (tag as `thought`,
  `source=agent-inference`) — ASK before persisting these too; an inference
  is still memory.

The goal is to capture **what the user gives you in conversation that isn't
already in BrainDB** — not to scrape every utterance. Information already in
recall doesn't need saving again; ephemeral task details
("currently debugging X") don't need saving at all.

---

## Example queries

### Recall (no confirmation needed — these are reads)

| Situation | Query to send to the agent |
|-----------|---------------------------|
| Start of conversation | `"Tell me who the user is - role, expertise, preferences, recent projects."` |
| User mentions a topic | `"What do you know about the user ML experience and AI projects?"` |
| User asks about past work | `"What has the user shipped recently? Check facts with source=user-stated from the last month."` |
| Need to find duplicates | `"Find near-duplicate entities in memory."` |
| Explore the graph | `"What are the densest topics in memory? Which entities have the most connections?"` |

### Save (RECALL → ASK → SAVE — only send the agent query after the user confirms)

| Situation | What Claude says to the user first | What Claude sends to the agent (on a 'yes') |
|---|---|---|
| User mentions something net-new | "I noticed you just said you're working on the IR pipeline multilingual extraction — that looks worth saving. Should I?" | `"Save: user is working on the IR pipeline multilingual extraction. Connect to existing IR entities."` |
| User shares a preference | "Should I save that as a long-term preference?" | `"Save as fact: user prefers simple code over abstractions. Source: user-stated. Keywords: user-preference, code-style."` |
| User explicitly states a rule | (no confirmation — they framed it as a rule) "Saving that as a rule." | `"Save as rule: always prefer simple code over abstractions. Source: user-stated. Category: behavior."` |
| You drew an inference about the user | "I'm getting the sense you're senior in ML — should I save that as a thought?" | `"Save as thought: user appears senior in ML based on the depth of their question. Source: agent-inference. Certainty: 0.6."` |

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

## File ingestion — automatic, no agent call needed

If the user wants a local file (article, transcript, note, document) ingested into BrainDB, **don't ask the agent to do it**. Instead, copy the file into the repo's `data/sources/` directory and the system handles the rest:

1. The `braindb_watcher` sidecar polls `data/sources/` every ~7 seconds.
2. New files are auto-ingested as `datasource` entities (content + hash + word count).
3. The watcher then runs an agent-driven extraction pass that creates one or more `fact` entities derived from the document and links them back via `derived_from` relations.
4. On success the file is moved to `data/sources/ingested/`; on failure to `data/sources/failed/` with a sidecar `.error.txt`.

What this means for you (Claude) and the user:

- **Tell the user**: "Just drop the file into `data/sources/` on the BrainDB repo. The watcher will pick it up within a few seconds and you'll see the facts appear in recall a minute or two later."
- **Do not** issue an `/agent/query` like `"Save this file..."` with the file contents pasted into the prompt — that bloats the LLM context and bypasses the proper extraction pipeline. The watcher path produces structured facts + `derived_from` relations + keyword auto-tagging; pasting bypasses all of it.
- **Watch progress** if you want to confirm completion:

```bash
docker logs braindb_watcher -f
```

You'll see `ingested NEW: <filename> -> <id> words=N` then later `extraction complete for <id>: N facts total`. After that the new facts surface naturally in `/agent/query` recall — no extra steps.

Edge cases:
- Very large files are chunked automatically; extraction takes proportionally longer (typically 60-180 seconds per chunk on local Qwen, faster on deepinfra).
- If a file ends up in `data/sources/failed/`, read the sidecar `.error.txt` next to it to see what went wrong.
- The watcher dedupes by file content hash, so re-dropping the same file won't re-extract.

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
- Agent calls can take up to 10 minutes if the LLM provider is slow. Add `--max-time 600` to long curl calls.
