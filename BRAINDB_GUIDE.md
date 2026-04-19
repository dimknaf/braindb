# BrainDB — LLM Agent Usage Guide

This is the memory system you (the LLM) can use to store and retrieve knowledge across sessions.
The API runs at **http://localhost:8000**. Everything is done via HTTP calls.

---

## Entity Types

| Type | What to store |
|------|--------------|
| `thought` | Inferences, hypotheses, observations — subjective, may be uncertain |
| `fact` | Objective information with a certainty score |
| `source` | URLs and links to external information |
| `datasource` | Full documents or files with rich content |
| `rule` | Behavioral guidelines for how you should act (THIS type) |

---

## Core Workflow

### Before answering anything non-trivial, always call:
```
POST /api/v1/memory/context
{"queries": ["topic 1", "topic 2"], "max_depth": 3, "max_results": 15}
```
This returns:
- Direct matches (fuzzy + full-text) across all queries, merged by best score
- Graph-connected entities up to 3 hops away (relevance fades: 100% -> 60% -> 30%)
- Always-on rules (always injected regardless of query)

Each item has a `final_rank` score. Trust higher-ranked items more.

You can also pass a single query for backward compatibility:
```
{"query": "single topic", "max_depth": 3}
```

### After learning something new, save it:
```
POST /api/v1/entities/facts      — for objective facts
POST /api/v1/entities/thoughts   — for your inferences/opinions
POST /api/v1/entities/sources    — for URLs you encounter
POST /api/v1/entities/rules      — for new behavioral guidelines
```

### Connect related items with relations:
```
POST /api/v1/relations
```
Relation types: `supports`, `contradicts`, `elaborates`, `refers_to`, `derived_from`, `similar_to`, `is_example_of`, `challenges`

---

## API Reference

### List / Filter Entities
```bash
# List all facts
curl http://localhost:8000/api/v1/entities?entity_type=fact&limit=50

# Filter by keyword
curl http://localhost:8000/api/v1/entities?keyword=user-profile&limit=50

# Filter by provenance source
curl http://localhost:8000/api/v1/entities?source=user-stated&limit=50

# Filter by minimum importance
curl http://localhost:8000/api/v1/entities?min_importance=0.7&limit=20

# Combine filters
curl "http://localhost:8000/api/v1/entities?entity_type=fact&source=user-stated&keyword=expertise&limit=20"
```
Query parameters: `entity_type`, `keyword`, `source`, `min_importance` (0-1), `limit` (1-200, default 50), `offset` (default 0).

### Get Entity by ID
```bash
curl http://localhost:8000/api/v1/entities/<UUID>
```

### Delete Entity
```bash
curl -X DELETE http://localhost:8000/api/v1/entities/<UUID>
```
Returns 204. Cascades to relations.

### Create a Fact
```bash
curl -X POST http://localhost:8000/api/v1/entities/facts \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The fact you want to store",
    "title": "Short title (optional)",
    "certainty": 0.9,
    "is_verified": false,
    "source": "user-stated",
    "keywords": ["keyword1", "keyword2"],
    "importance": 0.7,
    "notes": "Why you saved this, any caveats"
  }'
```

### Create a Thought
```bash
curl -X POST http://localhost:8000/api/v1/entities/thoughts \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Your inference or observation",
    "certainty": 0.6,
    "context": "What triggered this thought",
    "emotional_valence": 0.0,
    "keywords": ["topic"],
    "importance": 0.5
  }'
```

### Create a Rule
```bash
curl -X POST http://localhost:8000/api/v1/entities/rules \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The rule text",
    "always_on": true,
    "category": "behavior|ethics|personality|task|constraint",
    "priority": 80,
    "importance": 0.9
  }'
```
Rules with `always_on: true` are **always** returned with every context call (up to top 10 by priority).

### Create a Source (URL)
```bash
curl -X POST http://localhost:8000/api/v1/entities/sources \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Description of what this source contains",
    "title": "Article title",
    "url": "https://...",
    "domain": "example.com",
    "keywords": ["topic"],
    "importance": 0.6
  }'
```

### Create a Datasource (full document)
```bash
curl -X POST http://localhost:8000/api/v1/entities/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Full document text or summary",
    "title": "Document title",
    "file_path": "/path/to/file.md",
    "keywords": ["topic"],
    "importance": 0.7
  }'
```

### Update Any Entity (PATCH)
All 5 entity types support PATCH. Only send the fields you want to change:
```bash
# Update a fact
curl -X PATCH http://localhost:8000/api/v1/entities/facts/<UUID> \
  -H "Content-Type: application/json" \
  -d '{"notes": "Updated observation", "importance": 0.9}'

# Update a thought
curl -X PATCH http://localhost:8000/api/v1/entities/thoughts/<UUID> \
  -H "Content-Type: application/json" \
  -d '{"certainty": 0.8}'

# Update a rule
curl -X PATCH http://localhost:8000/api/v1/entities/rules/<UUID> \
  -H "Content-Type: application/json" \
  -d '{"always_on": true, "priority": 90}'

# Also: /entities/sources/<UUID>, /entities/datasources/<UUID>
```

### Create a Relation
```bash
curl -X POST http://localhost:8000/api/v1/relations \
  -H "Content-Type: application/json" \
  -d '{
    "from_entity_id": "<UUID>",
    "to_entity_id": "<UUID>",
    "relation_type": "supports",
    "relevance_score": 0.8,
    "importance_score": 0.7,
    "is_bidirectional": false,
    "description": "Why this relation exists",
    "notes": "Any evolving observations"
  }'
```

### Get Entity Relations
```bash
curl http://localhost:8000/api/v1/entities/<UUID>/relations
```
Returns all relations where the entity is either `from` or `to`.

### Get / Update / Delete a Relation
```bash
# Get
curl http://localhost:8000/api/v1/relations/<UUID>

# Update
curl -X PATCH http://localhost:8000/api/v1/relations/<UUID> \
  -H "Content-Type: application/json" \
  -d '{"relevance_score": 0.9, "notes": "stronger connection than initially thought"}'

# Delete
curl -X DELETE http://localhost:8000/api/v1/relations/<UUID>
```

### Search (fast, no graph traversal)
```bash
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "limit": 10}'
```

### Context (full retrieval — use this for most things)

**Multi-query** (recommended — covers multiple angles in one call):
```bash
curl -X POST http://localhost:8000/api/v1/memory/context \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["user profile expertise", "project architecture decisions"],
    "max_depth": 3,
    "max_results": 15,
    "include_always_on_rules": true
  }'
```

Each query runs fuzzy + full-text search independently. Seeds are merged keeping the **best score** per entity. One graph expansion runs on the combined seed set.

**Single query** (backward-compatible):
```bash
curl -X POST http://localhost:8000/api/v1/memory/context \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "max_depth": 3, "max_results": 30}'
```

### Entity Tree (explore connections)
```bash
curl http://localhost:8000/api/v1/memory/tree/<UUID>?max_depth=2
```
Returns the entity and all its graph connections organized by depth and relevance.

### Get All Rules
```bash
curl http://localhost:8000/api/v1/memory/rules
```

### Stats
```bash
curl http://localhost:8000/api/v1/memory/stats
```

### Activity Log
Every create, update, delete, search, context, ingest, and SQL query is logged. Query to reconstruct history.

```bash
# Recent activity
curl "http://localhost:8000/api/v1/memory/log?limit=20"

# Filter by operation (create, update, delete, search, context, ingest, sql_query)
curl "http://localhost:8000/api/v1/memory/log?operation=create&limit=20"

# History for a specific entity
curl "http://localhost:8000/api/v1/memory/log?entity_id=<UUID>"

# Since a timestamp
curl "http://localhost:8000/api/v1/memory/log?since=2026-04-08T00:00:00Z"
```

Response includes: `id`, `timestamp`, `operation`, `entity_type`, `entity_id`, `details`, `context_note`.

### Read-only SQL
For ad-hoc exploration. Only `SELECT` and `WITH` queries; 5s timeout; 1000 row limit.

```bash
curl -X POST http://localhost:8000/api/v1/memory/sql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"}'
```

Response: `{"columns": [...], "rows": [[...]], "row_count": N, "elapsed_ms": X}`

### Ingest a File as Datasource
Read a file from disk, hash it, count words, create a datasource entity.

```bash
curl -X POST http://localhost:8000/api/v1/entities/datasources/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/sources/article.md",
    "keywords": ["finance", "ml"],
    "importance": 0.7,
    "source": "document"
  }'
```

`file_path` is resolved relative to the repo root (mounted at `/app` in the container). Max 5 MB per file.

### BrainDB Agent — natural language queries

`POST /api/v1/agent/query` — instead of orchestrating individual API calls, send a plain English request and let BrainDB's internal agent handle it. The agent uses the OpenAI Agents SDK with LiteLLM (provider pluggable via `LLM_PROFILE` — default `deepinfra`, `nim` also supported) and has access to all 21 BrainDB operations as function tools.

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What do you know about the user role and recent projects?",
    "max_turns": 15
  }'
# {"answer": "The user is ...", "max_turns": 15}
```

**Save via the agent**:
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Save: user prefers simple code over abstractions. Source: user-stated. Connect to existing preference entities."}'
```

**Delegate to a subagent** (keeps main agent context clean for heavy work):
```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Delegate to a subagent: find near-duplicate facts and return top 10 pairs with their IDs."}'
```

The agent has these tools internally: `recall_memory`, `quick_search`, `save_fact`, `save_thought`, `save_source`, `save_rule`, `ingest_file`, `get_entity`, `list_entities`, `update_entity`, `delete_entity`, `create_relation`, `view_entity_relations`, `delete_relation`, `view_tree`, `search_sql`, `view_log`, `get_stats`, `generate_embeddings`, `delegate_to_subagent`, `submit_result`.

**Setup (pick a provider)**:
- **DeepInfra (default)**: set `LLM_PROFILE=deepinfra` and `DEEPINFRA_API_KEY=...` in `.env`. Get a key at https://deepinfra.com/
- **NVIDIA NIM**: set `LLM_PROFILE=nim` and `NVIDIA_NIM_API_KEY=...` in `.env`. Get a key at https://build.nvidia.com/
- Profiles live in `braindb/config.py::_LLM_PROFILES`. Add new providers there (e.g. `together`, `openai`) by adding a dict entry — no code change required.
- Optional override: set `AGENT_MODEL=` in `.env` to use a non-default model for the active profile.

**Verbose logging**: set `AGENT_VERBOSE=true` in `.env` to log every tool call to stdout (visible via `docker logs braindb_api -f`). The HTTP response stays clean — only `answer` and `max_turns`.

**Encoding**: when constructing the JSON body, use ASCII characters only (plain hyphens `-`, no em-dashes `—`). On Windows shells, special characters may get mangled and the server returns 400 Bad Request.

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok", "embeddings": true}
```

---

## Scoring Fields

| Field | Range | Meaning |
|-------|-------|---------|
| `importance` | 0-1 | How important this entity is overall |
| `certainty` | 0-1 | How confident you are (thoughts/facts) |
| `relevance_score` | 0-1 | How relevant a relation is |
| `emotional_valence` | -1 to 1 | Negative to positive sentiment (thoughts only) |
| `priority` | 1-100 | Rule priority (higher = more important) |

---

## Provenance — Tracking Where Information Came From

Every entity has an optional `source` field that tracks its origin:

| Value | Meaning |
|-------|---------|
| `user-stated` | The user explicitly said this |
| `agent-inference` | The agent inferred or observed this |
| `document` | Extracted from a file or document |
| `third-party` | Came from another person, system, or API |

Set `source` when creating any entity. Filter with `GET /entities?source=user-stated`.

This is complementary to `source_entity_id` (on facts — links to a specific source entity) and `derived_from` relations (graph connections). Use `source` for the KIND of origin, `source_entity_id` for WHICH specific entity, and relations for HOW they connect.

---

## How Search Works

The search uses a 4-tier scoring system:
1. **Full-text AND match** (all query words match) — highest weight (1.0)
2. **Full-text OR match** (any query word matches) — lower weight (0.3)
3. **Content trigram similarity** — fuzzy character matching (0.5)
4. **Title trigram similarity** — fuzzy title matching (0.3)

This means specific queries with terms that appear in stored content work best. Vague queries with stop words ("everything about X") may return fewer results. If you get 0 results, reformulate with more specific terms.

---

## Decay Behaviour

Memories fade over time (older = lower effective importance), but strengthen when accessed.
- Thoughts decay fastest (0.5%/day)
- Facts decay slowly (0.1%/day)
- Rules never decay

The `final_rank` in context results already accounts for decay.

---

## Categories for Rules

- `behavior` — how you act and communicate
- `ethics` — moral constraints
- `personality` — tone, style, character
- `task` — task-specific instructions
- `constraint` — hard limits

---

## Tips

1. **Keywords matter** — the more precise your keywords, the better retrieval works. Include both full terms and abbreviations: `["machine-learning", "ML"]`
2. **Relations are powerful** — linking entities enables graph traversal; a fact connected to a rule will surface that rule when the fact is found
3. **Notes are a log** — use `notes` on any entity to record how your understanding evolved
4. **always_on rules are limited to 10** — keep them high-signal; use on-demand rules for specifics
5. **access_count reinforces memory** — things you retrieve often stay important longer
6. **Multi-query for better recall** — use `queries` (array) instead of `query` (single) to search multiple angles at once
7. **Content should be concise** — 1-2 sentences, standalone, using full terms (not abbreviations)
8. **Use the tree endpoint** to explore how an entity connects to others: `GET /memory/tree/<id>`
9. **Use the list endpoint** to browse entities: `GET /entities?entity_type=fact&limit=50`
