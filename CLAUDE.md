# BrainDB — Claude Instructions

This project is a memory database and REST API designed to be driven by you (Claude) via HTTP calls.
The API runs at **http://localhost:8000**.

---

## ⚠ TOOL PRIORITY — read this first, it overrides habit

BrainDB's entire value is the **graph + embeddings + ranking**. Recall and
understanding must go through the sophisticated retrieval, never a flat SQL
`SELECT`.

1. **`POST /api/v1/memory/context`** (multi-query) — the default for ALL
   recall, discovery, disambiguation, "what do we know about X". BOTH
   the fuzzy and the embedding pathways are **keyword-mediated**: the
   query is matched against keyword-entity content (via pg_trgm) and
   keyword embeddings, then entities surface via `tagged_with`. A
   two-level diversity quota (per-search-term + per-keyword, geometric
   decay) keeps results balanced + graph traversal + temporal decay +
   `final_rank`.
2. **`POST /api/v1/agent/query`** (ask it to *delegate to a subagent* for
   anything multi-step) — research/investigation that needs several hops.
3. `GET /api/v1/entities…`, `/memory/tree/<id>`, `/entities/<id>/relations` —
   targeted structure lookups.
4. **`POST /api/v1/memory/sql` — exception ONLY.** A flat SELECT throws away
   embeddings, graph and ranking. Use it solely for a specific
   structured/aggregate question (counts, GROUP BY, activity-log joins) the
   above genuinely cannot express. **Never** for recall, discovery,
   similarity, or understanding. If you're using SQL to *find* or *understand*
   something, you're doing it wrong — use `/memory/context`.

**Previews vs full read:** all multi-item calls return short previews
(~1K/item; a clipped one ends `--truncated … get_entity("<id>")`). Read a
full body only by id: `GET /api/v1/entities/{id}`. For a large body, page it
with `?offset=&limit=` (follow `content_meta.next_offset`) or delegate it to a
subagent — never pull whole documents into context.

---

## At the Start of Every Session

Before doing any work, consult your memory:

```bash
# 1. Get always-on rules (behavioral guidelines)
curl -s http://localhost:8000/api/v1/memory/rules

# 2. Get context — use multi-query with NARROW queries for better coverage
curl -s -X POST http://localhost:8000/api/v1/memory/context \
  -H "Content-Type: application/json" \
  -d '{"queries": ["user-profile", "Dimitrios", "<one broader topic angle>"], "max_depth": 3}'
```

The context response gives you `items` (ranked memories) and `always_on_rules` (always injected).
Trust higher `final_rank` items more. Check `depth` — depth 0 is a direct match, 1-3 are graph-connected.

Multi-query runs each query independently, merges seeds (keeping the best score per entity), then does one graph expansion on the combined set. Use it to cover multiple angles in a single call.

**Query strategy**: prefer multiple **narrow** queries (single keywords / bare names) alongside one broader phrase, NOT a single long sentence. Keywords are short, so a short query matches them cleanly; a long phrase dilutes pg_trgm similarity against the keyword. The per-search-term diversity quota reserves slots for each query you pass, so a bare name like `"Petros"` will always surface its specific facts even when paired with broader semantic angles. `max_results` defaults to 30 — leave it unless you have a reason.

If results seem weak, retry with reformulated queries (up to 2 times).

---

## Project Structure

```
braindb/
├── braindb/
│   ├── main.py                        # FastAPI app
│   ├── config.py                      # Settings (decay rates, graph params, agent config)
│   ├── db.py                          # get_conn() — one psycopg2 connection per request
│   ├── ingest_watcher.py              # Always-on sidecar: polls data/sources/, triggers agent
│   ├── routers/                       # entities.py, relations.py, memory.py, agent.py, wiki.py, integrations.py
│   ├── schemas/                       # Pydantic models (entities, relations, search)
│   ├── services/                      # search.py, graph.py, context.py, embedding_service.py,
│   │                                  # keyword_service.py, activity_log.py
│   └── agent/                         # Internal agent (LiteLLM + pluggable provider via LLM_PROFILE)
│       ├── agent.py                   # builder + runner (singleton)
│       ├── tools.py                   # 21 @function_tool wrappers
│       └── prompts/system_prompt.md   # baked-in skill content (agent-voiced)
├── alembic/                           # DB migrations (raw SQL, no ORM) — current: 004
├── data/sources/                      # Drop files here — watcher picks them up
│   ├── ingested/                      # Auto-moved here on success
│   └── failed/                        # Auto-moved here on error (with .error.txt sidecar)
├── skills/                            # Shipped Claude Code skills
│   ├── braindb/SKILL.md               # Direct curl-based skill
│   └── braindb-agent/SKILL.md         # Thin wrapper around the agent endpoint
├── frontend/                          # Read-only browser UI — vanilla JS, no build (served at :8642)
├── integrations/hermes/               # BrainDB-as-memory for Hermes Agent (provider plugin + test sandbox)
├── docker-compose.yml                 # api + watcher + wiki_scheduler + frontend (+ optional bundled Postgres)
├── .env                               # Real credentials — DO NOT COMMIT
└── BRAINDB_GUIDE.md                   # Full API reference with curl examples
```

**No ORM** — all DB access is raw SQL via psycopg2 `RealDictCursor`.
**No async on the data layer** — plain sync `def` endpoints throughout, except the agent endpoint which is async to support the agent's `Runner.run()` loop.

---

## Saving What You Learn

```bash
# Save a fact
curl -X POST http://localhost:8000/api/v1/entities/facts \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "certainty": 0.9, "source": "user-stated", "keywords": ["k1","k2"], "importance": 0.7}'

# Save a thought / inference
curl -X POST http://localhost:8000/api/v1/entities/thoughts \
  -H "Content-Type: application/json" \
  -d '{"content": "...", "certainty": 0.6, "source": "agent-inference", "context": "what triggered this"}'

# Save a datasource (local file / document)
curl -X POST http://localhost:8000/api/v1/entities/datasources \
  -H "Content-Type: application/json" \
  -d '{"content": "description", "source": "document", "file_path": "/path/to/file", "keywords": ["topic"], "importance": 0.6}'

# Connect two entities
curl -X POST http://localhost:8000/api/v1/relations \
  -H "Content-Type: application/json" \
  -d '{"from_entity_id": "<id>", "to_entity_id": "<id>", "relation_type": "supports", "relevance_score": 0.8, "description": "why"}'
```

Relation types: `supports`, `contradicts`, `elaborates`, `refers_to`, `derived_from`, `similar_to`, `is_example_of`, `challenges`

---

## Useful Endpoints

```bash
# List entities by type / keyword / source
curl -s "http://localhost:8000/api/v1/entities?entity_type=fact&limit=50"
curl -s "http://localhost:8000/api/v1/entities?source=user-stated&limit=50"

# View entity relations
curl -s http://localhost:8000/api/v1/entities/<UUID>/relations

# Explore entity graph tree
curl -s http://localhost:8000/api/v1/memory/tree/<UUID>?max_depth=2

# Activity log (when and how things happened)
curl -s "http://localhost:8000/api/v1/memory/log?limit=20"
curl -s "http://localhost:8000/api/v1/memory/log?operation=create&entity_id=<UUID>"

# Read-only SQL (power queries)
curl -s -X POST http://localhost:8000/api/v1/memory/sql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"}'

# Ingest a file from data/sources/
curl -s -X POST http://localhost:8000/api/v1/entities/datasources/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/sources/article.md", "keywords": ["topic"], "importance": 0.7, "source": "document"}'

# Delete an entity
curl -s -X DELETE http://localhost:8000/api/v1/entities/<UUID>
```

---

## Two Paths: Direct API or Internal Agent

**Direct API** (what's shown above) — call individual endpoints yourself. Full control, more verbose context. Good when you want to be precise about what's saved or recalled.

**Agent endpoint** — `POST /api/v1/agent/query` — send a natural language request and let BrainDB's internal agent handle it. The agent (LiteLLM with pluggable provider via `LLM_PROFILE` — default `deepinfra/google/gemma-4-31B-it`, NIM also supported) has all 21 BrainDB operations as tools. Cleaner conversation context, but slower (5-30 seconds for a query).

```bash
# Recall via the agent
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What do you know about the user role and recent projects?"}'

# Save via the agent
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Save: user prefers simple code over abstractions. Source user-stated. Connect to existing preferences."}'

# Delegate to a subagent (keeps main agent context clean)
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Delegate to a subagent: find near-duplicate facts and return top 10 pairs."}'
```

When working **on this repo**, prefer direct API calls (you need precision). When working **in another project** that just wants memory, prefer the agent endpoint (less ceremony).

When debugging the agent: set `AGENT_VERBOSE=true` in `.env` and watch `docker logs braindb_api -f` — every tool call is logged with args and result preview.

**Encoding gotcha**: when passing JSON to the agent endpoint, use ASCII characters only (plain `-`, no em-dash `—`). Windows shells mangle Unicode and the server returns 400.

---

## Important Notes

- `.env` contains real DB credentials and provider API keys (`DEEPINFRA_API_KEY`, `NVIDIA_NIM_API_KEY`, etc.) — **never commit it**, it is in `.gitignore`. Active provider is picked by `LLM_PROFILE` (see `braindb/config.py::_LLM_PROFILES`). `LLM_PROFILE=deepinfra` (model `google/gemma-4-31B-it`) is the recommended starting point — fast, cheap, validated end-to-end; the `vllm_*` profiles need a GPU serving an OpenAI-compatible endpoint reachable from the docker network. Note the wiki pipeline's tuning constants (turn budget, request timeout, writer handoff budget, `AGENT_WIKI_REASONING_EFFORT`) were measured against self-hosted Qwen on vLLM — `vllm_workstation_qwen` is that profile.
- Always-on rules (priority 100, `always_on: true`) are returned on every `/memory/context` call
- `notes` field on any entity or relation is for running commentary — append observations over time
- Keywords are stored as both a `TEXT[]` column on the entity AND as separate keyword entities linked via `tagged_with` relations (the keyword entities carry the embeddings for semantic search)
- Full curl reference: [BRAINDB_GUIDE.md](BRAINDB_GUIDE.md)
