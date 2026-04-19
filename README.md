# BrainDB

A memory database and REST API for LLM agents. Store and retrieve thoughts, facts, sources, documents, and behavioral rules — with fuzzy + semantic keyword search, graph traversal up to 3 hops, temporal decay, and always-on rule injection. Built to be driven externally by an LLM via HTTP calls.

It also ships with **its own internal agent** (OpenAI Agents SDK + LiteLLM with pluggable providers — DeepInfra by default, NIM / others via config) so external callers can talk to BrainDB in plain English via a single endpoint instead of orchestrating individual API calls.

---

## Why BrainDB?

Inspired by Karpathy's [LLM wiki idea](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — give an LLM a persistent external memory it can read and write. BrainDB takes that further by adding structure, retrieval, and a graph on top of the "plain markdown files" baseline.

- **vs. RAG.** RAG is stateless: embed documents, retrieve similar chunks on every query, stuff them into context. There's no notion of *an entity* that persists, accrues connections, or ages. BrainDB stores typed entities (thoughts, facts, sources, documents, rules) with explicit `supports` / `contradicts` / `elaborates` / `derived_from` / `similar_to` relations, combined fuzzy + semantic search, graph traversal up to 3 hops, and temporal decay so stale items fade while accessed ones stay sharp. Retrieval returns a ranked graph neighbourhood, not a pile of chunks.
- **vs. classic graph DBs** (Neo4j, Memgraph). Those are general-purpose graph stores with their own query languages and ops cost. BrainDB is purpose-built for LLM agents: a plain HTTP API designed for tool-calling, semantically meaningful fields (`certainty`, `importance`, `emotional_valence`), built-in text + pgvector search with geometric-mean scoring, always-on rule injection, automatic provenance, and runs on plain PostgreSQL + `pg_trgm` + `pgvector` — no new infrastructure to operate.
- **vs. markdown files as memory.** Markdown wikis are flat and unstructured: the LLM has to grep, read whole files into context, and manage linking by hand. BrainDB's entities are atomic, queryable, ranked, and self-connecting. Facts extracted from a document automatically link back to the source via `derived_from`; recall returns relevant nodes plus their graph neighbourhood; nothing needs to be read in full unless the agent asks for it.

---

## Entity Types

| Type | What it stores |
|------|---------------|
| `thought` | Inferences, hypotheses, subjective observations |
| `fact` | Objective information with certainty score |
| `source` | URLs and external references |
| `datasource` | Full documents or files |
| `rule` | Behavioral guidelines (`always_on` rules inject into every context call) |

All entities share: `keywords`, `importance`, `source` (provenance: `user-stated`, `agent-inference`, `document`, `third-party`), `notes`, `created_at`, `updated_at`, `access_count`.  
Relations connect any two entities with `relation_type`, `relevance_score`, `importance_score`, `description`, and `notes`.

---

## Setup

BrainDB runs as two Docker services — `api` and `watcher` — against an **external** PostgreSQL you provide. The whole setup is six steps.

### 1. Prerequisites

- Docker Desktop (or any Docker Engine)
- A PostgreSQL 16 instance reachable from Docker (see step 3 for three common options)
- The PostgreSQL extensions `pg_trgm` and `pgvector` must exist on the target database, and the connecting user must have permission to create them on first connection (migrations will `CREATE EXTENSION IF NOT EXISTS` on startup). If you don't have DB admin rights, ask an admin to pre-install both extensions.

### 2. Clone and configure

```bash
git clone https://github.com/dimknaf/braindb.git
cd braindb
cp .env.example .env
```

### 3. Point `.env` at your PostgreSQL

Edit `.env` and set `DATABASE_URL`. The value depends on **where your Postgres runs**:

**Option A — Postgres running as another Docker container on the same network** (e.g. a `postgres_container`):
```
DATABASE_URL=postgresql://postgres:password@postgres_container:5432/braindb
```
Make sure that container is attached to the `local-network` network from step 5.

**Option B — Postgres running on your host machine** (Docker Desktop's bridge lets the container reach the host):
```
DATABASE_URL=postgresql://postgres:password@host.docker.internal:5432/braindb
```

**Option C — Remote Postgres** (AWS RDS, Supabase, a home server, anything):
```
DATABASE_URL=postgresql://user:password@db.example.com:5432/braindb
```
Any reachable hostname/IP works — the connecting user just needs network access, auth, and the extensions mentioned in step 1.

### 4. Pick an LLM provider (for the internal agent)

The agent talks to any LiteLLM-supported backend. BrainDB ships with two profiles pre-configured: **DeepInfra** (default, fast, paid) and **NVIDIA NIM** (free tier, can be flaky).

In `.env`:
```
LLM_PROFILE=deepinfra        # or 'nim' — default is 'deepinfra'
DEEPINFRA_API_KEY=...        # if profile=deepinfra — get from https://deepinfra.com/
NVIDIA_NIM_API_KEY=...       # if profile=nim       — get from https://build.nvidia.com/
```

Only the key matching your chosen profile needs to be filled. Leave the other blank or absent.

Adding a third provider (Together, OpenAI, local vLLM, whatever) is a two-line entry in [`braindb/config.py::_LLM_PROFILES`](braindb/config.py) + an env var — no other code changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the recipe.

### 5. Create the Docker network, then bring the stack up

`docker-compose.yml` expects an external network called `local-network` so the `api` and `watcher` containers can reach your Postgres (and each other) by DNS name:

```bash
docker network create local-network   # one-time, ignore error if it already exists
docker compose up -d --build
```

If your Postgres is a container (Option A in step 3), attach it to this network too:
```bash
docker network connect local-network postgres_container
```

### 6. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","embeddings":true}
```

API at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`. Database migrations run automatically on startup.

Drop a markdown file into `data/sources/` and the watcher sidecar picks it up within ~7 seconds — see [File Ingestion](#file-ingestion) below.

---

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/entities/thoughts` | Save a thought |
| POST | `/api/v1/entities/facts` | Save a fact |
| POST | `/api/v1/entities/sources` | Save a source URL |
| POST | `/api/v1/entities/datasources` | Save a document |
| POST | `/api/v1/entities/rules` | Save a behavioral rule |
| GET | `/api/v1/entities/{id}` | Get any entity |
| PATCH | `/api/v1/entities/{type}/{id}` | Update entity |
| DELETE | `/api/v1/entities/{id}` | Delete entity |
| POST | `/api/v1/relations` | Create relation between entities |
| GET | `/api/v1/entities` | List/filter entities by type, keyword, source, importance |
| GET | `/api/v1/entities/{id}/relations` | View all relations for an entity |
| POST | `/api/v1/entities/datasources/ingest` | Read a file from disk and create a datasource entity |
| POST | `/api/v1/memory/search` | Fast fuzzy search |
| POST | `/api/v1/memory/context` | Full retrieval: fuzzy → graph → decay → rank |
| GET | `/api/v1/memory/tree/{id}` | Entity graph tree — connections by depth |
| GET | `/api/v1/memory/log` | Activity log — when and how things happened |
| POST | `/api/v1/memory/sql` | Read-only SQL queries (SELECT/WITH only) |
| POST | `/api/v1/memory/generate-embeddings` | Batch-generate keyword embeddings |
| GET | `/api/v1/memory/rules` | All active rules |
| GET | `/api/v1/memory/stats` | Counts and activity |
| POST | `/api/v1/agent/query` | Natural language query — internal agent handles recall/save/relate |

See [BRAINDB_GUIDE.md](BRAINDB_GUIDE.md) for full API reference with curl examples.

---

## How Retrieval Works

`POST /api/v1/memory/context` is the main endpoint:

1. **Multi-query search** — pass `queries: ["topic1", "topic2"]` to search multiple angles at once. Each query runs 4-tier scoring (AND fulltext, OR fulltext fallback, content trigram, title trigram), seeds are merged keeping the best score per entity.
2. **Keyword embeddings** — query terms are also matched against keyword entity embeddings (Qwen3-Embedding-0.6B, 1024-dim, cosine similarity). Text and embedding scores are combined via geometric mean (with a configurable penalty when only one signal matches).
3. **Graph traversal** up to 3 hops via relations, relevance fading: `1.0 → 0.6 → 0.3`
4. **Temporal decay** — memories fade over time, strengthen on access
5. **Final rank** = `combined_score × effective_importance × accumulated_relevance`
6. **Always-on rules** injected regardless of query

Single `query` (string) still works for backward compatibility.

---

## The BrainDB Agent

Instead of orchestrating individual API calls, you can talk to BrainDB in plain English via `POST /api/v1/agent/query`. The agent (built on the OpenAI Agents SDK + LiteLLM) decides which tools to call and returns a summary.

```bash
curl -X POST http://localhost:8000/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What do you know about the user role and recent projects?"}'

# {"answer": "The user is ...", "max_turns": 15}
```

The agent has 21 tools — every single BrainDB endpoint plus `delegate_to_subagent` (which spawns a fresh agent in its own context for focused deep work) and `submit_result` (which ends the loop).

**LLM provider — pluggable via `.env`**:

`LLM_PROFILE` selects the backend. Profiles are defined in [braindb/config.py](braindb/config.py) (`_LLM_PROFILES`) — currently `deepinfra` (default, model `google/gemma-4-31B-it`) and `nim` (NVIDIA NIM, model `google/gemma-4-31b-it`). Each profile is a model-prefix + env-var pair; adding a new one is a dict entry.

```
LLM_PROFILE=deepinfra         # or nim — default is deepinfra
DEEPINFRA_API_KEY=...         # required if profile=deepinfra (https://deepinfra.com/)
NVIDIA_NIM_API_KEY=...        # required if profile=nim (https://build.nvidia.com/)
AGENT_MODEL=                  # optional: override the profile's default model
```

**Verbose logging**: set `AGENT_VERBOSE=true` in `.env` to log every tool call (entry args + exit elapsed/result) to stdout, visible via `docker logs braindb_api -f`.

---

## Use with Claude Code (Skills)

This repo ships **two** Claude Code skills. Pick one (or install both):

| Skill | When to use |
|-------|------------|
| **[`skills/braindb/SKILL.md`](skills/braindb/SKILL.md)** | Direct curl-based recall/save. Claude formulates queries, calls individual API endpoints, writes saves explicitly. More verbose context, full control. |
| **[`skills/braindb-agent/SKILL.md`](skills/braindb-agent/SKILL.md)** | Thin wrapper that delegates everything to `POST /agent/query`. Claude sends a natural-language request, the internal agent does the work. Cleaner conversation context. |

Both auto-detect when BrainDB is down and offer to start `docker compose up -d` themselves. No hooks, no settings.json editing.

### Install

**Linux / macOS:**
```bash
# Direct skill
mkdir -p ~/.claude/skills/braindb
cp skills/braindb/SKILL.md ~/.claude/skills/braindb/SKILL.md

# Agent skill
mkdir -p ~/.claude/skills/braindb-agent
cp skills/braindb-agent/SKILL.md ~/.claude/skills/braindb-agent/SKILL.md
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.claude\skills\braindb"
Copy-Item "skills\braindb\SKILL.md" "$HOME\.claude\skills\braindb\SKILL.md"
New-Item -ItemType Directory -Force -Path "$HOME\.claude\skills\braindb-agent"
Copy-Item "skills\braindb-agent\SKILL.md" "$HOME\.claude\skills\braindb-agent\SKILL.md"
```

**Verify**: open a new Claude Code session. Type `/braindb` or `/braindb-agent` — the skill should load.

### Self-updating

The skill checks whether the repo copy has been updated (e.g. after `git pull`). If the repo version is newer than your personal copy, Claude will automatically copy the update and tell you. No manual re-install needed after the initial setup.

> **Single source of truth**: the skill lives at [`skills/braindb/SKILL.md`](skills/braindb/SKILL.md) in this repo. If you edit your personal copy at `~/.claude/skills/braindb/SKILL.md`, also update the repo copy (and send a PR) so everyone benefits.

### Optional: silent auto-start via SessionStart hook

If you'd rather have BrainDB always running (even before any skill is invoked), add a `SessionStart` hook to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "curl -sf http://localhost:8000/health > /dev/null 2>&1 || (cd /ABSOLUTE/PATH/TO/braindb && docker compose up -d > /dev/null 2>&1) || true",
            "async": true,
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Replace `/ABSOLUTE/PATH/TO/braindb` with your repo path. The hook is async (non-blocking).

## File Ingestion

Drop a file in `data/sources/` — the always-on watcher sidecar picks it up within 7s, ingests it, and runs a chunked fact-extraction pipeline that saves atomic facts into the knowledge graph linked back to the source via `derived_from` relations. Processed files move to `data/sources/ingested/`, failures to `data/sources/failed/` with an `.error.txt` sidecar.

```bash
cp ~/some-article.md data/sources/
docker logs braindb_watcher -f   # watch the pipeline
```

If you prefer to trigger ingestion explicitly from code, the endpoint still works:

```bash
curl -X POST http://localhost:8000/api/v1/entities/datasources/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/sources/article.md", "keywords": ["topic"], "importance": 0.7, "source": "document"}'
```

It's idempotent by content hash — re-calling with the same bytes returns 200 (existing) instead of 201 (new).

## Stack

- Python 3.12 + FastAPI + psycopg2 (sync, no ORM)
- PostgreSQL 16 with `pg_trgm` and `pgvector`
- Alembic migrations
- `sentence-transformers` + `Qwen/Qwen3-Embedding-0.6B` for keyword embeddings
- `openai-agents[litellm]` + LiteLLM for the internal agent (DeepInfra / NIM / others pluggable via `LLM_PROFILE`)
- Docker Compose — `api` + `watcher` services, external PostgreSQL
