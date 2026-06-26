# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] — 2026-06-26

Headline: **custom profiles** — opt-in, self-contained overlays that reshape what BrainDB ingests
and how its auto-maintained wikis read, with ZERO effect on defaults when inactive. Two ship ready
to run: a keyless **Hacker News** chronicle that grows company/project/tech pages from the firehose,
and a **Hermes situational tracker** that asks an external web-search agent about an evolving
real-world situation and keeps a dated, sourced, dashboard-shaped status page. Also in this release:
the wiki writer now rejects blank/garbage bodies before persist, and the scheduler write-timeout
default is raised for slower local models.

### Added

- **Custom profiles framework.** An opt-in composition layer (`braindb/custom_profile.py`,
  `braindb/profile_runner.py`) that shapes ingestion + wiki output from a single self-contained
  folder under `custom-profiles/<name>/`, activated by `CUSTOM_PROFILE` — defaults are byte-for-byte
  unchanged when unset. A profile may carry `wiki_maintainer.add.md` / `wiki_writer.add.md` prompt
  fragments (appended after the base prompt; a `*.replace.md` swaps it wholesale) and an optional
  supervised `ingestor.py` (plus its own gitignored `.env`) that feeds a custom source into the
  existing file-watcher. Profiles compose (comma-separated). Core API, schema, and base prompts are
  untouched. Ships with an `example/` template and a `gdrive/` source ingestor alongside the two
  profiles below.
- **`hackernews` profile — keyless tech chronicle.** Polls the public Hacker News API, drops each new
  story into `data/sources/`, and shapes the resulting entity pages as tech-entity chronicles
  (type / homepage / category, a dated `current-developments` feed that ages into `background`, every
  claim cited). No API key; runs immediately.
- **`hermes` profile — situational tracker.** On a schedule, asks an external
  [Hermes Agent](https://github.com/NousResearch/hermes-agent) (web-search, OpenAI-compatible gateway)
  a dated, stateless question and folds each answer into a central situational wiki shaped as a
  dashboard — status + confidence + most-credible-source, then `now` / `recent-days` / `recent-weeks`
  / `forecast` / `history`, every line dated and `[[ref:UUID]]`-cited, conflicts resolved by
  credibility or flagged low-confidence — while the other entities each report mentions still get
  their own cross-linked wikis. Dormant by default (no key, so it idles and never calls); a
  `HERMES_SAMPLE_FILE` dry-run exercises the whole pipeline offline.
- **Current date in wiki prompts.** The maintainer and writer prompts now carry today's UTC date, so
  recency / "as of" / dated-bucket reasoning has an anchor.

### Changed

- **Wiki scheduler write-timeout default raised 1200 -> 2400s (20 -> 40 min).** `WIKI_AGENT_TIMEOUT`;
  smaller/slower quantised local models (e.g. Gemma 12B) can exceed 20 min on a full-body write, so
  the old default abandoned slow-but-completing writes and forced retries. Client-patience only — the
  API is not bounded by it and a write past the deadline still commits.

### Fixed

- **Writer rejects blank/garbage bodies before persist.** A weak model sometimes emits serialization
  junk (`""`, `body=""`, `{}`) instead of a page; the previous whitespace-only gate let it through to
  be written as content and bump the revision, self-perpetuating into empty, high-revision pages. A
  single blank-body check now routes such writes to the existing retry path instead of persisting —
  good pages can't be overwritten by junk, and the entity stays a healable orphan for the next cycle.

## [0.8.0] — 2026-06-16

Headline: the wiki maintainer now batches related orphan triage into ONE LLM call —
same-subject keywords that share a source fact are decided together, one decision per
seed — cutting maintainer LLM calls while keeping decisions and the job lifecycle
identical to before. Also in this release: a self-hosted Gemma 4 12B LLM profile and an
env-configurable Hermes ask-timeout.

### Added

- **Batched wiki triage (clustered maintainer).** When the maintainer runs and the
  claimed seed is a non-hub keyword, it also claims the other pending triage jobs whose
  entity shares a source fact (sibling keywords + that fact's own job) and decides them
  all in ONE agent call — one decision per seed. `run_cron` and the per-job lifecycle are
  unchanged (each job still closes with its own status); a singleton, or the flag off, is
  a one-seed unit identical to before. Gated by `WIKI_TRIAGE_CLUSTER` (default on, with
  `WIKI_TRIAGE_CLUSTER_MAX` / `WIKI_TRIAGE_HUB_DEGREE` to tune it). Validated A/B against
  the one-at-a-time path on the same snapshot: identical decision split, near-identical
  wikis, ~1.6 orphans decided per LLM call, and co-occurring seeds decided independently
  (no conflation).
- **`vllm_workstation_gemma12b` LLM profile.** A self-hosted vLLM profile for Gemma 4 12B;
  select it with `LLM_PROFILE=vllm_workstation_gemma12b` to run the internal agent on the
  local Gemma model.

### Changed

- **Hermes `MemoryProvider` ask-timeout is env-configurable.** `BRAINDB_ASK_TIMEOUT`
  (default 600s, previously a hardcoded 90s) so a slow `/agent/query` LLM loop does not
  time out before the agent finishes.

## [0.7.0] — 2026-06-14

Headline: BrainDB becomes a native long-term-memory backend for **Hermes Agent**
(Nous Research) — a small `MemoryProvider` plugin that exposes the whole of BrainDB
to a Hermes agent through one gateway tool, plus a file-ingest tool. Also in this
release: a fairer body-content signal in `/memory/context` recall, the
`openai_compatible` LLM profile, and a unit-test isolation fix.

### Added

- **Hermes Agent memory provider.** A native [Hermes Agent](https://github.com/NousResearch/hermes-agent)
  `MemoryProvider` plugin (`integrations/hermes/braindb/`) lets a Hermes agent use a
  self-hosted BrainDB as long-term memory through two tools: `braindb_ask` (one gateway
  to all of BrainDB — recall / save / relate / reason via BrainDB's own internal agent)
  and `braindb_ingest` (upload a local file for the watcher to fact-extract). Agent usage
  instructions load live from BrainDB's `/skill` endpoint, so there is no copied prompt to
  maintain. Backed by two additive, isolated endpoints in a new
  `braindb/routers/integrations.py` — `GET /api/v1/skill/{name}` and
  `POST /api/v1/entities/datasources/upload` — that change no existing route, schema, or
  behaviour. Ships with a tool-stripped Docker sandbox (`integrations/hermes/sandbox/`)
  for trying it on a throwaway host; verified end-to-end on an isolated Raspberry Pi.
  Install: copy the plugin folder into `~/.hermes/plugins/braindb`, then
  `hermes memory setup`. (#17)
- **`openai_compatible` LLM profile.** Point the internal agent at any
  OpenAI-compatible `/v1` endpoint (Ollama, LM Studio, copilot-api, a remote
  vLLM) without adding a provider abstraction: set
  `LLM_PROFILE=openai_compatible`, `AGENT_MODEL=openai/<model-id>`, and
  `OPENAI_BASE_URL` (plus `OPENAI_API_KEY` only if the server requires auth).
  The env-driven base URL is scoped to this profile alone — every other
  profile's `base_url` stays fixed in the table, so nothing is silently
  re-pointed. Based on @WarGloom's proposal in #7, reworked to the
  provider-prefix env convention and a profile-scoped override. (#16)

### Changed

- **`/memory/context` ranking — full-content signal + cleaner graph fade.**
  Retrieval is still keyword-mediated (fuzzy + embedding over keyword
  entities), but an entity's BODY now contributes a fair, additive signal:
  `seed = keyword_score + content_weight × c`, where `c` (0–1) is a
  body-only match (index-backed stemmed full-text + per-word
  `word_similarity`, with a size damper so long documents must match
  better). It can reinforce a keyword hit or surface a keyword-missed one,
  never lower a score; `content_weight=0` reproduces the old keyword-only
  behavior. This replaces a hard-coded 0.2 "discovery" fallback that
  double-penalized and discarded agreement. Retired/redirected wiki stubs
  are now excluded from results; the per-query reservation is
  importance-aware; and the final list is ordered by true rank. Graph
  hops now fade purely by the compounding per-edge `relevance × importance`
  product (the redundant 1.0/0.8/0.6 depth layer was removed). No public
  response field renamed.

### Fixed

- **Unit tests no longer demand the test stack.** The live-API guard in
  `tests/conftest.py` was session-wide (`autouse`), so even pure unit tests
  — including the two files CI runs — errored out when the isolated test
  stack wasn't up. The guard now applies only to tests that actually use
  the API fixture; unit tests run with no stack, and integration tests
  still fail fast with the start-the-stack instructions.

## [0.6.0] — 2026-06-12

### Added

- **Bundled internal Postgres (opt-in by one `.env` line).** A `braindb_db`
  service (pgvector/pg17) guarded by the `internal-db` compose profile.
  `.env.example` now ships `COMPOSE_PROFILES=internal-db`, so a fresh clone
  gets a working database with plain `docker compose up -d` — nothing to
  install. Existing setups are untouched: without that line the service does
  not exist and `DATABASE_URL` keeps pointing at your own Postgres exactly as
  before. Optional knobs: `POSTGRES_PORT` (host inspection port, default
  5435, loopback-only), `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`.
  Requires Docker Compose >= 2.20. Closes the request in issue #12.
- **Frontend served by the stack.** A `frontend` service (nginx:alpine over
  the static `frontend/` directory) joins `docker compose up -d` — the
  read-only browser UI is now at `http://localhost:8642` with zero manual
  steps (`FRONTEND_PORT` to change it; loopback-only). The manual
  static-server route still works as before.
- **Logo + architecture diagrams.** The BrainDB logo tops the README and the
  frontend header (plus favicon), and five SVG diagrams — the keyword-graph
  mechanism, retrieval pipeline, save/ingest lanes, wiki maintainer loop, and
  wiki writer pipeline — illustrate their README sections. Assets live in
  `docs/assets/`.

### Fixed

- **Test suite no longer touches your database — at all.** The suite now
  runs against an isolated test stack (`docker-compose.test.yml`: its own
  API on port 8002 + its own throwaway Postgres), and the session-end
  cleanup that deleted pattern-matched rows from the configured database
  has been removed entirely. Previously the tests ran against the live
  personal stack and their cleanup silently failed from the host, leaking
  `_pytest_*` keyword entities into real data (where the wiki maintainer
  wasted LLM calls triaging them). Now: `docker compose -f
  docker-compose.test.yml up -d && pytest`, and `down -v` wipes everything.

## [0.5.0] — 2026-06-09

### Changed

- **`WIKI_ENABLED` now defaults to `true`** in both `braindb/wiki_scheduler.py`
  and `docker-compose.yml`. A fresh `docker compose up` starts the wiki
  maintainer + writer automatically. The pipeline is part of what makes
  BrainDB a knowledge layer; idle ticks cost zero LLM (cheap SQL only),
  and the previous opt-in default was a friction point in onboarding.
  Set `WIKI_ENABLED=false` in `.env` to opt out (e.g. pure fact-store
  deployments).
- **Bench repo defaults flipped from workstation Qwen to deepinfra.**
  `docker-compose.bench.yml`'s `LLM_PROFILE_BENCH` default is now
  `deepinfra`; the judge endpoint defaults in `benchmarks/beam/config.py`
  also point at deepinfra. Self-hosted Qwen runs are still supported —
  set `LLM_PROFILE_BENCH=vllm_workstation_qwen` plus the judge overrides
  in `.env.bench` (see `.env.bench.example`).

## [0.4.0] — 2026-06-03

Headline: a focused pass on recall quality and the `view_tree` tool. The
per-edge LLM judgment that was missing on `create_relation` is now wired
through to graph scoring, and `view_tree` returns a nested JSON tree the
agent can actually navigate (vs the depth-grouped text that silently
clipped 70% of connections on popular wikis).

### Changed

- **`view_tree` / `GET /api/v1/memory/tree/<id>` — nested JSON shape.**
  Root keyed by `entity_type`, `children` arrays per node, multi-path
  first-wins by best accumulated path score, keyword + retired-wiki noise
  filtered by default, `_truncated` last-child marker when more remain.
  One shared builder (`build_entity_tree` in `braindb/services/tree.py`)
  for the HTTP endpoint and the agent tool — no behaviour drift. New
  optional query params: `include_keywords` (default `false`), `top_k`
  (default `40`), `min_path_score` (default `0.0`). Bench: Path B (Qwen
  27B) 5/5 PASS, **−25% wall-clock**, **−26% tool calls**, **zero
  `delegate` calls** on the hardest question (was 2). Path A (Claude
  Code + curl skill) 5/5 PASS, `view_tree` usage 0 → 1-2 calls — the
  structured shape is now usable in practice. Numbers in
  `benchmarks/runs/round-2f_comparison.md`.
- **`create_relation` writes both edge scores.** The `importance_score`
  column had been NULL for every agent-created row since day one; the
  parameter is now on the tool, the watcher's extraction prompt no
  longer dictates literal score values (the LLM judges per docstring),
  and the graph CTE multiplies `relevance_score × COALESCE(
  importance_score, 0.5) × depth_penalty` per hop. The
  `is_bidirectional` field on `relations` is now ignored by graph
  traversal — every edge walks both ways (matching what
  `services/tree.py` already did). The field stays in the schema for
  backwards compatibility.
- **Seed-similarity propagation in graph scoring.** Recall hops carry
  the seed's similarity score forward through the graph; the depth
  multiplier is softened so default-quality intermediates don't collapse
  depth-2/3 results.
- **Prompts**: `view_tree` reframed as a capability for "explore around
  this entity"; `search_sql` demoted to an aggregates-only exception.
  Same wording applied to `system_prompt.md`, both skill files
  (`skills/braindb/SKILL.md`, `skills/braindb-agent/SKILL.md`),
  `README.md`, and `BRAINDB_GUIDE.md`.

### Fixed

- **`view_tree` keyword noise**: keyword entities were leaking through
  non-`tagged_with` edges (e.g. `similar_to` between keywords). The
  filter now applies to target `entity_type='keyword'` as well as the
  edge type. Surfaced by the new nested JSON shape and fixed in the
  same commit.
- **`view_tree` duplicate retired-wiki siblings**: when the wiki
  maintainer creates a wiki and later consolidates duplicates, the
  retired ones used to still appear in `view_tree` output as duplicate
  siblings of the canonical wiki. The tree CTE now LEFT JOINs
  `wikis_ext` and skips rows where `retired_at IS NOT NULL`.
- **Test isolation in `tests/test_ingest.py`**: the three ingest tests
  used fixed content strings, so a previous run's row in the DB caused
  dedup-by-hash to fire and the 201 assertion to fail on subsequent
  runs. Each test now prepends a per-run `uuid.uuid4().hex` to its
  content.
- **Missing test dep**: added `pytest-asyncio==0.23.7` to
  `[project.optional-dependencies].dev`. Existing
  `@pytest.mark.asyncio` tests were silently failing on clean installs.

### Upgrading from v0.3.0

No DB migration. No env-var changes. The wiki maintainer's existing
retired-wiki pipeline now also gates `view_tree` traversal — old wikis
with `retired_at IS NOT NULL` are silently skipped in tree output (they
remain readable via `GET /api/v1/entities/<id>`). `pyproject.toml`
version field was at `0.2.0` in v0.3.0's tagged release (the bump was
missed); this release catches it up to `0.4.0` in one step.

## [0.3.0] — 2026-05-25

Headline: a small read-only frontend lands so humans can browse the same
knowledge graph the agents read and write to. Plus a YouTube walkthrough
linked from the README, and the merged contribution from @WarGloom that
tightens up the relations endpoints.

### Added

- **Frontend** (`frontend/`): vanilla-JS, no build step, no npm.
  - **Reader** — left rail wiki index + elastic-debounced semantic search
    with type-pill breakdown, central wiki body with citation chips,
    right-rail Relations panel that resolves every endpoint to a readable
    chip (type pill + canonical name) rather than a raw UUID.
  - **Graph** — force-layout view with custom shapes per entity type.
    Physics is one-shot (off after first settle) and new nodes added on
    expansion are placed in a deterministic ring around the click target,
    so the graph never drifts under the user's mouse. Zoom buttons,
    scroll-wheel zoom, drag-pan, retired-wiki badge in the search dropdown.
  - **Ops** — pipeline-queue and activity-log tables with actor pills
    (Maintainer / Writer / Scheduler / Watcher), readable target-wiki +
    entity chips, zebra stripes, and a NOTE column that distinguishes
    benign rationale from real errors.
  - **Ask drawer** — a textarea that posts to the agent endpoint
    (`POST /api/v1/agent/query`) with the same async semantics as the
    Claude Code skill (long calls supported).
  - **Universal entity-chip resolver**: a small async helper is wired into
    every render site that previously showed a UUID — relations, jobs,
    log rows, drawer titles. Cached after first fetch.
- **YouTube walkthrough**: linked from `README.md`
  ([youtu.be/AJ7iMOj4vvA](https://youtu.be/AJ7iMOj4vvA)).
- **README**: new "Frontend (optional, read-only)" section with one-line
  serve instructions.

### Changed

- **Relations endpoint hardening** — merges @WarGloom's
  [PR #5](https://github.com/dimknaf/braindb/pull/5): missing relation
  endpoints now reject cleanly instead of returning a confusing 200.

## [0.2.0] — 2026-05-24

The first substantial release beyond the v0.1.0 memory-store baseline. The
headline addition is the **wiki layer**: an always-on background pipeline
that turns the entity graph into self-maintaining, human-readable pages —
the same hands-off posture as the file watcher.

### Added

- **Wiki pipeline** (`braindb/wiki_scheduler.py`, `braindb/routers/wiki.py`):
  the in-house agent decides per-orphan whether to *attach* to an existing
  wiki, *create* a new one, *consolidate* duplicates, or *skip*. A separate
  writer agent then researches and writes/maintains each page, citing every
  claim with `[[ref:UUID]]`, with auto-self-healing on conflated subjects.
- **Wiki HTTP endpoints**: `POST /api/v1/wiki/cron` (orphan scan, idempotent),
  `POST /api/v1/wiki/maintain` (one triage decision per call),
  `POST /api/v1/wiki/write` (one writer pass), `GET /api/v1/wiki/jobs`
  (queue visibility). Normal operation is the scheduler sidecar; these are
  for hand-driving / observability.
- **Wiki section-edit tools**: `read_wiki_outline`, `read_wiki_section`,
  `edit_wiki_section`, `delete_wiki_section`, `validate_wiki` — let the
  writer do surgical edits on large pages without rewriting the full body.
- **Writer context-handoff**: when the writer's running context grows past
  a budget, it calls `handoff_to_successor` with a structured brief; the
  router respawns a successor agent with fresh context. Bounded by depth.
- **Typed agent termination**: every agent finish (`/agent/query`,
  maintainer, writer, subagent) is now a Pydantic model — schema-validated,
  no scraped free-text. Models live in `braindb/agent/schemas.py`.
- **Layer-4 retry-with-correction**: when a run ends without
  `final_answer`, the runner appends a synthetic correction message and
  re-invokes once with a small budget; recovers transparently.
- **`CountdownHooks` nudges**: a context-aware "wrap up" message arrives
  before `max_turns` is exhausted; a separate token-budget watch nudges
  the writer toward handoff when the conversation is getting big.
- **Auto-consolidation of duplicate wikis** via the maintainer's
  `consolidate` action, with reversible `wiki_revise` snapshots.
- **Per-wiki cooldown for attaches** in the scheduler so cron ticks don't
  thrash the same wiki across overlapping ticks.
- **Local vLLM profiles**: `vllm_workstation`, `vllm_workstation_qwen`,
  `vllm_workstation_gemma` for running against your own GPU box.
- **Tests**: session-teardown fixture in `tests/conftest.py` that sweeps
  any `_pytest_*` keyword artefacts that escape per-test cleanup.
- **CI**: minimal GitHub Actions workflow runs the typed-final + handoff
  unit tests on every PR + push to main.

### Configurable

New environment variables exposed in `.env.example` and consumed by the
api / wiki scheduler:

- `WIKI_ENABLED` — opt-in flag for the wiki scheduler (default `false`).
- `WIKI_INTERVAL` — scheduler tick in seconds (default `60`).
- `WIKI_FRESHNESS_MINUTES` — orphan eligibility gate; an entity must be
  this old before it's picked up for triage (default `30`).
- `WIKI_ATTACH_COOLDOWN_SECONDS` — per-wiki throttle between attach claims.
- `WIKI_AGENT_TIMEOUT` — HTTP timeout the scheduler uses for maintainer /
  writer calls (default `1200` seconds, i.e. 20 minutes).
- `AGENT_VERBOSE` — log every agent tool call with args and result preview
  (default `false`).

### Changed

- **Recall is keyword-mediated**: `/memory/context` now matches both the
  fuzzy (pg_trgm) and the embedding pathway against keyword entities, then
  surfaces facts via `tagged_with`. Two-level diversity quota
  (per-search-term + per-keyword, geometric decay) prevents one popular
  hub keyword from monopolising top-N. Narrow short queries outperform
  long phrases for keyword recall.
- **Multi-item recall returns previews**: `/memory/context` and
  `list_entities` now return short (~1 KB) previews per item; the full
  body is fetched on demand via `GET /api/v1/entities/{id}`, with optional
  `?offset=&limit=` paging for large documents. Keeps the LLM-visible
  context tight without losing access to the underlying content.
- **`deepinfra` (`google/gemma-4-31B-it`) promoted as the recommended
  default** across README, BRAINDB_GUIDE, CLAUDE, and CONTRIBUTING. Fast
  (5–30s per agent call), cheap, validated end-to-end. The `vllm_*`
  profiles are now documented as advanced / offline / requires GPU.
- **`WIKI_ENABLED` defaults to `false`** in compose so the scheduler
  sidecar boots but doesn't tick until explicitly opted in — keeps a
  fresh clone from spending on the LLM by accident.
- **Agent `max_turns` defaults bumped** (15 → 20) and `countdown_threshold`
  (5 → 8) after live observation on slower providers; deepinfra/Gemma is
  unaffected because it finishes well before the budget.
- **Wiki scheduler** collapsed three timers into one gated loop — no idle
  LLM spend, parallel maintain + writer fan-out per tick.
- **Skill files**: agent-call timeout guidance bumped to 10 minutes max
  for slow providers; wiki awareness + always-ASK-before-saving added.

### Fixed

- **Double-escaped JSON tool-call payload** (Qwen AWQ-INT4 quirk):
  `_maybe_parse_json_string` now unwraps the second layer when needed.
  Compliant providers (deepinfra/OpenAI/Anthropic via LiteLLM) unaffected.
- **JSON-string tool-call payload** (vLLM/Qwen format): typed schemas
  accept `arguments.payload` as either a JSON object or a JSON-encoded
  string of a dict; the LLM-visible contract is unchanged.
- **Writer no-op on already-cited members** no longer leaks the orphan
  back into the triage queue — it now closes the loop cleanly.
- **Big-body writes** retry on transient `BadRequestError` and stub out
  the body when the provider truncates, so the wiki isn't lost.
- **Reference-by-catalog-number** in maintainer prompts replaced the
  earlier uuid form to stop hallucinated wiki IDs.
- **Stale assigned jobs** in `wiki_job` are reclaimable on the next cron
  tick (stale-lease).
- **`output_type` dropped from agent builder** — restored tool use; typed
  `final_answer` still enforced via mutable-slot capture.
- **Compose**: no more `--reload` on the api command — code changes apply
  explicitly via `docker compose up -d --no-deps --force-recreate api`,
  preventing mid-run reloads that broke in-flight LLM calls.

### Upgrading from 0.1.0

Migration `005_wiki_system.py` adds two new tables (`wikis_ext`,
`wiki_job`) and the `wiki` entity type. It runs automatically on
container startup via `alembic upgrade head` (already in the api
`command`). Existing rows are untouched; no manual action required.

The wiki scheduler ships **disabled by default** — set
`WIKI_ENABLED=true` in `.env` to opt in. This prevents an upgraded
deployment from spending on the LLM until the operator says go.

## [0.1.0] — initial public baseline

Memory store: entities (`thought`, `fact`, `source`, `datasource`, `rule`),
relations, `pg_trgm` + `pgvector` retrieval, the BrainDB agent
(`/api/v1/agent/query`), the always-on file watcher (`data/sources/`),
Claude Code skills.
