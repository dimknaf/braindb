# BrainDB test suite

Integration tests that exercise the real HTTP API against a live PostgreSQL and, for the agent smoke tests, a live LLM provider. No mocks, no stubs.

The suite runs against an **isolated test stack** — its own API (port 8002) and its own throwaway Postgres (`braindb_test`, host port 5436). It never touches your personal BrainDB database; tearing the stack down wipes all test data.

Pure unit files (`test_split_chunks.py`, `test_wiki_sections.py`, `test_final_answer_rename.py`, `test_handoff_hooks.py`, `test_runhooks_countdown.py`) need **no stack at all** — only tests that talk to the API or database require the test stack to be up.

## Prerequisites

Start the test stack:

```bash
docker compose -f docker-compose.test.yml up -d
curl http://localhost:8002/health   # must return {"status":"ok","embeddings":true}
```

Dev dependencies:

```bash
pip install -e ".[dev]"
```

When you're done (stops the stack and wipes all test data):

```bash
docker compose -f docker-compose.test.yml down -v
```

## Running

```bash
pytest                            # full suite
pytest -v                         # verbose
pytest tests/test_split_chunks.py # one file
pytest -k "not agent"             # skip the LLM-dependent agent smoke tests
pytest -x                         # stop on first failure
```

You can point tests at a non-default URL (only do this for another *disposable* stack — never a database with real data):

```bash
BRAINDB_TEST_URL=http://other-host:8002 pytest
```

A few tests drive internal services (e.g. `graph_expand`) over a direct DB connection; they default to the test stack's Postgres at `localhost:5436` and skip with a clear reason if it isn't reachable.

## What is covered

| File | What it tests |
|---|---|
| `test_split_chunks.py` | Pure function — empty text, single word, exact boundary, overlap correctness, misconfigured overlap degrades safely, word preservation, byte offsets. |
| `test_entities.py` | CRUD round-trip for all 5 entity types (fact, thought, source, datasource, rule). PATCH field isolation, DELETE idempotency, 404 on missing, list filters by type and keyword. |
| `test_relations.py` | Relation CRUD, inbound + outbound listing on an entity, PATCH updates, DELETE, cascade on entity deletion, invalid type rejection, all 8 documented relation types accepted. |
| `test_search.py` | `/memory/search` finds created content, `/memory/context` structure, multi-query seed merging, graph traversal surfaces connected entities, `/memory/tree` returns a structure, `/memory/stats` returns counts. |
| `test_ingest.py` | `/datasources/ingest` — 201 new, 200 duplicate (by content_hash), dup preserves first-seen metadata (second call doesn't overwrite). Test files live in `data_test/` (the test stack's data dir). |
| `test_agent.py` | `/agent/query` smoke — 200 with an `answer` field on a trivial prompt, 4xx on empty or missing query. Needs a configured LLM provider (credentials pass through from `.env`). |
| `test_jev.py`, `test_reader.py`, `test_hops.py`, `test_wiki_triage.py`, `test_wiki_gate.py` (+ their `*_db.py`), `test_context_jev_db.py` | JEV against fake model servers, no LLM: every decision is one token, a failed probe or request falls back to the behaviour without JEV, and with JEV off the agent's outputs and tool schemas are unchanged. |
| `test_usage_ledger.py` | The per-request usage ledger: every run of an `/agent/query` request adds into it, and it is recorded on failure too. |
| `test_graph_expand_db.py` | Recall's graph traversal against the path-listing query it replaced, on a graph with a hub, a cycle, parallel relations, a NULL score, a self-loop and tied seeds: same rows, order and values. |

## What is NOT covered

Intentional gaps so the suite stays reliable and fast:

- **Agent LLM output quality** — the agent smoke test only checks that the endpoint returns a well-formed response. It doesn't assert anything about the answer's content, because LLM output varies and external providers can be flaky.
- **End-to-end watcher pipeline** — the test stack deliberately runs no watcher (no background extraction → deterministic tests). Watcher behavior is verified manually when its logic changes.
- **Datasource content guardrail via the agent tool** — lives in `braindb/agent/tools.py::update_entity`. Testing it cleanly requires driving the agent loop, which needs the LLM.
- **Alembic migrations** — run once at container startup (the test stack exercises them on every fresh boot, but the suite doesn't assert on them).
- **Embeddings generation** — slow model load; covered implicitly by any search test that matches a seeded keyword but not asserted specifically.

## Expected runtime

- Without agent tests: **under 30s** on a warm stack
- With agent tests: **30–90s** depending on provider latency

## If tests fail

1. `docker logs braindb_test_api --tail 50` — the API may have errored
2. Health check: `curl http://localhost:8002/health`
3. Fresh state: `docker compose -f docker-compose.test.yml down -v && docker compose -f docker-compose.test.yml up -d` — brand-new empty test DB
