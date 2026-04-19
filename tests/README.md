# BrainDB test suite

Integration tests that exercise the real HTTP API against a live PostgreSQL and, for the agent smoke tests, a live LLM provider. No mocks, no stubs — tests hit the running stack.

## Prerequisites

The BrainDB stack must be up and healthy:

```bash
docker compose up -d
curl http://localhost:8000/health   # must return {"status":"ok","embeddings":true}
```

Dev dependencies:

```bash
pip install -e ".[dev]"
```

## Running

```bash
pytest                            # full suite
pytest -v                         # verbose
pytest tests/test_split_chunks.py # one file
pytest -k "not agent"             # skip the LLM-dependent agent smoke tests
pytest -x                         # stop on first failure
```

You can point tests at a non-default URL:

```bash
BRAINDB_TEST_URL=http://other-host:8000 pytest
```

## What is covered

| File | What it tests |
|---|---|
| `test_split_chunks.py` | Pure function — empty text, single word, exact boundary, overlap correctness, misconfigured overlap degrades safely, word preservation. |
| `test_entities.py` | CRUD round-trip for all 5 entity types (fact, thought, source, datasource, rule). PATCH field isolation, DELETE idempotency, 404 on missing, list filters by type and keyword. |
| `test_relations.py` | Relation CRUD, inbound + outbound listing on an entity, PATCH updates, DELETE, cascade on entity deletion, invalid type rejection, all 8 documented relation types accepted. |
| `test_search.py` | `/memory/search` finds created content, `/memory/context` structure, multi-query seed merging, graph traversal surfaces connected entities, `/memory/tree` returns a structure, `/memory/stats` returns counts. |
| `test_ingest.py` | `/datasources/ingest` — 201 new, 200 duplicate (by content_hash), dup preserves first-seen metadata (second call doesn't overwrite). |
| `test_agent.py` | `/agent/query` smoke — 200 with an `answer` field on a trivial prompt, 4xx on empty or missing query. |

Every test that creates entities self-registers its IDs and cleans up in teardown. Your existing data (the real facts and datasources in the DB) is not touched.

## What is NOT covered

Intentional gaps so the suite stays reliable and fast:

- **Agent LLM output quality** — the agent smoke test only checks that the endpoint returns a well-formed response. It doesn't assert anything about the answer's content, because LLM output varies and external providers can be flaky.
- **End-to-end watcher pipeline** — dropping a file in `data/sources/` and waiting for the chunked fact-extraction + central review to complete is slow (~5 min) and depends on the LLM being responsive. Manually verified during Phase A with the Smart Sand article; re-run when the watcher logic changes.
- **Datasource content guardrail via the agent tool** — lives in `braindb/agent/tools.py::update_entity`. Testing it cleanly requires driving the agent loop, which needs the LLM. The behavior was manually verified during Phase A (content preserved across three watcher runs).
- **Alembic migrations** — run once at container startup. Not exercised in the Python test suite.
- **Embeddings generation** — slow model load; covered implicitly by any search test that matches a seeded keyword but not asserted specifically.

## Expected runtime

- Without agent tests: **under 10s** on a warm stack
- With agent tests: **30–90s** depending on provider latency

## If tests fail

1. `docker logs braindb_api --tail 50` — the API may have errored or be mid-reload
2. `docker logs braindb_watcher --tail 50` — the watcher sidecar shouldn't be relevant for tests but look if confused
3. Health check: `curl http://localhost:8000/health`
4. Fresh state: `docker compose restart api` (picks up bind-mount code changes if you've been editing)
