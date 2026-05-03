# Contributing to BrainDB

Thanks for your interest. BrainDB is a small, opinionated project — the bar is "does this make the memory system more useful for LLM agents?" If your change meets that bar, we'd love the PR.

## Dev setup

Prerequisites: Docker Desktop (or any Docker Engine), Python 3.12, a Postgres 16 instance reachable from the container.

```bash
git clone <repo-url> braindb
cd braindb
cp .env.example .env
# edit .env — set DATABASE_URL, pick an LLM_PROFILE, fill in the matching API key or OpenAI-compatible endpoint

docker network create local-network       # one-time; docker-compose expects this
docker compose up -d --build
curl http://localhost:8000/health         # {"status":"ok","embeddings":true}
```

For a native Python workflow (tests, IDE imports):

```bash
python -m venv .venv && source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

## Running tests

```bash
pytest                              # full suite (needs the stack up)
pytest -k "not agent"               # skip the live-LLM smoke tests
pytest tests/test_split_chunks.py   # a single file
```

See [`tests/README.md`](tests/README.md) for what is and isn't covered.

## Adding a new LLM provider

LiteLLM does the heavy lifting — providers are selected by a prefix in the model string. To add a provider:

1. Open [`braindb/config.py`](braindb/config.py) and add an entry to `_LLM_PROFILES`:
   ```python
   "my_provider": {
       "model": "my_provider/vendor/model-id",   # exact string LiteLLM expects
       "api_key_env": "MY_PROVIDER_API_KEY",
   },
   ```
2. Add `MY_PROVIDER_API_KEY=` to [`.env.example`](.env.example) if the provider needs auth.
3. Add the env passthrough to [`docker-compose.yml`](docker-compose.yml) under the `api` service. OpenAI-compatible endpoints can use `LLM_PROFILE=openai_compatible` plus `AGENT_BASE_URL` / `AGENT_API_KEY` variables.
4. (Optional) Document the provider in the README and BRAINDB_GUIDE.

No other code changes required — the agent resolves model and key through `settings.resolved_agent_model` and `settings.resolved_api_key`, which read the active profile.

### Self-hosted OpenAI-compatible servers (vLLM, Ollama, llama.cpp)

For a server you run yourself that speaks the OpenAI REST shape, the profile takes an optional third field, `base_url`, and uses LiteLLM's `openai/` prefix to route through the OpenAI-compatible code path:

```python
"vllm_workstation": {
    "model": "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "api_key_env": "VLLM_API_KEY",
    "base_url": "http://host.docker.internal:8002/v1",
},
```

When `base_url` points at the Docker host (`host.docker.internal`), the `api` service in [`docker-compose.yml`](docker-compose.yml) needs `extra_hosts: ["host.docker.internal:host-gateway"]` so the container can reach the host's loopback. The compose file in this repo already declares it.

If your server runs without auth, leave the matching `*_API_KEY` env var unset — `settings.resolved_api_key` falls back to the literal `"EMPTY"` for any profile that has a `base_url`, which keeps the OpenAI client happy.

## Adding a new database migration

BrainDB uses raw-SQL Alembic migrations (no ORM). Current revision is in `alembic/versions/`.

```bash
# Create a new revision file from a template
alembic revision -m "short description" --rev-id=005
```

Edit the generated file's `upgrade()` and `downgrade()` functions with raw SQL. Migrations run automatically on container startup (see `docker-compose.yml`'s command), but you can run them manually:

```bash
docker exec braindb_api alembic upgrade head
docker exec braindb_api alembic downgrade -1    # roll back one step
```

Keep migrations small and independently reversible where possible.

## Code style

- Python 3.12. Prefer modern type hints (`list[str]`, `str | None`), f-strings, dataclasses or Pydantic models where appropriate.
- Raw SQL via `psycopg2` with `RealDictCursor` — no ORM, don't introduce one.
- Sync `def` endpoints; the only async path is the agent loop.
- No internal "framework" abstractions; the project is small enough that clarity beats indirection.

## Pull requests

- One logical change per PR. A feature + its tests in the same PR is fine; a feature plus unrelated cleanup is not.
- If your change touches the agent's toolset, the watcher pipeline, or the data model, update both:
  - [`BRAINDB_GUIDE.md`](BRAINDB_GUIDE.md) — user-facing API reference
  - [`CLAUDE.md`](CLAUDE.md) — project context for the LLM assistants working in this repo
- Add tests for any new HTTP endpoint, tool, or scorer. The suite should stay green.

## Reporting issues

Include:
- What you tried (exact command or curl, expected result)
- What happened instead (response body, docker logs, stack trace)
- `docker logs braindb_api --tail 100` often has the real story

## License

By contributing you agree your contributions are licensed under Apache 2.0 — the same license as the rest of the project. See [`LICENSE`](LICENSE).
