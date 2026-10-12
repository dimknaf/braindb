# Contributing to BrainDB

Thanks for your interest. BrainDB is a small, opinionated project — the bar is "does this make the memory system more useful for LLM agents?" If your change meets that bar, we'd love the PR.

## Dev setup

Prerequisites: Docker Desktop (or Docker Engine) with Compose v2.20+, Python 3.12. The database is bundled by default (`COMPOSE_PROFILES=internal-db` in `.env.example`); to use your own Postgres instead, remove that line and set `DATABASE_URL` — see the README's Setup section.

```bash
git clone <repo-url> braindb
cd braindb
cp .env.example .env
# edit .env — recommended LLM_PROFILE=deepinfra + DEEPINFRA_API_KEY (or any other profile)

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

The suite runs against an **isolated test stack** (its own API on port 8002 + its own throwaway Postgres) — never against your personal BrainDB database:

```bash
docker compose -f docker-compose.test.yml up -d     # start the test stack
pytest                              # full suite
pytest -k "not agent"               # skip the live-LLM smoke tests
pytest tests/test_split_chunks.py   # a single file
docker compose -f docker-compose.test.yml down -v   # stop + wipe test data
```

See [`tests/README.md`](tests/README.md) for what is and isn't covered.

## Adding a new LLM provider

The reference implementation and recommended default is `deepinfra` with `deepseek-ai/DeepSeek-V4.1-Flash` — fast, cheap, and its replies carry the logprobs JEV reads. Other providers are configured the same way.

Every profile is an OpenAI-compatible Chat Completions endpoint, driven by the OpenAI Agents SDK through the OpenAI client. To add a provider:

1. Open [`braindb/config.py`](braindb/config.py) and add an entry to `_LLM_PROFILES`:
   ```python
   "my_provider": {
       "model": "vendor/model-id",                      # the provider's own model id
       "api_key_env": "MY_PROVIDER_API_KEY",
       "base_url": "https://api.my-provider.example/v1",
   },
   ```
   If it is a hosted gateway (a real key is always required), also add its name to `_HOSTED_PROFILES`.
2. Add `MY_PROVIDER_API_KEY=` to [`.env.example`](.env.example).
3. Add the env passthrough to [`docker-compose.yml`](docker-compose.yml) under the `api` service.
4. (Optional) Document the provider in the README and BRAINDB_GUIDE.

No other code changes required — the agent resolves model, key and endpoint through `settings.resolved_agent_model`, `settings.resolved_api_key` and `settings.resolved_base_url`, which read the active profile. A provider that does *not* speak the OpenAI protocol would need a different model class in `braindb/agent/agent.py::_model()`, the one place the model is built.

### Self-hosted OpenAI-compatible servers (vLLM, Ollama, llama.cpp)

If you just want to point at *any* OpenAI-compatible `/v1` endpoint without touching code, use the built-in **`openai_compatible`** profile: set `LLM_PROFILE=openai_compatible`, `AGENT_MODEL=<model-id>`, and `OPENAI_BASE_URL` in your `.env` (plus `OPENAI_API_KEY` if the server needs auth). It takes its base URL from the environment — uniquely; every other profile's `base_url` is fixed in the table below.

To **bake in** a recurring self-hosted target as its own named profile instead, add an entry with its `base_url`:

```python
"vllm_workstation": {
    "model": "cyankiwi/gemma-4-31B-it-AWQ-4bit",
    "api_key_env": "VLLM_API_KEY",
    "base_url": "http://host.docker.internal:8002/v1",
},
```

When `base_url` points at the Docker host (`host.docker.internal`), the `api` service in [`docker-compose.yml`](docker-compose.yml) needs `extra_hosts: ["host.docker.internal:host-gateway"]` so the container can reach the host's loopback. The compose file in this repo already declares it.

If your server runs without auth, leave the matching `*_API_KEY` env var unset — `settings.resolved_api_key` falls back to the literal `"EMPTY"` for any self-hosted profile (one with a `base_url` that is not a hosted gateway), which keeps the OpenAI client happy. Model ids written with a profile's old LiteLLM prefix (`deepinfra/`, `nvidia_nim/`, or `openai/` / `hosted_vllm/` on self-hosted profiles) are still accepted, with a warning; a self-hosted model id that itself starts with `openai/` is then written with the prefix twice (`openai/openai/gpt-oss-20b`).

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
