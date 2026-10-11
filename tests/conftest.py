"""
Shared pytest fixtures for BrainDB integration tests.

The suite runs against the ISOLATED TEST STACK — its own API + its own
throwaway Postgres, never your personal BrainDB database. Start it first:

    docker compose -f docker-compose.test.yml up -d

then run `pytest`, and tear down (wiping all test data) with:

    docker compose -f docker-compose.test.yml down -v

The tests exercise the real HTTP endpoints and real PostgreSQL — no mocks.
Each test registers the entity IDs it creates and deletes them at teardown,
but because the whole database is disposable, nothing depends on cleanup
being perfect.

Nothing here touches the agent's LLM backend; tests that hit /agent/query
send trivial prompts and don't rely on any specific model.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Callable, Iterator

import pytest
import requests


# The isolated test stack (docker-compose.test.yml). Override only if you
# really mean to point the suite somewhere else.
API_URL = os.getenv("BRAINDB_TEST_URL", "http://localhost:8002")

# A handful of tests drive internal services (e.g. graph_expand) over a
# direct DB connection. Default it to the test stack's Postgres, published
# on the host at 5436. An explicit DATABASE_URL in the environment wins.
os.environ.setdefault(
    "DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test"
)


def _wait_for_health(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="session")
def _require_live_api() -> None:
    """Fail fast and loud if the test stack isn't up.

    Attached to the `api` fixture (NOT autouse) so pure unit tests — the
    validator/handoff/chunking files that never touch HTTP — run with no
    stack at all (that's also what CI runs). Deliberately NOT defaulting
    to the personal stack on :8000 — the suite must never run against a
    database holding real data.
    """
    if not _wait_for_health(API_URL):
        pytest.fail(
            f"BrainDB test API not healthy at {API_URL}. Start the isolated "
            "test stack first:\n"
            "    docker compose -f docker-compose.test.yml up -d\n"
            "and tear it down (wiping all test data) with:\n"
            "    docker compose -f docker-compose.test.yml down -v"
        )


@pytest.fixture
def api(_require_live_api: None) -> str:
    """Base URL for the API — tests append paths like f'{api}/api/v1/...'."""
    return API_URL


@pytest.fixture
def test_tag() -> str:
    """Short unique marker to embed in test-created entities so we can filter
    them in queries without mistakenly touching real data. Unique per test.
    """
    return f"_pytest_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def created_entities(api: str) -> Iterator[list[str]]:
    """Collector the test appends entity IDs to. Everything in it gets deleted
    at teardown. Ignore 404s (already cleaned up).
    """
    ids: list[str] = []
    yield ids
    for eid in ids:
        try:
            requests.delete(f"{api}/api/v1/entities/{eid}", timeout=5)
        except requests.RequestException:
            pass


@pytest.fixture
def make_fact(api: str, test_tag: str, created_entities: list[str]) -> Callable[..., dict]:
    """Factory that POSTs a fact and registers it for cleanup. Returns the entity dict."""
    def _make(content: str, keywords: list[str] | None = None, certainty: float = 0.8, importance: float = 0.5) -> dict:
        body = {
            "content": content,
            "certainty": certainty,
            "source": "user-stated",
            "keywords": (keywords or []) + [test_tag],
            "importance": importance,
        }
        r = requests.post(f"{api}/api/v1/entities/facts", json=body, timeout=30)
        assert r.status_code == 201, f"create fact failed: {r.status_code} {r.text}"
        ent = r.json()
        created_entities.append(ent["id"])
        return ent
    return _make


@pytest.fixture
def make_thought(api: str, test_tag: str, created_entities: list[str]) -> Callable[..., dict]:
    def _make(content: str, certainty: float = 0.6, context: str | None = None, importance: float = 0.4) -> dict:
        body = {
            "content": content,
            "certainty": certainty,
            "source": "agent-inference",
            "context": context,
            "keywords": [test_tag],
            "importance": importance,
        }
        r = requests.post(f"{api}/api/v1/entities/thoughts", json=body, timeout=30)
        assert r.status_code == 201, f"create thought failed: {r.status_code} {r.text}"
        ent = r.json()
        created_entities.append(ent["id"])
        return ent
    return _make


@pytest.fixture
def make_source(api: str, test_tag: str, created_entities: list[str]) -> Callable[..., dict]:
    def _make(content: str, url: str = "https://example.test/doc", title: str | None = None) -> dict:
        body = {
            "content": content,
            "title": title or "Test source",
            "url": url,
            "domain": "example.test",
            "keywords": [test_tag],
            "importance": 0.5,
            "source": "third-party",
        }
        r = requests.post(f"{api}/api/v1/entities/sources", json=body, timeout=30)
        assert r.status_code == 201, f"create source failed: {r.status_code} {r.text}"
        ent = r.json()
        created_entities.append(ent["id"])
        return ent
    return _make


@pytest.fixture
def make_datasource(api: str, test_tag: str, created_entities: list[str]) -> Callable[..., dict]:
    """Creates a datasource via the JSON endpoint (not ingest-from-file)."""
    def _make(content: str, title: str = "Test datasource") -> dict:
        body = {
            "content": content,
            "title": title,
            "url": f"pytest://{test_tag}/{title}",   # schema requires file_path OR url
            "keywords": [test_tag],
            "importance": 0.6,
            "source": "document",
        }
        r = requests.post(f"{api}/api/v1/entities/datasources", json=body, timeout=30)
        assert r.status_code == 201, f"create datasource failed: {r.status_code} {r.text}"
        ent = r.json()
        created_entities.append(ent["id"])
        return ent
    return _make


@pytest.fixture
def make_rule(api: str, test_tag: str, created_entities: list[str]) -> Callable[..., dict]:
    def _make(content: str, always_on: bool = False, priority: int = 50, category: str = "behavior") -> dict:
        body = {
            "content": content,
            "always_on": always_on,
            "category": category,
            "priority": priority,
            "importance": 0.7,
            "keywords": [test_tag],
            "source": "user-stated",
        }
        r = requests.post(f"{api}/api/v1/entities/rules", json=body, timeout=30)
        assert r.status_code == 201, f"create rule failed: {r.status_code} {r.text}"
        ent = r.json()
        created_entities.append(ent["id"])
        return ent
    return _make


@pytest.fixture
def make_relation(api: str) -> Callable[..., dict]:
    """Factory for creating a relation. Relations are cascade-deleted with their
    endpoint entities, so no explicit cleanup needed as long as the entity
    teardown fixture runs.
    """
    def _make(from_id: str, to_id: str, relation_type: str = "supports", relevance: float = 0.8, description: str | None = None) -> dict:
        body = {
            "from_entity_id": from_id,
            "to_entity_id": to_id,
            "relation_type": relation_type,
            "relevance_score": relevance,
            "description": description,
        }
        r = requests.post(f"{api}/api/v1/relations", json=body, timeout=30)
        assert r.status_code == 201, f"create relation failed: {r.status_code} {r.text}"
        return r.json()
    return _make


# ---------------------------------------------------------------------------
# JEV's model server, faked: no test ever reaches a real one
# ---------------------------------------------------------------------------

class FakeChat:
    """An OpenAI-compatible chat completions endpoint for the JEV tests, put
    behind the real openai client (a MockTransport of httpx2, the HTTP
    library the openai client uses), so every request
    is built and every reply read exactly as in production.

    Each request's JSON body is kept in `bodies`. `reply(body)` says what the
    model writes: a list of tokens, each `(text, {token: probability})` for a
    token with alternatives (the chosen one is `text`) or `(text, None)`; or
    an HTTP status for that request alone (`asked(body)` gives the question
    line and the content it is asked about, in either of jev's forms).
    `status` answers every request with that status; `alternatives = False`
    returns only the chosen token's logprob (as some gateways do);
    `continues = False` makes it a server that does not continue a reply the
    request opens (it starts a new turn, as a hosted gateway did), so jev
    asks it in the standard form. jev's own check of that (`Server.mode`) is
    answered here. `max_tokens` and `stop` apply as a server applies them;
    usage counts 4 chars of the request a token, and caches what it shares
    with the request before."""

    def __init__(self, model: str, reply):
        import threading

        self.model, self.reply = model, reply
        self.bodies: list[dict] = []
        self.status: int | None = None
        self.alternatives = True
        self.continues = True
        self._lock = threading.Lock()
        self._last = ""

    @staticmethod
    def asked(body: dict) -> tuple[str, str]:
        """The question line a JEV request asks, and the content it asks it
        about: the reply's opening and the message before it (continue form),
        or the last line of the last message and the rest of it, or the
        message before when the question is its own turn (standard form)."""
        msgs = body["messages"]
        last = msgs[-1]
        if last["role"] == "assistant":
            return last["content"], msgs[-2]["content"]
        lines = (last["content"] or "").rstrip("\n").split("\n")
        if last["role"] == "user" and len(msgs) > 1 and msgs[-2]["role"] == "user":
            return lines[-1], msgs[-2]["content"]
        return lines[-1], "\n".join(lines[:-1])

    def _check(self, body: dict):
        """jev's check that a reply the request opens is continued."""
        from braindb.services import jev

        if not self.continues:
            return [("\n", None)]                   # a new turn begins: no answer
        want = "yes" if jev._PROBE_YES in body["messages"][-2]["content"] else "no"
        return [(f" {want}", {f" {want}": 0.98, f" {'no' if want == 'yes' else 'yes'}": 0.01})]

    def handle(self, request):
        import json as _json
        import math

        import httpx2

        from braindb.services import jev

        body = _json.loads(request.content)
        with self._lock:
            self.bodies.append(body)
        status = self.status or None
        if not status and body.get("continue_final_message") and body["messages"][0]["content"] == jev._GUIDE:
            tokens = self._check(body)
        elif not status and body.get("continue_final_message") and not self.continues:
            tokens = [("\n", None)]
        else:
            tokens = None if status else self.reply(body)
        if isinstance(tokens, int):
            status = tokens
        if status:
            return httpx2.Response(status, json={"error": {"message": "refused by the fake", "type": "test"}})
        out, finish = [], "stop"
        for text, alts in tokens:
            if len(out) >= body.get("max_tokens", 16):
                finish = "length"
                break
            if any(s and s in text for s in body.get("stop") or []):
                break
            out.append((text, alts))
        lp = []
        for text, alts in out:
            alts = alts or {text: 1.0}
            p = alts.get(text, 1.0)
            lp.append({"token": text, "logprob": math.log(p) if p > 0 else -9999.0, "bytes": None,
                       "top_logprobs": [{"token": t, "logprob": math.log(q) if q > 0 else -9999.0, "bytes": None}
                                        for t, q in alts.items()] if self.alternatives else []})
        sent = _json.dumps(body.get("messages")) + _json.dumps(body.get("tools") or [])
        with self._lock:
            shared = next((i for i, (a, b) in enumerate(zip(sent, self._last)) if a != b), min(len(sent), len(self._last)))
            self._last = sent
        payload = {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": self.model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": "".join(t for t, _ in out)},
                         "logprobs": {"content": lp} if body.get("logprobs") else None}],
            "usage": {"prompt_tokens": len(sent) // 4, "completion_tokens": len(out),
                      "total_tokens": len(sent) // 4 + len(out),
                      "prompt_tokens_details": {"cached_tokens": shared // 4}}}
        # NaN must survive into the reply (a broken forward pass): json.dumps writes it, httpx2's json= would not.
        return httpx2.Response(200, content=_json.dumps(payload).encode(),
                              headers={"content-type": "application/json"})


def _serve(monkeypatch, handler) -> None:
    """Every openai client JEV builds talks to `handler` instead of a network."""
    import httpx2
    import openai

    from braindb.services import jev

    monkeypatch.setattr(jev, "OpenAI", lambda **kw: openai.OpenAI(
        **kw, http_client=httpx2.Client(transport=httpx2.MockTransport(handler))))


@pytest.fixture(autouse=True)
def _no_model_server(monkeypatch):
    """JEV's features are on by default and ask the agent profile's server on
    first use: no test may reach a real one, and none may inherit another's
    JEV state. Unless a test installs a fake (`fake_chat`), every JEV request
    is refused as unauthorised, so JEV stays off, as on a server it cannot use."""
    try:
        import httpx2

        from braindb.services import hops, jev, reader, wiki_gate, wiki_triage
    except ImportError:  # an environment without the package: integration tests only
        yield
        return
    _serve(monkeypatch, lambda request: httpx2.Response(401, json={"error": {"message": "no model server in tests"}}))
    monkeypatch.setattr(jev, "_judge", None)
    monkeypatch.setattr(jev, "_off_until", 0.0)
    monkeypatch.setattr(jev, "_modes", {})
    monkeypatch.setattr(reader, "_state", {"server": None, "off_until": 0.0})
    monkeypatch.setattr(hops, "_state", {"off_until": 0.0})
    monkeypatch.setattr(wiki_gate, "_state", {"server": None, "off_until": 0.0})
    monkeypatch.setattr(wiki_triage, "_state", {"choice": None, "options": None, "off_until": 0.0})
    yield


@pytest.fixture
def fake_chat(monkeypatch):
    """Factory: `fake_chat(reply)` serves every JEV request from a FakeChat
    answering with `reply` (see FakeChat), on the agent profile's model."""
    from braindb.services import jev

    def install(reply) -> FakeChat:
        fake = FakeChat(jev.settings.resolved_agent_model, reply)
        _serve(monkeypatch, fake.handle)
        return fake

    return install
