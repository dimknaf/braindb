"""
Shared pytest fixtures for BrainDB integration tests.

These tests run against a live, running BrainDB stack (`docker compose up -d`).
They don't mock the API or the DB — they exercise the real HTTP endpoints and
real PostgreSQL. Each test self-registers the entity IDs it creates so a
session-scoped teardown can delete exactly those, leaving your real data
untouched.

Requirements to run the suite:
  - API reachable at http://localhost:8000 (override with BRAINDB_TEST_URL)
  - A healthy stack (/health returns 200)

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


API_URL = os.getenv("BRAINDB_TEST_URL", "http://localhost:8000")


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


@pytest.fixture(scope="session", autouse=True)
def _require_live_api() -> None:
    """Fail fast and loud if the stack isn't up — tests have nothing to run against."""
    if not _wait_for_health(API_URL):
        pytest.fail(
            f"BrainDB API not healthy at {API_URL}. "
            "Run `docker compose up -d` from the repo root first."
        )


@pytest.fixture
def api() -> str:
    """Base URL for the API — tests append paths like f'{api}/api/v1/...'."""
    return API_URL


@pytest.fixture
def test_tag() -> str:
    """Short unique marker to embed in test-created entities so we can filter
    them in queries without mistakenly touching real data. Unique per test.
    """
    return f"_pytest_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def created_entities() -> Iterator[list[str]]:
    """Collector the test appends entity IDs to. Everything in it gets deleted
    at teardown. Ignore 404s (already cleaned up).
    """
    ids: list[str] = []
    yield ids
    for eid in ids:
        try:
            requests.delete(f"{API_URL}/api/v1/entities/{eid}", timeout=5)
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
