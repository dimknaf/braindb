"""
Ingest-path behaviors that aren't about the watcher specifically:

- /datasources/ingest is idempotent by content_hash (same bytes → 200, not 201)

The datasource content-guardrail lives in the agent's update_entity tool
(braindb/agent/tools.py), not in the REST PATCH endpoint — direct PATCH on a
datasource IS allowed (operator use). Because testing the agent-side guardrail
requires an actual LLM loop (flaky, slow, expensive), that path isn't tested
in the automated suite. The behavior was manually verified during Phase A
end-to-end ingestion: the Smart Sand article's 10,357-char content was
preserved across three full runs of the watcher's chunked pipeline.
"""
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = REPO_ROOT / "data" / "sources" / "pytest_ingest_sample.md"
TEST_FILE_RELATIVE = "data/sources/pytest_ingest_sample.md"


def _write_sample(text: str) -> None:
    TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEST_FILE.write_text(text, encoding="utf-8")


def _cleanup_file() -> None:
    for p in [
        TEST_FILE,
        REPO_ROOT / "data" / "sources" / "ingested" / "pytest_ingest_sample.md",
        REPO_ROOT / "data" / "sources" / "failed" / "pytest_ingest_sample.md",
        REPO_ROOT / "data" / "sources" / "failed" / "pytest_ingest_sample.md.error.txt",
    ]:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass


def _find_datasource_by_title(api: str, title: str) -> dict | None:
    r = requests.get(
        f"{api}/api/v1/entities",
        params={"entity_type": "datasource", "limit": 200},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    for d in r.json():
        if d.get("title") == title:
            return d
    return None


def test_ingest_new_returns_201(api, created_entities):
    """First ingest of a fresh file returns 201."""
    content = "A unique pytest ingest-test body. " * 10
    _write_sample(content)
    try:
        r = requests.post(
            f"{api}/api/v1/entities/datasources/ingest",
            json={
                "file_path": TEST_FILE_RELATIVE,
                "keywords": ["_pytest_ingest"],
                "importance": 0.5,
                "source": "document",
            },
            timeout=30,
        )
        assert r.status_code == 201, f"expected 201, got {r.status_code}: {r.text[:200]}"
        ent = r.json()
        created_entities.append(ent["id"])
        assert ent["entity_type"] == "datasource"
        assert ent["content"] == content
    finally:
        _cleanup_file()


def test_ingest_duplicate_returns_200(api, created_entities):
    """Second ingest with the same bytes returns 200 (idempotent)."""
    content = "Idempotency pytest body. " * 15
    _write_sample(content)
    try:
        # First call — 201
        r = requests.post(
            f"{api}/api/v1/entities/datasources/ingest",
            json={
                "file_path": TEST_FILE_RELATIVE,
                "keywords": ["_pytest_dup"],
                "importance": 0.5,
                "source": "document",
            },
            timeout=30,
        )
        assert r.status_code == 201
        first_id = r.json()["id"]
        created_entities.append(first_id)

        # Second call — same bytes, should return 200 with the existing id
        r2 = requests.post(
            f"{api}/api/v1/entities/datasources/ingest",
            json={
                "file_path": TEST_FILE_RELATIVE,
                "keywords": ["_pytest_dup"],
                "importance": 0.5,
                "source": "document",
            },
            timeout=30,
        )
        assert r2.status_code == 200, f"expected 200 on dup, got {r2.status_code}: {r2.text[:200]}"
        assert r2.json()["id"] == first_id
    finally:
        _cleanup_file()


def test_ingest_dup_preserves_first_seen_metadata(api, created_entities):
    """When a dup fires, the returned entity is the ORIGINAL one (not a new one
    with new metadata). A second ingest with different keywords must not
    overwrite the first-seen keywords or swap the id.
    """
    content = "Dup-metadata pytest body. " * 20
    _write_sample(content)
    try:
        r1 = requests.post(
            f"{api}/api/v1/entities/datasources/ingest",
            json={
                "file_path": TEST_FILE_RELATIVE,
                "keywords": ["_pytest_first"],
                "importance": 0.5,
                "source": "document",
            },
            timeout=30,
        )
        assert r1.status_code == 201
        first = r1.json()
        created_entities.append(first["id"])

        r2 = requests.post(
            f"{api}/api/v1/entities/datasources/ingest",
            json={
                "file_path": TEST_FILE_RELATIVE,
                "keywords": ["_pytest_second"],     # different keywords
                "importance": 0.9,                   # different importance
                "source": "third-party",             # different source
            },
            timeout=30,
        )
        # 200 dup path must return the original untouched
        assert r2.status_code == 200
        second = r2.json()
        assert second["id"] == first["id"]
        # The dup response reflects what's stored (the original), not what
        # the second call passed. Confirm importance wasn't silently mutated.
        r3 = requests.get(f"{api}/api/v1/entities/{first['id']}", timeout=10)
        assert r3.json()["importance"] == 0.5
    finally:
        _cleanup_file()
