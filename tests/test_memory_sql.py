"""Targeted tests for read-only SQL execution and activity logging."""
from __future__ import annotations

import uuid

import requests


def _recent_sql_log_details(api: str, limit: int = 20) -> list[dict]:
    r = requests.get(f"{api}/api/v1/memory/log", params={"operation": "sql_query", "limit": limit}, timeout=10)
    assert r.status_code == 200, f"activity log query failed: {r.status_code} {r.text[:200]}"
    return [entry.get("details") or {} for entry in r.json()]


def test_read_only_sql_success_logs_in_separate_transaction(api):
    marker = f"pytest_sql_{uuid.uuid4().hex}"
    query = f"SELECT 1 AS ok /* {marker} */"

    r = requests.post(f"{api}/api/v1/memory/sql", json={"query": query}, timeout=10)

    assert r.status_code == 200, f"sql query failed: {r.status_code} {r.text[:200]}"
    body = r.json()
    assert body["columns"] == ["ok"]
    assert body["rows"] == [[1]]
    assert body["row_count"] == 1

    details = _recent_sql_log_details(api)
    assert any(d.get("query") == query and d.get("rows") == 1 for d in details)


def test_read_only_sql_failure_logs_in_separate_transaction(api):
    marker = f"pytest_sql_{uuid.uuid4().hex}"
    query = f"SELECT definitely_missing_column /* {marker} */"

    r = requests.post(f"{api}/api/v1/memory/sql", json={"query": query}, timeout=10)

    assert r.status_code == 400
    assert "Query error" in r.text

    details = _recent_sql_log_details(api)
    assert any(d.get("query") == query and "error" in d for d in details)
