"""Targeted tests for agent tool ID validation."""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import contextmanager

from agents.tool import ToolContext

from braindb.agent import tools


def _invoke_tool(tool, args: dict) -> str:
    payload = json.dumps(args)
    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="pytest",
        tool_arguments=payload,
    )
    return asyncio.run(tool.on_invoke_tool(ctx, payload))


def test_get_entity_rejects_placeholder_before_db(monkeypatch):
    def fail_get_conn():
        raise AssertionError("database should not be queried for placeholder IDs")

    monkeypatch.setattr(tools, "get_conn", fail_get_conn)

    result = _invoke_tool(tools.get_entity, {"entity_id": "<root-entity-id>"})

    assert result.startswith("ERROR:")
    assert "placeholder" in result


def test_view_log_rejects_non_uuid_filter_before_db(monkeypatch):
    def fail_get_conn():
        raise AssertionError("database should not be queried for invalid IDs")

    monkeypatch.setattr(tools, "get_conn", fail_get_conn)

    result = _invoke_tool(tools.view_log, {
        "operation": None,
        "entity_id": "search-mode-context",
        "limit": 30,
    })

    assert result.startswith("ERROR:")
    assert "placeholder" in result


def test_create_relation_rejects_invalid_type_before_db(monkeypatch):
    def fail_get_conn():
        raise AssertionError("database should not be queried for invalid relation types")

    monkeypatch.setattr(tools, "get_conn", fail_get_conn)

    result = _invoke_tool(tools.create_relation, {
        "from_entity_id": str(uuid.uuid4()),
        "to_entity_id": str(uuid.uuid4()),
        "relation_type": "i_like_it",
        "relevance_score": 0.7,
        "description": None,
    })

    assert result.startswith("ERROR:")
    assert "relation_type" in result


def test_create_relation_prechecks_missing_entities_before_insert(monkeypatch):
    queries: list[str] = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, query, params=None):
            queries.append(query)

        def fetchall(self):
            return []

    class FakeConn:
        def cursor(self, *args, **kwargs):
            return FakeCursor()

    @contextmanager
    def fake_get_conn():
        yield FakeConn()

    monkeypatch.setattr(tools, "get_conn", fake_get_conn)
    missing_from = str(uuid.uuid4())

    result = _invoke_tool(tools.create_relation, {
        "from_entity_id": missing_from,
        "to_entity_id": str(uuid.uuid4()),
        "relation_type": "supports",
        "relevance_score": 0.7,
        "description": None,
    })

    assert result == f"ERROR: entity {missing_from} not found"
    assert len(queries) == 1
    assert "SELECT id FROM entities" in queries[0]
