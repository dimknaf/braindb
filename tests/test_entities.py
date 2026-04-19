"""
CRUD round-trip tests for all 5 entity types: fact, thought, source, datasource, rule.

Verifies: create returns 201 with the right type, get returns the same fields,
patch updates only what you sent, delete returns 204 and subsequent get is 404.
"""
import requests


def test_fact_full_roundtrip(api, make_fact):
    fact = make_fact("The Moon has approximately one-sixth Earth's gravity.", keywords=["astronomy"], certainty=0.95, importance=0.6)
    assert fact["entity_type"] == "fact"
    assert fact["certainty"] == 0.95
    assert fact["importance"] == 0.6
    assert "astronomy" in fact["keywords"]

    # GET by id returns the same
    r = requests.get(f"{api}/api/v1/entities/{fact['id']}", timeout=10)
    assert r.status_code == 200
    got = r.json()
    assert got["id"] == fact["id"]
    assert got["content"] == fact["content"]

    # PATCH — update only notes
    r = requests.patch(
        f"{api}/api/v1/entities/facts/{fact['id']}",
        json={"notes": "Source: NASA fact sheet."},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json()["notes"] == "Source: NASA fact sheet."

    # PATCH left certainty untouched
    r = requests.get(f"{api}/api/v1/entities/{fact['id']}", timeout=10)
    assert r.json()["certainty"] == 0.95

    # DELETE
    r = requests.delete(f"{api}/api/v1/entities/{fact['id']}", timeout=10)
    assert r.status_code == 204

    # GET after delete is 404
    r = requests.get(f"{api}/api/v1/entities/{fact['id']}", timeout=10)
    assert r.status_code == 404


def test_thought_create_and_fields(api, make_thought):
    thought = make_thought(
        "I suspect the user prefers terse answers over verbose ones.",
        certainty=0.7,
        context="Inferred from multiple short replies.",
    )
    assert thought["entity_type"] == "thought"
    assert thought["certainty"] == 0.7
    # Thought has a context field distinct from content
    r = requests.get(f"{api}/api/v1/entities/{thought['id']}", timeout=10)
    assert r.json().get("context", "").startswith("Inferred")


def test_source_has_url_and_domain(api, make_source):
    src = make_source(
        "Reference article on X.",
        url="https://example.test/article-42",
        title="Article 42",
    )
    assert src["entity_type"] == "source"
    assert src["url"] == "https://example.test/article-42"
    assert src["domain"] == "example.test"


def test_datasource_create_has_content(api, make_datasource):
    body = "This is a small embedded document body. " * 20   # ~160 words
    ds = make_datasource(body, title="Test DS")
    assert ds["entity_type"] == "datasource"
    assert ds["content"] == body
    # PATCH on a datasource does NOT touch content (covered in test_ingest.py guardrail),
    # but patching notes should work
    r = requests.patch(
        f"{api}/api/v1/entities/datasources/{ds['id']}",
        json={"notes": "Added by tests"},
        timeout=10,
    )
    assert r.status_code == 200
    assert r.json()["notes"] == "Added by tests"


def test_rule_create_and_always_on(api, make_rule):
    rule = make_rule("Always favor clarity over cleverness in code.", always_on=True, priority=90, category="behavior")
    assert rule["entity_type"] == "rule"
    assert rule["always_on"] is True
    assert rule["priority"] == 90
    # Always-on rules appear in /memory/rules
    r = requests.get(f"{api}/api/v1/memory/rules", timeout=10)
    assert r.status_code == 200
    ids = [x["id"] for x in r.json()]
    assert rule["id"] in ids


def test_list_filter_by_type(api, test_tag, make_fact, make_thought):
    make_fact("A test fact for listing.")
    make_thought("A test thought for listing.")
    r = requests.get(
        f"{api}/api/v1/entities",
        params={"entity_type": "fact", "keyword": test_tag, "limit": 10},
        timeout=10,
    )
    assert r.status_code == 200
    types = {e["entity_type"] for e in r.json()}
    # Only facts come back because of the filter
    assert types == {"fact"} or types == set()


def test_list_filter_by_keyword(api, test_tag, make_fact):
    make_fact("Fact alpha.")
    make_fact("Fact beta.")
    r = requests.get(
        f"{api}/api/v1/entities",
        params={"keyword": test_tag, "limit": 20},
        timeout=10,
    )
    # Both should be findable by the pytest tag we attached to them
    contents = [e["content"] for e in r.json() if e.get("content")]
    assert any("alpha" in c for c in contents)
    assert any("beta" in c for c in contents)


def test_get_nonexistent_entity_is_404(api):
    r = requests.get(f"{api}/api/v1/entities/00000000-0000-0000-0000-000000000000", timeout=10)
    assert r.status_code == 404


def test_delete_is_idempotent_on_missing(api):
    # Deleting a non-existent entity should not 500; 404 is acceptable
    r = requests.delete(
        f"{api}/api/v1/entities/00000000-0000-0000-0000-000000000000",
        timeout=10,
    )
    assert r.status_code in (204, 404)
