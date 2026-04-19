"""
Search, context (single + multi-query), and graph traversal coverage.
"""
import requests


def test_search_finds_created_fact(api, test_tag, make_fact):
    make_fact("The Aardvark is a medium-sized, nocturnal mammal native to Africa.")
    r = requests.post(
        f"{api}/api/v1/memory/search",
        json={"query": f"Aardvark mammal Africa {test_tag}", "limit": 10},
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    items = body if isinstance(body, list) else (body.get("items") or body.get("results") or [])
    assert isinstance(items, list)
    # Distinctive term "Aardvark" should surface at least one hit
    contents = " ".join(str(x.get("content", "")) for x in items)
    assert "Aardvark" in contents


def test_context_single_query_returns_structured_response(api, test_tag, make_fact):
    make_fact("Bismuth has the chemical symbol Bi and atomic number 83.")
    r = requests.post(
        f"{api}/api/v1/memory/context",
        json={"query": f"Bismuth chemistry {test_tag}", "max_depth": 3, "max_results": 10},
        timeout=20,
    )
    assert r.status_code == 200
    body = r.json()
    # The context endpoint returns both ranked items and always_on_rules
    assert "items" in body or "results" in body
    assert "always_on_rules" in body


def test_context_multi_query_merges_seeds(api, test_tag, make_fact):
    make_fact(f"Pytest marker entry one about Xerophyte plants {test_tag}.")
    make_fact(f"Pytest marker entry two about Yellowfin tuna {test_tag}.")
    r = requests.post(
        f"{api}/api/v1/memory/context",
        json={
            "queries": [f"Xerophyte {test_tag}", f"Yellowfin {test_tag}"],
            "max_depth": 2,
            "max_results": 20,
        },
        timeout=20,
    )
    assert r.status_code == 200
    items = r.json().get("items") or r.json().get("results") or []
    contents = " ".join(str(x.get("content", "")) for x in items)
    assert "Xerophyte" in contents and "Yellowfin" in contents


def test_graph_traversal_surfaces_connected_entity(api, test_tag, make_fact, make_relation):
    # A direct fact contains a distinctive term; B is connected to A but doesn't
    # contain the term itself. Context with max_depth >= 1 should surface B via A.
    seed_token = f"ZephyrMarker{test_tag[-4:]}"
    a = make_fact(f"Direct fact mentioning {seed_token} for search.")
    b = make_fact("Secondary fact with no distinctive term, linked to A.")
    make_relation(a["id"], b["id"], "elaborates")

    r = requests.post(
        f"{api}/api/v1/memory/context",
        json={"query": seed_token, "max_depth": 3, "max_results": 20},
        timeout=20,
    )
    assert r.status_code == 200
    items = r.json().get("items") or r.json().get("results") or []
    ids = [x.get("id") for x in items]
    # A must appear (direct match); B should appear too (graph-expanded)
    assert a["id"] in ids, "direct match not found"
    assert b["id"] in ids, "graph-connected entity not surfaced through traversal"


def test_tree_endpoint_returns_structure(api, make_fact, make_relation):
    root = make_fact("Root node of a small tree for test.")
    child1 = make_fact("Child 1 of the test tree.")
    child2 = make_fact("Child 2 of the test tree.")
    make_relation(root["id"], child1["id"], "elaborates")
    make_relation(root["id"], child2["id"], "elaborates")

    r = requests.get(
        f"{api}/api/v1/memory/tree/{root['id']}",
        params={"max_depth": 2},
        timeout=15,
    )
    assert r.status_code == 200
    # Tree structure is an opaque but non-empty payload
    body = r.json()
    assert body   # not empty or None


def test_stats_endpoint_returns_counts(api):
    r = requests.get(f"{api}/api/v1/memory/stats", timeout=10)
    assert r.status_code == 200
    body = r.json()
    # Stats must include at least a total count or per-type breakdown
    assert isinstance(body, dict)
    assert len(body) > 0
