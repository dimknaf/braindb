"""graph_expand against the path-enumerating query it replaced, on the real
database.

The old recursive CTE listed every path from the seeds and kept, per entity,
the row with the smallest depth and the best relevance. graph_expand keeps
one state per (entity, depth) instead. On a graph with a hub, a cycle,
parallel relations, a NULL score, a self-loop, a tie between seeds and an
entity whose best path runs through an entity at a longer depth, both must
return the same rows, in the same order, with the same values. Where seeds
tie, the old query is given the tie-break graph_expand uses (smallest seed,
then smallest relation id); its own choice there was arbitrary.

Each test runs inside one transaction that is rolled back at teardown.
"""
from __future__ import annotations

import os
import uuid

import psycopg2
import psycopg2.extras
import pytest

from braindb.services.graph import graph_expand

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://braindb:braindb@localhost:5436/braindb_test"
)

# The query graph_expand replaced, with a deterministic tie-break added.
_PATHS_SQL = """
WITH RECURSIVE traversal AS (
    SELECT
        e.id, e.entity_type, e.title, e.content, e.summary,
        e.keywords, e.importance, e.source, e.notes,
        e.created_at, e.updated_at, e.accessed_at, e.access_count, e.metadata,
        0                   AS depth,
        1.0::FLOAT          AS accumulated_relevance,
        ARRAY[e.id]         AS visited,
        NULL::UUID          AS via_relation_id,
        NULL::TEXT          AS via_relation_type,
        NULL::TEXT          AS via_description,
        NULL::TEXT          AS via_notes,
        e.id                AS seed_origin_id
    FROM entities e
    WHERE e.id = ANY(%s::uuid[])

    UNION ALL

    SELECT
        target.id, target.entity_type, target.title, target.content, target.summary,
        target.keywords, target.importance, target.source, target.notes,
        target.created_at, target.updated_at, target.accessed_at, target.access_count, target.metadata,
        t.depth + 1,
        (
            t.accumulated_relevance
            * r.relevance_score
            * COALESCE(r.importance_score, 0.5)
        )::FLOAT,
        t.visited || target.id,
        r.id,
        r.relation_type,
        r.description,
        r.notes,
        t.seed_origin_id
    FROM traversal t
    JOIN relations r ON r.from_entity_id = t.id OR r.to_entity_id = t.id
    JOIN entities target ON
        target.id = CASE
            WHEN r.from_entity_id = t.id THEN r.to_entity_id
            ELSE r.from_entity_id
        END
    WHERE t.depth < %s
      AND NOT (target.id = ANY(t.visited))
      AND (
            t.accumulated_relevance
            * r.relevance_score
            * COALESCE(r.importance_score, 0.5)
          ) > %s
)
SELECT DISTINCT ON (id)
    id, entity_type, title, content, summary,
    keywords, importance, source, notes,
    created_at, updated_at, accessed_at, access_count, metadata,
    depth           AS min_depth,
    accumulated_relevance AS relevance,
    via_relation_id, via_relation_type, via_description, via_notes,
    seed_origin_id
FROM traversal
ORDER BY id, depth, accumulated_relevance DESC, seed_origin_id, via_relation_id
"""


@pytest.fixture
def conn():
    c = psycopg2.connect(DB_URL)
    yield c
    c.rollback()
    c.close()


def _paths(conn, seeds, max_depth, min_relevance):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_PATHS_SQL, (seeds, max_depth, min_relevance))
        return [dict(r) for r in cur.fetchall()]


@pytest.fixture
def graph(conn):
    """Facts f0-f12 and keywords k0-k2, wired to cover every case above."""
    token = f"zqgraph{uuid.uuid4().hex[:10]}"
    ids = {}
    with conn.cursor() as cur:
        for i in range(13):
            ids[f"f{i}"] = str(uuid.uuid4())
            cur.execute("""INSERT INTO entities (id, entity_type, content, keywords, source, importance)
                           VALUES (%s, 'fact', %s, %s, 'user-stated', 0.6)""",
                        (ids[f"f{i}"], f"{token} fact {i}", [token]))
        for i in range(3):
            ids[f"k{i}"] = str(uuid.uuid4())
            cur.execute("""INSERT INTO entities (id, entity_type, content, importance)
                           VALUES (%s, 'keyword', %s, 0.5)""", (ids[f"k{i}"], f"{token}-kw{i}"))

        def rel(a, b, kind, relevance, importance):
            cur.execute("""INSERT INTO relations (from_entity_id, to_entity_id, relation_type,
                                                  relevance_score, importance_score, description)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (ids[a], ids[b], kind, relevance, importance, f"{a}-{kind}-{b}"))

        for f in ("f0", "f1", "f2", "f3", "f4", "f8"):          # k0: a hub
            rel(f, "k0", "tagged_with", 0.8, 0.5)
        for f in ("f0", "f9"):
            rel(f, "k1", "tagged_with", 0.8, 0.5)
        for f in ("f9", "f10"):                                 # two seeds tie on k2
            rel(f, "k2", "tagged_with", 0.8, 0.5)
        rel("f0", "f1", "supports", 0.9, 0.9)                   # a cycle
        rel("f1", "f2", "supports", 0.9, 0.9)
        rel("f2", "f0", "supports", 0.9, 0.9)
        rel("f3", "f4", "supports", 0.5, 0.5)                   # parallel relations,
        rel("f3", "f4", "elaborates", 0.9, None)                # one importance unset
        rel("f5", "f6", "refers_to", 0.06, 1.0)                 # f7's best path runs
        rel("f5", "f11", "supports", 0.9, 0.9)                  # through f6 at depth 2,
        rel("f11", "f6", "supports", 0.9, 0.9)                  # not its depth 1
        rel("f6", "f7", "supports", 0.3, 1.0)
        rel("f8", "f9", "refers_to", None, 0.9)                 # a NULL score: never followed
        rel("f10", "f10", "refers_to", 0.9, 0.9)                # a self-loop
        rel("f12", "f11", "derived_from", 0.7, 0.7)             # reached against its direction
    return ids


@pytest.mark.parametrize("max_depth", [1, 2, 3])
@pytest.mark.parametrize("min_relevance", [0.05, 0.15])
def test_same_rows_as_the_path_query(conn, graph, max_depth, min_relevance):
    seed_sets = [["f0"], ["f5"], ["f9", "f10"], ["f8"], ["f0", "f5", "f9", "f12"],
                 [k for k in graph if k.startswith("f")]]
    for names in seed_sets:
        seeds = [graph[n] for n in names]
        expected = _paths(conn, seeds, max_depth, min_relevance)
        assert graph_expand(conn, seeds, max_depth, min_relevance) == expected, (names, max_depth)


def test_an_entity_reached_through_a_longer_path_gets_that_paths_depth(conn, graph):
    rows = {r["id"]: r for r in graph_expand(conn, [graph["f5"]], 3, 0.05)}
    f6, f7 = rows[graph["f6"]], rows[graph["f7"]]
    assert (f6["min_depth"], f6["relevance"]) == (1, 0.06)       # the weak direct relation
    assert f7["min_depth"] == 3 and f7["seed_origin_id"] == graph["f5"]
    assert f7["relevance"] == 1.0 * 0.9 * 0.9 * 0.9 * 0.9 * 0.3 * 1.0


def test_unknown_seeds_and_no_seeds(conn, graph):
    assert graph_expand(conn, [], 3, 0.05) == []
    stray = str(uuid.uuid4())
    assert graph_expand(conn, [stray, graph["f0"]], 2, 0.05) == _paths(conn, [stray, graph["f0"]], 2, 0.05)
