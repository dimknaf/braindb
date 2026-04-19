"""
Relation CRUD + inbound/outbound listing + type validation.
"""
import requests


def test_create_relation_between_facts(api, make_fact, make_relation):
    a = make_fact("The sky appears blue due to Rayleigh scattering.")
    b = make_fact("Rayleigh scattering is inversely proportional to the fourth power of wavelength.")
    rel = make_relation(a["id"], b["id"], relation_type="elaborates", relevance=0.9, description="B explains the mechanism in A")
    assert rel["relation_type"] == "elaborates"
    assert rel["from_entity_id"] == a["id"]
    assert rel["to_entity_id"] == b["id"]
    assert 0.85 <= rel["relevance_score"] <= 0.95


def test_list_relations_returns_both_directions(api, make_fact, make_relation):
    a = make_fact("Fact A.")
    b = make_fact("Fact B.")
    c = make_fact("Fact C.")
    make_relation(a["id"], b["id"], "supports")
    make_relation(c["id"], a["id"], "contradicts")

    r = requests.get(f"{api}/api/v1/entities/{a['id']}/relations", timeout=10)
    assert r.status_code == 200
    rels = r.json()
    # A has one outbound (to B) and one inbound (from C)
    rel_by_type = {x["relation_type"]: x for x in rels}
    assert "supports" in rel_by_type and rel_by_type["supports"]["to_entity_id"] == b["id"]
    assert "contradicts" in rel_by_type and rel_by_type["contradicts"]["from_entity_id"] == c["id"]


def test_patch_relation_updates_fields(api, make_fact, make_relation):
    a = make_fact("Fact X.")
    b = make_fact("Fact Y.")
    rel = make_relation(a["id"], b["id"], "similar_to", relevance=0.5)
    r = requests.patch(
        f"{api}/api/v1/relations/{rel['id']}",
        json={"relevance_score": 0.95, "notes": "Closer on re-read"},
        timeout=10,
    )
    assert r.status_code == 200
    updated = r.json()
    assert updated["relevance_score"] == 0.95
    assert updated["notes"] == "Closer on re-read"


def test_delete_relation(api, make_fact, make_relation):
    a = make_fact("Fact one.")
    b = make_fact("Fact two.")
    rel = make_relation(a["id"], b["id"], "refers_to")
    r = requests.delete(f"{api}/api/v1/relations/{rel['id']}", timeout=10)
    assert r.status_code == 204
    # Subsequent GET is 404
    r = requests.get(f"{api}/api/v1/relations/{rel['id']}", timeout=10)
    assert r.status_code == 404


def test_deleting_entity_cascades_its_relations(api, make_fact, make_relation):
    a = make_fact("About to be deleted.")
    b = make_fact("Peer of the deleted one.")
    rel = make_relation(a["id"], b["id"], "supports")
    # Delete A
    r = requests.delete(f"{api}/api/v1/entities/{a['id']}", timeout=10)
    assert r.status_code == 204
    # Relation should no longer exist
    r = requests.get(f"{api}/api/v1/relations/{rel['id']}", timeout=10)
    assert r.status_code == 404
    # B should no longer have that relation in its list
    r = requests.get(f"{api}/api/v1/entities/{b['id']}/relations", timeout=10)
    ids = {x["id"] for x in r.json()}
    assert rel["id"] not in ids


def test_invalid_relation_type_is_rejected(api, make_fact):
    a = make_fact("A.")
    b = make_fact("B.")
    # 'supports' is valid; 'i_like_it' is not
    r = requests.post(
        f"{api}/api/v1/relations",
        json={
            "from_entity_id": a["id"],
            "to_entity_id": b["id"],
            "relation_type": "i_like_it",
            "relevance_score": 0.5,
        },
        timeout=10,
    )
    assert r.status_code in (400, 422), f"expected 4xx, got {r.status_code}: {r.text[:200]}"


def test_all_documented_relation_types_accepted(api, make_fact, make_relation):
    """Verify every relation_type documented in BRAINDB_GUIDE.md is accepted."""
    types = ["supports", "contradicts", "elaborates", "refers_to",
             "derived_from", "similar_to", "is_example_of", "challenges"]
    for t in types:
        a = make_fact(f"Source for {t}.")
        b = make_fact(f"Target for {t}.")
        # make_relation asserts 201 internally; if any type is rejected the test fails here
        make_relation(a["id"], b["id"], t)
