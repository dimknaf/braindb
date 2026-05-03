"""Agent endpoint fast-path tests."""
import re

import requests


_ID_RE = re.compile(r"id=([0-9a-f-]{36})")


def _extract_entity_id(answer: str) -> str:
    match = _ID_RE.search(answer)
    assert match, f"missing entity id in answer: {answer!r}"
    return match.group(1)


def test_agent_fast_path_saves_fact(api, test_tag, created_entities):
    content = f"{test_tag} user prefers deterministic BrainDB save fast paths"
    response = requests.post(
        f"{api}/api/v1/agent/query",
        json={"query": f"Save: {content}"},
        timeout=10,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "fast_path"
    assert body["max_turns"] == 0
    entity_id = _extract_entity_id(body["answer"])
    created_entities.append(entity_id)

    entity = requests.get(f"{api}/api/v1/entities/{entity_id}", timeout=10)
    assert entity.status_code == 200, entity.text
    payload = entity.json()
    assert payload["entity_type"] == "fact"
    assert payload["content"] == content


def test_agent_fast_path_saves_rule(api, test_tag, created_entities):
    content = f"{test_tag} always prefer deterministic fast paths for simple memory saves"
    response = requests.post(
        f"{api}/api/v1/agent/query",
        json={"query": f"Save as rule: {content}"},
        timeout=10,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "fast_path"
    assert body["max_turns"] == 0
    entity_id = _extract_entity_id(body["answer"])
    created_entities.append(entity_id)

    entity = requests.get(f"{api}/api/v1/entities/{entity_id}", timeout=10)
    assert entity.status_code == 200, entity.text
    payload = entity.json()
    assert payload["entity_type"] == "rule"
    assert payload["content"] == content


def test_agent_save_question_bypasses_fast_path(api):
    response = requests.post(
        f"{api}/api/v1/agent/query",
        json={"query": "Save: what does the user prefer?", "max_turns": 1},
        timeout=80,
    )

    assert response.status_code == 200, response.text
    assert response.json().get("status") != "fast_path"


def test_agent_max_turns_returns_structured_status(api):
    response = requests.post(
        f"{api}/api/v1/agent/query",
        json={
            "query": "Recall everything about the user, then save a thought summarizing it, then connect them with relations.",
            "max_turns": 1,
        },
        timeout=80,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "max_turns_exceeded"
    assert body["max_turns"] == 1
    assert body["turns_used"] == 1
    assert body["answer"]
