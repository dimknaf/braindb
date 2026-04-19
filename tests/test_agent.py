"""
Agent endpoint smoke tests.

The agent goes through LiteLLM → an external LLM provider (DeepInfra / NIM).
External calls are flaky and slow. So the assertions here are intentionally
loose: the endpoint returns a well-formed response for a trivial prompt,
without asserting anything about the content of the answer.

If you're running this offline or the provider is down, these tests will time
out. They're scoped with a generous timeout but should be skipped in that
case (pytest -k 'not agent' to exclude them).
"""
import pytest
import requests


# Agent calls can take up to ~60s on a cold start; pytest-timeout guards us.
AGENT_TIMEOUT = 90


@pytest.mark.timeout(AGENT_TIMEOUT + 10)
def test_agent_query_smoke(api):
    """Trivial prompt, small max_turns. Must return 200 with an 'answer' field."""
    r = requests.post(
        f"{api}/api/v1/agent/query",
        json={"query": "Respond with exactly the word: ok.", "max_turns": 2},
        timeout=AGENT_TIMEOUT,
    )
    assert r.status_code == 200, f"agent failed: {r.status_code} {r.text[:300]}"
    body = r.json()
    assert "answer" in body, f"response missing 'answer' field: {body}"
    assert isinstance(body["answer"], str)
    assert "max_turns" in body


def test_agent_endpoint_rejects_empty_query(api):
    r = requests.post(
        f"{api}/api/v1/agent/query",
        json={"query": ""},
        timeout=10,
    )
    assert r.status_code in (400, 422), f"expected 4xx for empty query, got {r.status_code}"


def test_agent_endpoint_rejects_missing_query(api):
    r = requests.post(
        f"{api}/api/v1/agent/query",
        json={},
        timeout=10,
    )
    assert r.status_code in (400, 422), f"expected 4xx for missing query, got {r.status_code}"
