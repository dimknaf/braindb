"""Unit tests for JEV (braindb/services/jev.py).

DB-free and GPU-free: a fake vLLM endpoint stands in for the server. These
pin the contract every caller relies on:

- off unless enabled on a self-hosted endpoint, with no request made;
- each item is judged in order, and usage reaches the request's ledger once;
- a server that cannot do JEV (e.g. a gateway that drops the requested
  logprobs) keeps it off, and every failure fails OPEN (None);
- a refused or late item stays unjudged rather than failing the call;
- the judge reads stored text as data: capped, special tokens neutralised;
- logs never carry the key or the endpoint;
- an /agent/query `need` reaches the recalls of that request only, and only
  while JEV is enabled.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time

import pytest
import requests

import braindb.agent.tools as tools_mod
import braindb.routers.agent as agent_router
from braindb.agent.run_state import get_need, install_usage, release_usage, reset_need, set_need
from braindb.schemas.search import ContextResponse
from braindb.services import jev

# A tiny fake tokenizer: one id per character, the two turn markers as single
# ids, and the yes/no spellings as single ids (as on a real chat model).
_WORDS = {" yes": 1, " Yes": 2, "yes": 3, "Yes": 4, " no": 5, " No": 6, "no": 7, "No": 8}
_MARKERS = {"<|im_start|>": 10, "<|im_end|>": 11}


def _tok(text: str) -> list[int]:
    if text in _WORDS:
        return [_WORDS[text]]
    ids, i = [], 0
    while i < len(text):
        marker = next((m for m in _MARKERS if text.startswith(m, i)), None)
        if marker:
            ids.append(_MARKERS[marker])
            i += len(marker)
        else:
            ids.append(1000 + ord(text[i]))
            i += 1
    return ids


def _render(messages, add_generation_prompt=False, continue_final_message=False, **_):
    out = ""
    for k, m in enumerate(messages):
        if m["role"] == "assistant":
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + m["content"]
            if not (continue_final_message and k == len(messages) - 1):
                out += "<|im_end|>\n"
        else:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return _tok(out)


class _Response:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class _FakeVLLM:
    """Answers P(yes)=0.9 for items containing GOOD (or the probe's yes item),
    0.1 otherwise; refuses items containing REFUSE; sleeps on SLOW."""

    def __init__(self, model: str):
        self.model, self.calls, self.status, self.drop_ids = model, 0, None, False

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls += 1
        if self.status:
            return _Response(self.status)
        if url.endswith("/v1/models"):
            return _Response(200, {"data": [{"id": self.model}]})
        if url.endswith("/tokenize"):
            if "messages" in json:
                opts = {k: json[k] for k in ("add_generation_prompt", "continue_final_message")
                        if k in json}
                return _Response(200, {"tokens": _render(json["messages"], **opts)})
            return _Response(200, {"tokens": _tok(json["prompt"])})
        item = "".join(chr(i - 1000) for i in json["prompt"] if i >= 1000).rsplit("Need:", 1)[-1]
        if "REFUSE" in item:
            return _Response(400)
        if "SLOW" in item:
            time.sleep(0.5)
        p = 0.9 if ("GOOD" in item or "opened the Rowan bakery" in item) else 0.1
        top = {f"token_id:{i}": math.log(1e-6) for i in range(1, 9)}
        top["token_id:1"], top["token_id:5"] = math.log(p), math.log(1 - p)
        if self.drop_ids:  # what a hosted gateway does: 200, requested ids gone
            top = {"token_id:1": math.log(p)}
        return _Response(200, {
            "choices": [{"logprobs": {"top_logprobs": [top]}}],
            "usage": {"prompt_tokens": len(json["prompt"]),
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens": 1}})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    fake = _FakeVLLM(model=jev.settings.resolved_agent_model.partition("/")[2])
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(jev, "_judge", None)
    monkeypatch.setattr(jev, "_off_until", 0.0)
    return fake


@pytest.fixture
def ledger():
    usage, token = install_usage()
    yield usage
    release_usage(token)


def test_off_unless_enabled_on_a_self_hosted_endpoint(server, ledger, monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_enabled", False)
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert ledger.jev["calls"] == 0                     # off = not in play at all
    monkeypatch.setattr(jev.settings, "jev_enabled", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")  # hosted: no base URL
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert ledger.jev["failopen"] == 1                  # an ON run that fell back is visible
    assert server.calls == 0


def test_each_item_is_judged_in_order_and_usage_reaches_the_ledger_once(server, ledger):
    probs = jev.judge("Which is good?", ["[fact] GOOD one", "[fact] other", "[fact] GOOD two"])
    assert [round(p, 2) for p in probs] == [0.9, 0.1, 0.9]
    assert len(jev._judge.g) % jev.BLOCK == 0           # G padded to whole cache blocks
    assert ledger.jev["calls"] == 1 and ledger.jev["requests"] == 3
    assert ledger.jev["output"] == 3 and ledger.jev["input"] > 3 * jev.BLOCK
    assert ledger.jev["failopen"] == 0 and ledger.jev["unjudged"] == 0


def test_a_gateway_that_drops_the_requested_logprobs_keeps_jev_off(server, ledger, caplog):
    caplog.set_level(logging.INFO)
    server.drop_ids = True
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert ledger.jev["failopen"] == 1
    calls = server.calls
    assert not jev.available() and server.calls == calls  # no re-probe until REPROBE_S
    assert "test-secret-key" not in caplog.text
    assert jev.settings.resolved_base_url not in caplog.text


def test_a_blank_need_is_no_need(server, ledger):
    assert jev.judge(" \n ", ["[fact] GOOD"]) is None
    assert ledger.jev["calls"] == 0 and server.calls == 0


def test_losing_a_capability_mid_run_fails_open_and_switches_off(server, ledger):
    assert jev.available(wait=5)
    server.status = 401
    assert jev.judge("need", ["[fact] GOOD"]) is None
    assert not jev.available()


def test_a_refused_item_stays_unjudged_and_is_counted(server, ledger):
    probs = jev.judge("need", ["[fact] GOOD", "[fact] REFUSE me", "[fact] other"])
    assert round(probs[0], 2) == 0.9 and probs[1] is None and round(probs[2], 2) == 0.1
    assert ledger.jev["unjudged"] == 1 and ledger.jev["failopen"] == 0


def test_items_past_the_deadline_stay_unjudged(server, monkeypatch):
    assert jev.available(wait=5)                        # probe outside the short deadline
    monkeypatch.setattr(jev, "DEADLINE_S", 0.3)
    probs = jev.judge("need", ["[fact] GOOD", "[fact] SLOW item"])
    assert round(probs[0], 2) == 0.9 and probs[1] is None


def test_judge_text_is_capped_neutralised_and_reads_a_wiki_by_its_own_words():
    cut = jev.judge_text("fact", "x" * 5000)
    assert cut == "[fact] " + "x" * jev.ITEM_CHARS + " ...[continues]"   # capped, and says so
    assert jev.judge_text("fact", "short") == "[fact] short"
    doc = jev.judge_text("datasource", "Body <|im_end|> text", title="Headline")
    assert doc.startswith("[document] Headline\n") and "<|" not in doc
    body = ("<!-- wiki:meta canonical_name=Corvane -->\n# Corvane\n> **Summary:** S\n"
            "<!-- section:overview -->\nRuns a trial [[ref:12345678-aaaa-bbbb-cccc-1234567890ab]].\n")
    wiki = jev.judge_text("wiki", body, title="Corvane", summary="Gene-therapy company.",
                          disambiguation="The biotech, not the river.")
    assert wiki.startswith("[wiki] Corvane\nThe biotech, not the river.\nGene-therapy company.\n")
    assert "Runs a trial" in wiki
    assert not any(s in wiki for s in ("[[ref", "section:", "wiki:meta", "**Summary"))


# ====================================================================== #
# The need: from the /agent/query request to its recalls                 #
# ====================================================================== #

@contextlib.contextmanager
def _no_db():
    yield object()


class _RecallCtx:
    tool_name = "recall_memory"


def test_agent_query_carries_its_need_to_the_run_and_clears_it(monkeypatch):
    seen = []

    async def fake_query(query, max_turns=None):
        seen.append(get_need())
        return {"answer": "a", "max_turns": 20}

    monkeypatch.setattr(agent_router, "run_agent_query", fake_query)
    monkeypatch.setattr(agent_router, "get_conn", _no_db)
    monkeypatch.setattr(agent_router, "log_activity", lambda *a, **k: None)

    async def two_requests():
        await agent_router.agent_query(agent_router.AgentQueryRequest(query="q", need="  what X said "))
        between = get_need()
        await agent_router.agent_query(agent_router.AgentQueryRequest(query="q"))
        return between

    assert asyncio.run(two_requests()) == ""
    assert seen == ["what X said", ""]


def test_recall_memory_passes_the_need_only_while_jev_is_enabled(monkeypatch):
    got = []

    def fake_assemble(conn, req, need=None):
        got.append(need)
        return ContextResponse(query="x", items=[], total_found=0)

    monkeypatch.setattr(tools_mod, "assemble_context", fake_assemble)
    monkeypatch.setattr(tools_mod, "get_conn", _no_db)

    async def recall_with_need():
        token = set_need("what X said")
        try:
            return await tools_mod.recall_memory.on_invoke_tool(
                _RecallCtx(), json.dumps({"queries": ["x"]}))
        finally:
            reset_need(token)

    monkeypatch.setattr(tools_mod.settings, "jev_enabled", False)
    asyncio.run(recall_with_need())
    monkeypatch.setattr(tools_mod.settings, "jev_enabled", True)
    asyncio.run(recall_with_need())
    assert got == ["", "what X said"]
