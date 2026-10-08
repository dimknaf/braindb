"""Unit tests for wiki triage by JEV (braindb/services/wiki_triage.py and
`Choice` in braindb/services/jev.py).

DB-free and GPU-free: a fake vLLM endpoint renders, tokenizes and answers by
cues in the seed. These pin the contract the maintainer relies on:

- every decision is ONE output token (max_tokens=1) read from the options'
  logprobs, with thinking off (the answer header is the template's empty think
  block); the only generated text is a new page's name; a seed's text is
  encoded once for all its picks;
- an attach maps the picked candidate to its catalog number, a consolidate
  needs two duplicates, a create carries the written name;
- anything uncertain (low mass, no fitting candidate, fewer than two
  duplicates, an empty or cut-off name, a failed probe, a failed seed) is left
  to the maintainer agent;
- a profile's maintainer additions shape the agent only (never the guide),
  and triage is off under a profile that replaces the maintainer's prompt;
- off unless enabled on a self-hosted server, with no request made.
"""
from __future__ import annotations

import contextlib
import math

import pytest

from braindb.agent.run_state import install_usage, release_usage
from braindb.services import jev, wiki_triage

_WORDS = [" skip", " Skip", " ambiguous", " Ambiguous", " consolidate", " Consolidate", " attach", " Attach",
          " create", " Create", " none", " None", " yes", " Yes", " no", " No"] + [f" {c}" for c in "ABCDEFGH"]
_IDS = {w: i for i, w in enumerate(_WORDS, start=1)}
_MARKERS = {"<|im_start|>": 900, "<|im_end|>": 901}


def _tok(text: str) -> list[int]:
    if text in _IDS:
        return [_IDS[text]]
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


def _text(ids: list[int]) -> str:
    back = {v: k for k, v in {**_IDS, **_MARKERS}.items()}
    return "".join(back.get(i, chr(i - 1000) if i >= 1000 else "?") for i in ids)


def _render(messages, add_generation_prompt=False, thinking=True):
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:  # as Qwen3 templates do: thinking off closes an empty think block
        out += "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")
    return _tok(out)


class _Response:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class _FakeVLLM:
    """Answers each pick by cues in the current seed (the text after the last
    'SEED:'): CUE_CREATE / CUE_SKIP / CUE_CONSOL / CUE_LOWMASS (else attach);
    PICK_<label> or PICK_NONE for the wiki; DUP_<labels> for duplicates;
    CUE_NONAME / CUE_NONENAME / CUE_LONGNAME for an empty / "None" / cut-off
    name. The template closes an empty think block only with thinking off."""

    def __init__(self):
        self.requests, self.encoded = [], []
        self.probe_ok, self.probe_wiki = True, "B"

    def request(self, method, url, **kwargs):
        body = kwargs.get("json") or {}
        if url.endswith("/tokenize"):
            if "messages" in body:
                thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
                return _Response(200, {"tokens": _render(body["messages"], body.get("add_generation_prompt"),
                                                         thinking)})
            self.encoded.append(body["prompt"])
            return _Response(200, {"tokens": _tok(body["prompt"])})
        self.requests.append(body)
        prompt = _text(body["prompt"])
        seed = prompt.rsplit("SEED:", 1)[-1]
        lead = prompt.rsplit("<think>\n\n</think>\n\n", 1)[-1]
        if "logprob_token_ids" not in body:                       # write: a new page's name
            name = " None" if "CUE_NONENAME" in seed else "" if "CUE_NONAME" in seed else " Ferrow Bridge"
            finish = "length" if "CUE_LONGNAME" in seed else "stop"
            return _Response(200, {"choices": [{"text": name, "finish_reason": finish}],
                                   "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 3}})
        want, mass = None, 0.98
        if lead == "action:":
            want = ("create" if "CUE_CREATE" in seed else "skip" if "CUE_SKIP" in seed
                    else "consolidate" if "CUE_CONSOL" in seed else "attach")
            if "CUE_LOWMASS" in seed or ("Calder Labs hired" in seed and not self.probe_ok):
                mass = 0.05
        elif lead.endswith("wiki:"):
            want = next((c for c in "ABCDEFGH" if f"PICK_{c}" in seed),
                        self.probe_wiki if "Calder Labs hired" in seed else "none")
        elif lead.endswith("same subject (yes or no):"):
            label = lead.rsplit("[", 1)[-1][0]
            dups = seed.split("DUP_", 1)[-1].split()[0] if "DUP_" in seed else ""
            want = "yes" if label in dups else "no"
        top = {}
        for i in body["logprob_token_ids"]:
            word = _WORDS[i - 1].strip().lower()
            top[f"token_id:{i}"] = math.log(mass * 0.97 if word == want.lower() else mass * 0.01)
        return _Response(200, {"choices": [{"logprobs": {"top_logprobs": [top]}}],
                               "usage": {"prompt_tokens": len(body["prompt"]),
                                         "prompt_tokens_details": {"cached_tokens": 0},
                                         "completion_tokens": 1}})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    fake = _FakeVLLM()
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(wiki_triage, "_state", {"choice": None, "options": None, "off_until": 0.0})
    return fake


ACTIVE = {"w1": (3, {"canonical_name": "Calder Labs", "char_count": 5200}),
          "w2": (7, {"canonical_name": "Calder Labs Ltd", "char_count": 900})}
LABELS = {"A": "w1", "B": "w2"}
CAT = [{"id": "w0", "canonical_name": "Other", "char_count": 1},
       {"id": "w1", "canonical_name": "Calder Labs", "char_count": 5200},
       {"id": "w2", "canonical_name": "Calder Labs Ltd", "char_count": 900}]


def _one(text, labels=LABELS):
    choice, options = wiki_triage._choice()
    return wiki_triage._decide_one(choice, options, "seed-1", f"SEED: {text}", labels, ACTIVE)


def test_attach_maps_the_picked_candidate_to_its_catalog_number(server):
    decision, usage = _one("a fact CUE_ATTACH PICK_B")
    assert decision.action == "attach" and decision.target_wiki_no == 7
    assert decision.entity_id == "seed-1" and decision.proposed_name is None
    assert usage["requests"] == 2 and usage["output"] == 2           # one token per decision
    assert sum("CUE_ATTACH" in t for t in server.encoded) == 1        # the seed is encoded once


def test_every_pick_is_one_token_with_thinking_off(server):
    _one("a fact CUE_ATTACH PICK_A")
    picks = [r for r in server.requests if "logprob_token_ids" in r]
    assert picks and all(r["max_tokens"] == 1 and r["temperature"] == 0 for r in picks)
    for r in picks:
        tail = _text(r["prompt"]).rsplit("<|im_start|>assistant\n", 1)[-1]
        assert tail.startswith("<think>\n\n</think>\n\n")             # the answer opens with thinking closed
        assert tail.endswith(("action:", "wiki:"))


def test_create_writes_only_the_name(server):
    decision, usage = _one("a bridge reopened CUE_CREATE")
    assert decision.action == "create" and decision.proposed_name == "Ferrow Bridge"
    assert usage["requests"] == 2


def test_skip_needs_nothing_more(server):
    decision, usage = _one("_kw_token CUE_SKIP")
    assert decision.action == "skip" and usage["requests"] == 1


def test_consolidate_needs_two_duplicates(server):
    decision, _ = _one("CUE_CONSOL DUP_AB")
    assert decision.action == "consolidate" and decision.consolidate_nos == [3, 7]
    assert _one("CUE_CONSOL DUP_A")[0] is None                      # one duplicate is not a merge


@pytest.mark.parametrize("text,labels", [
    ("CUE_LOWMASS", LABELS),             # the options hold little probability
    ("CUE_ATTACH PICK_NONE", LABELS),     # attach, but no candidate fits
    ("CUE_ATTACH PICK_A", {}),           # attach with no candidate at all
    ("CUE_CREATE CUE_NONAME", LABELS),    # create, but no name came
    ("CUE_CREATE CUE_NONENAME", LABELS),  # create, but the name says none
    ("CUE_CREATE CUE_LONGNAME", LABELS),  # create, but the name was cut off
])
def test_uncertain_seeds_are_left_to_the_agent(server, text, labels):
    assert _one(text, labels)[0] is None


def test_decide_returns_only_confident_seeds_and_counts_the_rest(server, monkeypatch):
    texts = {"s1": "SEED: CUE_ATTACH PICK_A", "s2": "SEED: CUE_LOWMASS"}
    monkeypatch.setattr(wiki_triage, "_brief", lambda conn, seed, active: (texts[seed["id"]], LABELS))
    monkeypatch.setattr(wiki_triage, "get_conn", contextlib.nullcontext)
    ledger, token = install_usage()
    try:
        decided = wiki_triage.decide([{"brief": {"id": "s1"}}, {"brief": {"id": "s2"}}], CAT)
    finally:
        release_usage(token)
    assert set(decided) == {"s1"} and decided["s1"].target_wiki_no == 2   # catalog numbering from `cat`
    used = ledger.as_dict()["jev"]
    assert used["calls"] == 2 and used["failopen"] == 1 and used["output"] == used["requests"]


def test_a_seed_that_fails_goes_to_the_agent_alone(server, monkeypatch):
    def brief(conn, seed, active):
        if seed["id"] == "s2":
            raise RuntimeError("recall failed")
        return "SEED: CUE_ATTACH PICK_A", LABELS

    monkeypatch.setattr(wiki_triage, "_brief", brief)
    monkeypatch.setattr(wiki_triage, "get_conn", contextlib.nullcontext)
    decided = wiki_triage.decide([{"brief": {"id": "s1"}}, {"brief": {"id": "s2"}}], CAT)
    assert set(decided) == {"s1"} and wiki_triage.wanted()


def test_off_unless_enabled_on_a_self_hosted_server(server, monkeypatch):
    seeds = [{"brief": {"id": "s1"}}]
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", False)
    assert not wiki_triage.wanted() and wiki_triage.decide(seeds, []) == {}
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")          # a hosted gateway
    assert not wiki_triage.wanted() and wiki_triage.decide(seeds, []) == {}
    assert server.requests == []


@pytest.mark.parametrize("failure", ["action", "wiki"])
def test_a_failed_probe_switches_triage_off(server, monkeypatch, failure):
    if failure == "action":
        server.probe_ok = False
    else:
        server.probe_wiki = "A"                                         # the wrong candidate
    monkeypatch.setattr(wiki_triage, "get_conn", contextlib.nullcontext)
    assert wiki_triage.decide([{"brief": {"id": "s1"}}], []) == {}
    assert not wiki_triage.wanted()                                     # off until re-probed


def test_the_maintainers_profile_additions_stay_out_of_the_guide(server, monkeypatch, tmp_path):
    # Measured: written for an agent that researches first, they made the
    # one-token decisions skip pages the agent creates.
    (tmp_path / "news").mkdir()
    (tmp_path / "news" / "wiki_maintainer.add.md").write_text("Skip non-subjects.", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "news")
    assert wiki_triage.wanted()
    choice, _ = wiki_triage._choice()
    assert "Skip non-subjects." not in _text(choice.g)


def test_off_under_a_profile_that_replaces_the_maintainer_prompt(server, monkeypatch, tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "wiki_maintainer.replace.md").write_text("Another contract: {seeds}", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "own")
    assert not wiki_triage.wanted() and wiki_triage.decide([{"brief": {"id": "s1"}}], CAT) == {}
    assert server.requests == []
