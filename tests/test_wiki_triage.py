"""Unit tests for wiki triage by JEV (braindb/services/wiki_triage.py and
`Choice` in braindb/services/jev.py).

DB-free and GPU-free: a fake OpenAI-compatible chat endpoint (conftest.py
`fake_chat`) writes the model's answer by cues in the seed. These pin the
contract the maintainer relies on:

- one standard request per seed, thinking off, the guide and its worked
  examples as chat turns; every decision is ONE token of that one answer,
  read from its logprobs where its lead ends; the only words taken from
  the answer are a new page's name;
- an attach maps the picked candidate to its catalog number, a consolidate
  needs two duplicates, a create carries the written name;
- anything uncertain (low mass, no fitting candidate, fewer than two
  duplicates, an empty or cut-off name, a failed probe, a failed seed) is left
  to the maintainer agent;
- a profile's maintainer additions shape the agent only (never the guide),
  and triage is off under a profile that replaces the maintainer's prompt;
- off unless enabled, with no request made; on any profile whose server
  passes the probe.
"""
from __future__ import annotations

import contextlib

import pytest

from braindb.agent.run_state import install_usage, release_usage
from braindb.services import jev, wiki_triage

ACTIONS = wiki_triage.ACTIONS


def _pick(word: str, options: list[str], mass: float = 0.98) -> tuple[str, dict]:
    """The answer token `word`, with the options' probabilities around it."""
    alts = {f" {o}": mass * (0.97 if o == word else 0.01) for o in options}
    alts["\n"] = max(0.0, 1.0 - sum(alts.values()))
    return f" {word}", alts


class _Model:
    """Answers each seed by cues in it (the brief after 'SEED:'): CUE_CREATE /
    CUE_SKIP / CUE_CONSOL / CUE_LOWMASS (else attach); PICK_<label> or
    PICK_NONE for the wiki; DUP_<labels> for duplicates; CUE_NONAME /
    CUE_NONENAME / CUE_LONGNAME for an empty / "None" / cut-off name. The
    probe seed is the examples' own probe."""

    def __init__(self):
        self.probe_ok, self.probe_wiki = True, "B"

    def __call__(self, body: dict):
        seed = body["messages"][-1]["content"]
        probe = "Calder Labs hired" in seed
        action = ("create" if "CUE_CREATE" in seed else "skip" if "CUE_SKIP" in seed
                  else "consolidate" if "CUE_CONSOL" in seed else "attach")
        mass = 0.05 if "CUE_LOWMASS" in seed or (probe and not self.probe_ok) else 0.98
        out = [("action", None), (":", None), _pick(action, ACTIONS, mass)]
        if action == "attach":
            wiki = next((c for c in "ABCDEFGH" if f"PICK_{c}" in seed), self.probe_wiki if probe else "none")
            out += [("\n", None), ("wiki", None), (":", None), _pick(wiki, list("ABCDEFGH") + ["none"])]
        elif action == "consolidate":
            dups = seed.split("DUP_", 1)[-1].split()[0] if "DUP_" in seed else ""
            for label in "AB":
                out += [("\n", None), (f"[{label}] same subject (yes or no):", None),
                        _pick("yes" if label in dups else "no", ["yes", "no"])]
        elif action == "create":
            name = " None" if "CUE_NONENAME" in seed else "" if "CUE_NONAME" in seed else " Ferrow Bridge"
            out += [("\n", None), ("name", None), (":", None)]
            out += [(" word", None)] * 200 if "CUE_LONGNAME" in seed else ([(name, None)] if name else [])
        return out


@pytest.fixture
def server(monkeypatch, fake_chat):
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    model = _Model()
    fake = fake_chat(model)
    fake.model_cues = model
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
    assert usage["requests"] == 1                                    # every decision reads one answer
    assert sum("CUE_ATTACH" in b["messages"][-1]["content"] for b in server.bodies) == 1


def test_a_seed_is_one_standard_request_with_thinking_off_and_the_examples_as_turns(server):
    _one("a fact CUE_ATTACH PICK_A")
    body = next(b for b in server.bodies if "CUE_ATTACH" in b["messages"][-1]["content"])
    assert body["reasoning_effort"] == "none" and body["temperature"] == 0 and body["logprobs"] is True
    assert body["messages"][0] == {"role": "system", "content": wiki_triage._GUIDE}
    examples = wiki_triage._EXAMPLES["examples"]
    assert [m["role"] for m in body["messages"][1:-1]] == ["user", "assistant"] * len(examples)
    assert body["messages"][2]["content"] == examples[0]["answer"]  # answered as the examples answer


def test_create_writes_only_the_name(server):
    decision, usage = _one("a bridge reopened CUE_CREATE")
    assert decision.action == "create" and decision.proposed_name == "Ferrow Bridge"
    assert usage["requests"] == 1


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
    assert used["calls"] == 2 and used["failopen"] == 1 and used["requests"] == 2


def test_a_seed_that_fails_goes_to_the_agent_alone(server, monkeypatch):
    def brief(conn, seed, active):
        if seed["id"] == "s2":
            raise RuntimeError("recall failed")
        return "SEED: CUE_ATTACH PICK_A", LABELS

    monkeypatch.setattr(wiki_triage, "_brief", brief)
    monkeypatch.setattr(wiki_triage, "get_conn", contextlib.nullcontext)
    decided = wiki_triage.decide([{"brief": {"id": "s1"}}, {"brief": {"id": "s2"}}], CAT)
    assert set(decided) == {"s1"} and wiki_triage.wanted()


def test_off_unless_enabled_and_on_any_profile_whose_server_passes_the_probe(server, monkeypatch):
    seeds = [{"brief": {"id": "s1"}}]
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", False)
    assert not wiki_triage.wanted() and wiki_triage.decide(seeds, []) == {}
    assert server.bodies == []
    monkeypatch.setattr(jev.settings, "jev_wiki_triage", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")          # a hosted gateway: the probe decides
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-secret-key")
    assert wiki_triage.wanted() and wiki_triage._choice() is not None


@pytest.mark.parametrize("failure", ["action", "wiki"])
def test_a_failed_probe_switches_triage_off(server, monkeypatch, failure):
    if failure == "action":
        server.model_cues.probe_ok = False
    else:
        server.model_cues.probe_wiki = "A"                              # the wrong candidate
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
    assert not any("Skip non-subjects." in m["content"] for m in choice.turns)


def test_off_under_a_profile_that_replaces_the_maintainer_prompt(server, monkeypatch, tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "wiki_maintainer.replace.md").write_text("Another contract: {seeds}", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "own")
    assert not wiki_triage.wanted() and wiki_triage.decide([{"brief": {"id": "s1"}}], CAT) == {}
    assert server.bodies == []
