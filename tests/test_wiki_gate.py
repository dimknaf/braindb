"""Unit tests for the wiki writer's JEV gate (braindb/services/wiki_gate.py).

DB-free and GPU-free: a fake vLLM endpoint renders, tokenizes and answers by
cues in the fact. These pin the contract the writer relies on:

- every decision is ONE output token (max_tokens=1) read from the options'
  logprobs, with thinking off;
- only facts about the subject reach the writer: a fact judged no or
  ambiguous, or one whose options hold little probability, is left out;
  wikis, keywords, the job's own members and facts the page already cites are
  never offered as found facts;
- a job none of whose members is kept goes to the full writer (None);
- on a long page each kept fact gets its section by name (or "new"); an
  unsure placement is left to the writer;
- off unless enabled on a self-hosted server, never under a profile that
  replaces the writer's prompt, never for consolidate; a failed probe turns
  it off.
"""
from __future__ import annotations

import contextlib
import math
import uuid
from types import SimpleNamespace

import pytest

from braindb.services import jev, wiki_gate

_WORDS = [" yes", " Yes", " no", " No", " ambiguous", " Ambiguous", " new", " New"] + [f" {c}" for c in "ABCDEFGHIJKLMNOP"]
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
    if add_generation_prompt:
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
    """Answers by cues in the current fact (the text after the last 'FACT:'):
    identity — CUE_NO / CUE_AMB / CUE_LOWMASS, else yes; placement —
    PLACE_<letter> / PLACE_NEW / CUE_LOWPLACE. The probe's facts are answered
    as the probe expects unless `probe_ok` is False."""

    def __init__(self):
        self.requests, self.probe_ok = [], True

    def request(self, method, url, **kwargs):
        body = kwargs.get("json") or {}
        if url.endswith("/tokenize"):
            if "messages" in body:
                thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking", True)
                return _Response(200, {"tokens": _render(body["messages"], body.get("add_generation_prompt"), thinking)})
            return _Response(200, {"tokens": _tok(body["prompt"])})
        self.requests.append(body)
        prompt = _text(body["prompt"])
        fact = prompt.rsplit("FACT:", 1)[-1]
        lead = prompt.rsplit("<think>\n\n</think>\n\n", 1)[-1]
        mass = 0.98
        if lead.endswith("(yes, no or ambiguous):"):
            want = "no" if ("CUE_NO" in fact or "Calder Insurance" in fact) else "ambiguous" if "CUE_AMB" in fact else "yes"
            if "CUE_LOWMASS" in fact or ("Calder Labs hired" in fact and not self.probe_ok):
                mass = 0.05
        else:
            want = "new" if "PLACE_NEW" in fact else next((c for c in "ABCDEFGHIJKLMNOP" if f"PLACE_{c} " in fact), None)
            want = want or ("C" if "revenue target" in fact else "A")
            if "CUE_LOWPLACE" in fact:
                mass = 0.05
        top = {}
        for i in body["logprob_token_ids"]:
            word = _WORDS[i - 1].strip()
            top[f"token_id:{i}"] = math.log(mass * 0.97 if word.lower() == want.lower() else mass * 0.01)
        return _Response(200, {"choices": [{"logprobs": {"top_logprobs": [top]}}],
                               "usage": {"prompt_tokens": len(body["prompt"]),
                                         "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens": 1}})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    fake = _FakeVLLM()
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(wiki_gate, "_state", {"identity": None, "placement": None, "options": None, "off_until": 0.0})
    monkeypatch.setattr(wiki_gate, "get_conn", contextlib.nullcontext)
    return fake


def _entity(content, entity_type="fact"):
    return {"id": str(uuid.uuid4()), "entity_type": entity_type, "content": content, "keywords": []}


def _recall(monkeypatch, items):
    """Recall returns `items` (entity dicts) as cut previews; the full rows
    come from fetch_members. Records what recall was asked."""
    asked, rows = [], {i["id"]: i for i in items}

    def fake(conn, req, need=None):
        asked.append(req)
        return SimpleNamespace(items=[SimpleNamespace(id=uuid.UUID(i["id"]), entity_type=i["entity_type"],
                                                      content=i["content"][:20] + "--truncated-- get_entity(...)",
                                                      keywords=i["keywords"]) for i in items])

    monkeypatch.setattr(wiki_gate, "assemble_context", fake)
    monkeypatch.setattr(wiki_gate, "fetch_members", lambda conn, ids: [rows[i] for i in reversed(ids)])
    return asked


SHORT_PAGE = "<!-- wiki:meta canonical_name=Calder Labs keywords=calder;robots -->\n# Calder Labs\n> **Summary:** battery testing robots maker.\n"
LONG_PAGE = (SHORT_PAGE + "<!-- section: overview -->\nOverview text.\n<!-- section: timeline -->\n2019 founded.\n"
             "<!-- section: finances -->\nRevenue 41M.\n")


def _picks(server, lead):
    return [r for r in server.requests if _text(r["prompt"]).rstrip().endswith(lead)]


def test_only_facts_about_the_subject_reach_the_writer(server, monkeypatch):
    m1 = _entity("Calder Labs opened a factory")
    cited = _entity("already on the page")
    f_yes = _entity("Calder Labs hired engineers in May, a long fact that recall only previews")
    f_amb, f_low = _entity("CUE_AMB he said so"), _entity("CUE_LOWMASS x")
    page = SHORT_PAGE + f"<!-- section: overview -->\nText [[ref:{cited['id']}]]\n"
    asked = _recall(monkeypatch, [_entity("a page", "wiki"), _entity("calder", "keyword"), m1, cited,
                                  f_yes, f_amb, f_low])
    gate = wiki_gate.prepare("attach", "Calder Labs", page, [m1], long_page=False)
    assert [m["id"] for m in gate.members] == [m1["id"]]
    assert gate.found == [f_yes]                                         # the full row, not recall's preview
    assert set(gate.excluded) == {f_amb["id"], f_low["id"]}
    # the wiki, the keyword, the member and the cited fact were never asked about
    picks = _picks(server, wiki_gate.ID_LEAD)[2:]                         # after the two probe picks
    assert len(picks) == 4
    assert all("SUBJECT: Calder Labs\nSUMMARY: battery testing robots maker.\nFACT: [fact] " in _text(r["prompt"])
               for r in picks)                                           # every pick states the subject
    assert "--truncated--" not in "".join(_text(r["prompt"]) for r in picks)
    assert asked[0].queries == ["Calder Labs", "calder", "robots"] and asked[0].include_always_on_rules is False
    assert gate.sections == {}


def test_a_doubtful_member_leaves_the_job_to_the_full_writer(server, monkeypatch):
    # A member must be cited wherever it is written: the gate never drops one.
    _recall(monkeypatch, [_entity("Calder Labs hired engineers")])
    members = [_entity("Calder Labs opened a factory"), _entity("CUE_NO Calder Insurance raised premiums")]
    assert wiki_gate.prepare("attach", "Calder Labs", SHORT_PAGE, members, long_page=False) is None
    assert wiki_gate.prepare("attach", "Calder Labs", SHORT_PAGE, [_entity("CUE_LOWMASS x")], long_page=False) is None


def test_every_pick_is_one_token_with_thinking_off(server, monkeypatch):
    _recall(monkeypatch, [])
    wiki_gate.prepare("create", "Calder Labs", "", [_entity("Calder Labs opened a factory")], long_page=False)
    picks = [r for r in server.requests if "logprob_token_ids" in r]
    assert picks and all(r["max_tokens"] == 1 and r["temperature"] == 0 for r in picks)
    for r in picks:
        tail = _text(r["prompt"]).rsplit("<|im_start|>assistant\n", 1)[-1]
        assert tail.startswith("<think>\n\n</think>\n\n")


def test_the_subject_is_the_summary_else_the_seed_facts(server, monkeypatch):
    m = _entity("Calder Labs opened a factory")
    assert wiki_gate._subject("Calder Labs", SHORT_PAGE, [m]) == "SUBJECT: Calder Labs\nSUMMARY: battery testing robots maker."
    assert wiki_gate._subject("Calder Labs", "", [m]).startswith("SUBJECT: Calder Labs\nSEED FACTS (assigned to this page):\n- [fact]")


def test_on_a_long_page_each_kept_fact_gets_its_section(server, monkeypatch):
    m1, m2, m3 = _entity("PLACE_B opened a factory"), _entity("PLACE_NEW chief executive resigned"), _entity("CUE_LOWPLACE x")
    f_kept, f_out = _entity("PLACE_C revenue up"), _entity("CUE_NO PLACE_A other firm")
    _recall(monkeypatch, [f_kept, f_out])
    gate = wiki_gate.prepare("attach", "Calder Labs", LONG_PAGE, [m1, m2, m3], long_page=True)
    # the unsure one is left to the writer; a found fact left out is never placed
    assert gate.sections == {m1["id"]: "timeline", m2["id"]: "new", f_kept["id"]: "finances"}
    outline = _text(_picks(server, wiki_gate.PL_LEAD)[-1]["prompt"])
    assert "A. overview - Overview text." in outline and "C. finances - Revenue 41M." in outline


def test_a_page_with_more_sections_than_letters_is_not_placed(server, monkeypatch):
    page = SHORT_PAGE + "".join(f"<!-- section: s{i} -->\nText {i}.\n" for i in range(len(wiki_gate.LETTERS) + 1))
    _recall(monkeypatch, [])
    gate = wiki_gate.prepare("attach", "Calder Labs", page, [_entity("PLACE_B opened a factory")], long_page=True)
    assert gate is not None and gate.sections == {}                      # never "new" for a home past the letters
    assert _picks(server, wiki_gate.PL_LEAD) == [server.requests[2]]     # the probe's pick only


def test_the_deadline_leaves_the_job_to_the_full_writer(server, monkeypatch):
    monkeypatch.setattr(wiki_gate, "DEADLINE_S", -1)
    _recall(monkeypatch, [])
    assert wiki_gate.prepare("attach", "Calder Labs", SHORT_PAGE, [_entity("Calder Labs opened a factory")], False) is None
    assert len(server.requests) == 3                                     # the probe's three picks, no more


def test_every_jev_request_counts_toward_one_in_flight_limit(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    state = {"now": 0, "max": 0}
    lock = threading.Lock()

    class _Slow:
        def request(self, method, url, **kwargs):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return _Response(200, {"ok": True})

    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.setattr(jev, "_session", _Slow())
    server = jev._Server()
    with ThreadPoolExecutor(max_workers=3 * jev.WORKERS) as pool:     # several features at once
        list(pool.map(lambda _: server.complete({"prompt": [1]}), range(3 * jev.WORKERS)))
    assert 1 < state["max"] <= jev.WORKERS


def test_the_gated_writer_has_only_the_page_tools():
    from braindb.agent import agent as agent_mod
    gated = agent_mod.get_gated_writer_agent()
    assert {t.name for t in gated.tools} == {"read_wiki_outline", "read_wiki_section", "check_members_cited",
                                             "edit_wiki_section", "delete_wiki_section", "validate_wiki",
                                             "handoff_to_successor", "final_answer"}
    assert gated.instructions == agent_mod._GATED_WRITER_PROMPT and gated is not agent_mod.get_writer_agent()


def test_off_unless_enabled_on_a_self_hosted_server_and_never_for_consolidate(server, monkeypatch):
    members = [_entity("Calder Labs opened a factory")]
    assert wiki_gate.prepare("consolidate", "x", SHORT_PAGE, members, long_page=False) is None
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", False)
    assert not wiki_gate.wanted() and wiki_gate.prepare("attach", "x", SHORT_PAGE, members, False) is None
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")          # a hosted gateway
    assert not wiki_gate.wanted()
    assert server.requests == []


def test_off_under_a_profile_that_replaces_the_writer_prompt(server, monkeypatch, tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "wiki_writer.replace.md").write_text("Another contract", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "own")
    assert not wiki_gate.wanted()


def test_a_failed_probe_switches_the_gate_off(server, monkeypatch):
    server.probe_ok = False
    _recall(monkeypatch, [])
    assert wiki_gate.prepare("attach", "Calder Labs", SHORT_PAGE, [_entity("Calder Labs opened a factory")], False) is None
    assert not wiki_gate.wanted()                                          # off until re-probed
