"""Unit tests for the wiki writer's JEV gate (braindb/services/wiki_gate.py).

DB-free and GPU-free: a fake vLLM endpoint renders turns with a template
shaped like the served model's (messages trimmed, a reasoning line in the
system block unless thinking is off, every assistant turn opened with a
think block), tokenizes one token per character with the special markers
and "\\n\\n" as single tokens, keeps a hybrid model's cache (only the last
whole block a request ended in), and answers by cues in the fact asked
about. These pin the contract the writer relies on:

- the decisions are taken INSIDE the gated writer's own first request: each
  is that request (its system prompt, tools and template arguments) through
  the facts so far, then ONE token with thinking off; the writer's own
  request then reuses what was kept;
- only the found facts judged about the subject stay: one judged no, or
  whose options hold little probability, is rolled back and the next fact
  follows what was kept; wikis, keywords, the job's members and the facts
  the page already cites are never offered;
- the members are never asked about and never send the job to the full
  writer;
- on a long page each member and kept fact gets its section (an unsure one
  is left to the writer); a page with more sections than letters is not
  placed;
- off unless enabled on a self-hosted server, never under a profile that
  replaces the writer's prompt; a failed probe turns it off; a request the
  server renders differently, an error or the deadline gives None.
"""
from __future__ import annotations

import json
import math
import re
import uuid
from types import SimpleNamespace

import pytest

from braindb.services import jev, reader, wiki_gate
from braindb.services.wiki_sections import parse_sections

BLOCK = 64
_SPECIAL = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
_WORDS = [" yes", " Yes", " no", " No", " new", " New"] + [f" {c}" for c in "ABCDEFGHIJKLMNOP"]
_IDS = {w: i for i, w in enumerate(_SPECIAL + _WORDS + ["\n\n"], start=1)}
SYSTEM = "You are the BrainDB Wiki Writer. (gated, with worked examples)"
TOOLS = [{"type": "function", "function": {"name": "edit_wiki_section", "description": "Edit one section.",
                                           "parameters": {"type": "object", "properties": {}}}}]
KWARGS = {"reasoning_effort": "low"}


def _tok(text: str) -> list[int]:
    if text in _WORDS:
        return [_IDS[text]]
    ids, i = [], 0
    while i < len(text):
        mark = next((s for s in _SPECIAL + ["\n\n"] if text.startswith(s, i)), None)
        if mark:
            ids.append(_IDS[mark])
            i += len(mark)
        else:
            ids.append(1000 + ord(text[i]))
            i += 1
    return ids


def _text(ids: list[int]) -> str:
    back = {v: k for k, v in _IDS.items()}
    return "".join(back.get(i, chr(i - 1000) if i >= 1000 else "?") for i in ids)


def _render(messages, tools=None, kwargs=None, add_generation_prompt=False,
            continue_final_message=False) -> list[int]:
    kwargs = kwargs or {}
    thinking = kwargs.get("enable_thinking", True) is not False
    out = "<|im_start|>system\n"
    if thinking:
        out += f"Reasoning effort is set to {kwargs.get('reasoning_effort', 'xhigh')}.\n\n"
    if tools:
        out += "# Tools " + json.dumps(tools, sort_keys=True) + "\n\n"
    rest = messages
    if messages and messages[0]["role"] == "system":
        out += (messages[0]["content"] or "").strip()
        rest = messages[1:]
    out += "<|im_end|>\n"
    for i, m in enumerate(rest):
        content = (m.get("content") or "").strip()
        if m["role"] == "user":
            out += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            out += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}"
            if not (i == len(rest) - 1 and continue_final_message):
                out += "<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")
    return _tok(out)


class _Cache:
    """A hybrid model's prefix cache: a request leaves one checkpoint, at the
    last whole block it ended in, and reuses the longest one that prefixes it."""

    def __init__(self):
        self.points: list[tuple[int, ...]] = []

    def use(self, prompt: list[int]) -> int:
        cached = max((len(p) for p in self.points if tuple(prompt[:len(p)]) == p), default=0)
        if len(prompt) >= BLOCK:
            self.points.append(tuple(prompt[:len(prompt) // BLOCK * BLOCK]))
        return cached


class _Response:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class _FakeVLLM:
    """Answers the fact or member a decision asks about by cues in it:
    identity — CUE_NO (or the probe's other Rowan) is no, CUE_LOWMASS holds
    little probability, anything else yes; placement — PLACE_<letter> or
    PLACE_NEW, CUE_LOWPLACE holds little probability. `probe_ok` False
    inverts the probe's answers; `mismatch` makes the server render the
    job's requests differently from what the gate assembles."""

    def __init__(self, model: str):
        self.model, self.probe_ok, self.mismatch = model, True, False
        self.decisions, self.warms, self.cache = [], [], _Cache()

    def request(self, method, url, **kwargs):
        body = kwargs.get("json") or {}
        if url.endswith("/v1/models"):
            return _Response(200, {"data": [{"id": self.model, "max_model_len": 200_000}]})
        if url.endswith("/tokenize"):
            if "messages" in body:
                ids = _render(body["messages"], body.get("tools"), body.get("chat_template_kwargs"),
                              body.get("add_generation_prompt", False), body.get("continue_final_message", False))
                last, content = body["messages"][-1]["content"], body["messages"][-2]["content"]
                if self.mismatch and last.startswith("[") and "Rowan" not in content:  # a job's request
                    ids = ids + [999]
                return _Response(200, {"tokens": ids})
            return _Response(200, {"tokens": _tok(body["prompt"])})
        assert url.endswith("/v1/completions"), url
        prompt = body["prompt"]
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1,
                 "prompt_tokens_details": {"cached_tokens": self.cache.use(prompt)}}
        if "logprob_token_ids" not in body:
            self.warms.append(prompt)
            return _Response(200, {"choices": [{"text": "x"}], "usage": usage})
        text = _text(prompt)
        lead = text.rsplit("</think>\n\n", 1)[-1]
        anchor = re.match(r"\[(\w+)\]", lead).group(1)
        content = text.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
        item = content.rsplit(f"[{anchor}] id:", 1)[-1].split("\n[", 1)[0]
        mass = 0.98
        if lead.endswith(wiki_gate.ID_LEAD):
            want = "no" if ("CUE_NO" in item or "Rowan Atkinson" in item) else "yes"
            if not self.probe_ok and "Rowan" in item:
                want = "no" if want == "yes" else "yes"
            if "CUE_LOWMASS" in item:
                mass = 0.05
        else:
            want = "new" if "PLACE_NEW" in item else next((c for c in "ABCDEFGHIJKLMNOP" if f"PLACE_{c}" in item), "A")
            if "CUE_LOWPLACE" in item:
                mass = 0.05
        top = {f"token_id:{i}": math.log(mass * (0.97 if _WORDS[i - 5].strip().lower() == want.lower() else 0.01))
               for i in body["logprob_token_ids"]}
        self.decisions.append({"prompt": prompt, "anchor": anchor, "lead": lead, "content": content,
                               "max_tokens": body["max_tokens"], "temperature": body.get("temperature"),
                               "cached": usage["prompt_tokens_details"]["cached_tokens"],
                               "probe": "Rowan bakery" in content})
        return _Response(200, {"choices": [{"logprobs": {"top_logprobs": [top]}}], "usage": usage})


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "vllm_workstation_qwen")
    monkeypatch.setattr(jev.settings, "agent_model", "")
    monkeypatch.setenv("VLLM_API_KEY", "test-secret-key")
    monkeypatch.delenv("CUSTOM_PROFILE", raising=False)
    fake = _FakeVLLM(jev.settings.resolved_agent_model)
    monkeypatch.setattr(jev, "_session", fake)
    monkeypatch.setattr(jev, "BLOCK", BLOCK)
    monkeypatch.setattr(reader, "ROOM", 200)
    monkeypatch.setattr(wiki_gate, "_state", {"server": None, "off_until": 0.0})
    monkeypatch.setattr(wiki_gate, "get_conn", _NoConn)
    return fake


class _NoConn:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


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


def _turn(members, page=SHORT_PAGE) -> reader.Turn:
    """The gated writer's first request as the router builds it: the job,
    with its members, as the user message."""
    head = (f"You maintain the page Calder Labs.\n### Seed member entities for this job\n"
            f"{wiki_gate.members_block(members)}\n### Current wiki body\n{page}\nThe time is now.")
    return reader.Turn([{"role": "system", "content": SYSTEM}, {"role": "user", "content": head.strip()}],
                       TOOLS, KWARGS)


def _prepare(members, page=SHORT_PAGE, long_page=False):
    sections = parse_sections(page)[1] if long_page else []
    return wiki_gate.prepare(_turn(members, page), "Calder Labs", page, members, sections)


def _job(server):
    return [d for d in server.decisions if not d["probe"]]


def test_only_facts_about_the_subject_stay_and_a_left_out_fact_is_rolled_back(server, monkeypatch):
    m1 = _entity("Calder Labs opened a factory")
    cited = _entity("already on the page")
    f_yes = _entity("Calder Labs hired engineers in May, a long fact that recall only previews")
    f_no, f_low, f_yes2 = _entity("CUE_NO Calder Insurance raised premiums"), _entity("CUE_LOWMASS x"), _entity("Calder Labs won a prize")
    page = SHORT_PAGE + f"<!-- section: overview -->\nText [[ref:{cited['id']}]]\n"
    asked = _recall(monkeypatch, [_entity("a page", "wiki"), _entity("calder", "keyword"), m1, cited,
                                  f_yes, f_no, f_low, f_yes2])
    gate = _prepare([m1], page)
    assert [m["id"] for m in gate.members] == [m1["id"]]
    assert gate.found == [f_yes, f_yes2]                                # full rows, not recall's previews
    assert gate.excluded == [f_no["id"], f_low["id"]]
    asked_about = [d["anchor"] for d in _job(server)]
    assert asked_about == ["f1", "f2", "f3", "f4"]                       # only the found facts: never wiki,
    assert asked[0].queries == ["Calder Labs", "calder", "robots"]       # keyword, member or cited fact
    assert asked[0].include_always_on_rules is False
    # a fact left out is rolled back: the next one follows what was kept
    last = _job(server)[-1]["content"]
    assert f_yes["content"] in last and f_no["content"] not in last and "CUE_LOWMASS" not in last
    assert f_yes["content"] in gate.prompt and f_yes2["content"] in gate.prompt
    assert f_no["id"] not in gate.prompt and f_low["id"] not in gate.prompt
    assert "--truncated--" not in gate.prompt
    assert "SUBJECT: Calder Labs\nSUMMARY: battery testing robots maker." in gate.prompt
    assert gate.sections == {}


def test_each_decision_is_one_token_inside_the_writer_s_own_first_request(server, monkeypatch):
    m1 = _entity("Calder Labs opened a factory")
    _recall(monkeypatch, [_entity("Calder Labs hired engineers"), _entity("CUE_NO other firm")])
    turn = _turn([m1])
    gate = wiki_gate.prepare(turn, "Calder Labs", SHORT_PAGE, [m1], [])
    for d in _job(server):
        assert d["max_tokens"] == 1 and d["temperature"] == 0
        # the writer's own rendering (its system prompt, tools and template
        # arguments: the reasoning line is there) up to the content, then the
        # answer turn opened with a closed think block and the lead
        expected = _render([turn.messages[0], {"role": "user", "content": d["content"]},
                            {"role": "assistant", "content": d["lead"]}], TOOLS, KWARGS,
                           continue_final_message=True)
        assert d["prompt"] == expected
        assert "Reasoning effort is set to low." in _text(d["prompt"])
        assert _text(d["prompt"]).endswith(f"<think>\n\n</think>\n\n{d['lead']}")
    # every decision, and then the writer's own first request, reuses the job
    # read once: at least the checkpoint the warm request left in it
    warm = len(server.warms[0]) // BLOCK * BLOCK
    assert warm > len(_render([turn.messages[0]], TOOLS, KWARGS)) and all(d["cached"] >= warm for d in _job(server))
    writer = _render([turn.messages[0], {"role": "user", "content": gate.prompt}], TOOLS, KWARGS,
                     add_generation_prompt=True)
    assert server.cache.use(writer) >= warm


def test_the_members_are_never_asked_about_and_never_leave_the_job(server, monkeypatch):
    _recall(monkeypatch, [])
    members = [_entity("Calder Labs opened a factory"), _entity("CUE_NO Calder Insurance raised premiums")]
    gate = _prepare(members)
    assert gate is not None and gate.members == members
    assert _job(server) == [] and server.warms == []                      # nothing to decide: no request at all
    assert members[1]["id"] in gate.prompt and gate.prompt.endswith("(none)")


def test_on_a_long_page_each_member_and_kept_fact_gets_its_section(server, monkeypatch):
    m1, m2, m3 = _entity("PLACE_B opened a factory"), _entity("PLACE_NEW chief executive resigned"), _entity("CUE_LOWPLACE x")
    f_kept, f_out = _entity("PLACE_C revenue up"), _entity("CUE_NO PLACE_A other firm")
    _recall(monkeypatch, [f_kept, f_out])
    gate = _prepare([m1, m2, m3], LONG_PAGE, long_page=True)
    # the unsure one is left to the writer; a found fact left out is never placed
    assert gate.sections == {m1["id"]: "timeline", m2["id"]: "new", f_kept["id"]: "finances"}
    placed = [d["anchor"] for d in _job(server) if d["lead"].endswith(wiki_gate.PL_LEAD)]
    assert sorted(placed) == ["f1", "m1", "m2", "m3"]


def test_a_page_with_more_sections_than_letters_is_not_placed(server, monkeypatch):
    page = SHORT_PAGE + "".join(f"<!-- section: s{i} -->\nText {i}.\n" for i in range(len(wiki_gate.LETTERS) + 1))
    _recall(monkeypatch, [])
    gate = _prepare([_entity("PLACE_B opened a factory")], page, long_page=True)
    assert gate is not None and gate.sections == {} and _job(server) == []
    assert all(line.startswith("- ") for line in wiki_gate.outline(parse_sections(page)[1]))  # no letters


def test_the_outline_is_lettered_with_each_section_s_first_words():
    lines = wiki_gate.outline(parse_sections(LONG_PAGE)[1])
    assert lines[0].startswith("A. overview: ") and lines[0].endswith(" - Overview text.")
    assert lines[2].startswith("C. finances: ") and lines[2].endswith(" - Revenue 41M.")


def test_a_request_the_server_renders_differently_leaves_the_job_to_the_full_writer(server, monkeypatch):
    _recall(monkeypatch, [_entity("Calder Labs hired engineers")])
    server.mismatch = True
    assert _prepare([_entity("Calder Labs opened a factory")]) is None
    assert wiki_gate.wanted()                                             # the gate itself stays on


def test_the_deadline_leaves_the_job_to_the_full_writer(server, monkeypatch):
    monkeypatch.setattr(wiki_gate, "DEADLINE_S", -1)
    _recall(monkeypatch, [_entity("Calder Labs hired engineers")])
    assert _prepare([_entity("Calder Labs opened a factory")]) is None
    assert _job(server) == []


def test_a_failed_probe_switches_the_gate_off(server, monkeypatch):
    server.probe_ok = False
    _recall(monkeypatch, [])
    assert _prepare([_entity("Calder Labs opened a factory")]) is None
    assert not wiki_gate.wanted()                                         # off until re-probed


def test_off_unless_enabled_on_a_self_hosted_server(server, monkeypatch):
    members = [_entity("Calder Labs opened a factory")]
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", False)
    assert not wiki_gate.wanted() and _prepare(members) is None
    monkeypatch.setattr(jev.settings, "jev_wiki_gate", True)
    monkeypatch.setattr(jev.settings, "llm_profile", "deepinfra")         # a hosted gateway
    assert not wiki_gate.wanted()
    assert server.decisions == [] and server.warms == []


def test_off_under_a_profile_that_replaces_the_writer_prompt(server, monkeypatch, tmp_path):
    (tmp_path / "own").mkdir()
    (tmp_path / "own" / "wiki_writer.replace.md").write_text("Another contract", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("CUSTOM_PROFILE", "own")
    assert not wiki_gate.wanted()


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


def test_the_gated_writer_has_only_the_page_tools_and_its_first_request_is_the_sdk_s():
    import asyncio

    from braindb.agent import agent as agent_mod
    from braindb.agent.tools import first_turn
    gated = agent_mod.get_gated_writer_agent()
    assert {t.name for t in gated.tools} == {"read_wiki_outline", "read_wiki_section", "check_members_cited",
                                             "edit_wiki_section", "delete_wiki_section", "validate_wiki",
                                             "handoff_to_successor", "final_answer"}
    assert gated.instructions == agent_mod._GATED_WRITER_PROMPT and gated is not agent_mod.get_writer_agent()
    turn = asyncio.run(first_turn(gated, "THE JOB"))
    assert turn.messages == [{"content": agent_mod._GATED_WRITER_PROMPT, "role": "system"},
                             {"role": "user", "content": "THE JOB"}]
    assert sorted(t["function"]["name"] for t in turn.tools) == sorted(t.name for t in gated.tools)
