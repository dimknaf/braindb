"""
JEV — closed yes/no judgements read from one token's probability.

BrainDB's agents read a lot to decide a little: a recall returns up to 30
items, and the model then reasons at length about which of them matter.
JEV asks the served model the CLOSED question instead — "is this item
useful for this need?" — one item per request, and reads the answer as the
probability that the next token is "yes" rather than "no". Code only
compares that probability with a threshold: the model still owns the
judgement (no keyword rule, no new similarity path); it states it in one
token instead of a paragraph.

Why it is cheap on a self-hosted vLLM with prefix caching: every request
starts with the same fixed prompt G (`agent/prompts/jev_judge.md` plus the
worked examples in `jev_examples.json`), padded to whole cache blocks, so
the server computes G once and reuses it. A judgement then costs only its
short uncached tail (need + item + answer header) and ONE output token.

SELF-HOSTED vLLM ONLY: it relies on vLLM's token-id prompts, `/tokenize`
and `logprob_token_ids`. Hosted gateways answer 200 and silently drop
`logprob_token_ids`, so `available()` proves every capability on first use
and keeps JEV off otherwise. Every failure fails OPEN: the caller keeps
exactly what it would have had without JEV.

Plain sync functions (requests + a small thread pool); async callers wrap
them in `asyncio.to_thread`.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import requests

# The usage ledger lives in a dependency-free module; JEV adds its tokens to
# the request's ledger exactly like the agent's own LLM calls do.
from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.services.wiki_jobs import REF_RE
from braindb.services.wiki_sections import parse_sections

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"

# The answer every judgement completes, and the spellings read back from
# the logits. Each spelling must be ONE token (`available()` checks).
LEAD = "useful (yes or no):"
_SPELLINGS = {"yes": (" yes", " Yes", "yes", "Yes"), "no": (" no", " No", "no", "No")}

# What the judge reads: the need and each item's own text, capped so need +
# item + answer header stay inside one cache block (see BLOCK).
NEED_CHARS = 500
ITEM_CHARS = 1024

# Prefix-cache block of the hybrid Qwen3.8 server JEV was measured on. That
# server reuses a prefix only up to the last whole block an earlier request
# ENDED in, so G is padded to whole blocks: every judgement's request then
# ends its last whole block exactly at G and reuses all of it. 784 is a
# multiple of vLLM's usual block of 16, so G stays whole-block there too.
BLOCK = 784

# Measured on that server: 30 items took 6.2 s with 4 workers and 3.7 s
# with 8. The pool is module-wide, so concurrent recalls together never put
# more than 8 JEV requests on a server that other work shares.
WORKERS = 8
# Past the deadline, judged items keep their scores and the rest are kept
# unjudged, so a busy server can only cost latency, never items.
DEADLINE_S = 20
REQUEST_TIMEOUT_S = 15
# After a capability failure JEV stays off this long, then probes again.
REPROBE_S = 600

# Item kinds as the judge's guide names them.
_TAG = {"datasource": "document", "thought": "note"}

# A clear-cut pair the model must get right before JEV is trusted.
_PROBE_NEED = "When did Kel Aster open the Rowan bakery?"
_PROBE_YES = "[fact] Kel Aster opened the Rowan bakery on 2 June 2018."
_PROBE_NO = "[fact] The Rowan bakery sells rye bread on Saturdays."


class _Unavailable(Exception):
    """The server cannot do JEV: off until re-probed. Its message is ours —
    never a URL, a key or stored text."""


class _ItemError(Exception):
    """The server refused ONE request (400/413/422): that item stays unjudged."""


def _clean(text: str) -> str:
    """Stored text is data, never prompt structure: '<' becomes '‹', so no
    stored string can open or close a chat turn ('<|im_end|>' inside an item
    would otherwise be read as the real special token). Blank runs are
    squeezed — they cost tokens and say nothing."""
    text = re.sub(r"[ \t]+", " ", text.replace("<", "\u2039"))
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def _user(need: str, text: str) -> str:
    return f"Need: {need}\n\n{text}"


def judge_text(entity_type: str, content: str | None, title: str | None = None,
               summary: str | None = None, disambiguation: str | None = None) -> str:
    """The text the judge reads for one item: its kind and its own words,
    capped at ITEM_CHARS. Never a recall preview — its truncation marker
    would be judged as content. A wiki is read as title, disambiguation,
    summary, then its sections' text: its header only repeats the first
    three, and section markers and citations carry no meaning here."""
    body = content or ""
    if entity_type == "wiki":
        _, sections = parse_sections(body)
        if sections:
            body = "\n".join(s.content for s in sections)
        parts = (title, disambiguation, summary, body)
    else:
        parts = (title, body)
    text = _clean(REF_RE.sub("", "\n".join(p for p in parts if p)))
    if len(text) > ITEM_CHARS:
        # The guide tells the judge a cut item says more after the cut: on
        # dev this kept long items whose answer lies past the window.
        text = text[:ITEM_CHARS] + " ...[continues]"
    return f"[{_TAG.get(entity_type, entity_type)}] {text}"


_session = requests.Session()


class _Server:
    """The agent profile's own vLLM endpoint and model: JEV has no endpoint
    settings of its own."""

    def __init__(self) -> None:
        self.base = settings.resolved_base_url.rstrip("/")
        self.root = self.base.removesuffix("/v1")  # /tokenize lives beside /v1
        self.headers = {"Authorization": f"Bearer {settings.resolved_api_key}"}
        # LiteLLM routes on a provider prefix the server itself never sees.
        provider, _, name = settings.resolved_agent_model.partition("/")
        self.model = name if provider in ("openai", "hosted_vllm") and name \
            else settings.resolved_agent_model

    def _send(self, method: str, url: str, body: dict | None) -> dict:
        for attempt in (1, 2):  # one retry on a dropped connection or a 5xx
            try:
                r = _session.request(method, url, json=body, headers=self.headers,
                                     timeout=REQUEST_TIMEOUT_S)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                continue
            if r.status_code >= 500 and attempt == 1:
                continue
            if r.status_code in (401, 403, 404):
                raise _Unavailable(f"HTTP {r.status_code}")
            if r.status_code in (400, 413, 422):
                raise _ItemError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        raise AssertionError("unreachable")

    def get(self, path: str) -> dict:
        return self._send("GET", self.base + path, None)

    def complete(self, body: dict) -> dict:
        return self._send("POST", self.base + "/completions", {"model": self.model, **body})

    def encode(self, text: str) -> list[int]:
        return self._send("POST", self.root + "/tokenize", {
            "model": self.model, "prompt": text, "add_special_tokens": False})["tokens"]

    def render(self, messages: list[dict], **options) -> list[int]:
        """Token ids of `messages` through the server's own chat template,
        thinking off (the answer header then closes an empty think block)."""
        return self._send("POST", self.root + "/tokenize", {
            "model": self.model, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}, **options})["tokens"]


class _Judge:
    """What the probe proved, reused by every judgement: the padded prefix G,
    the answer header + lead, and the yes/no token ids."""

    def __init__(self, server: _Server, g: list[int], tail: list[int],
                 options: dict[str, list[int]]) -> None:
        self.server, self.g, self.tail, self.options = server, g, tail, options
        self.ids = sorted(i for ids in options.values() for i in ids)

    def ask(self, need: str, text: str) -> tuple[float | None, float, dict]:
        """P(yes) for one item, the yes+no probability mass, and the usage."""
        prompt = self.g + self.server.encode(_user(need, text)) + self.tail
        out = self.server.complete({
            "prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": 0,
            "logprob_token_ids": self.ids, "return_tokens_as_token_ids": True})
        try:
            top = out["choices"][0]["logprobs"]["top_logprobs"][0]
            mass = {answer: sum(math.exp(top[f"token_id:{i}"]) for i in ids)
                    for answer, ids in self.options.items()}
        except (KeyError, IndexError, TypeError):
            raise _Unavailable("the server did not return the requested logprobs") from None
        u = out.get("usage") or {}
        usage = {"input": u.get("prompt_tokens") or 0,
                 "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                 "output": u.get("completion_tokens") or 0}
        total = mass["yes"] + mass["no"]
        return (mass["yes"] / total if total > 0 else None), total, usage


def _probe() -> _Judge:
    server = _Server()
    if server.model not in {m.get("id") for m in server.get("/models").get("data", [])}:
        raise _Unavailable("the agent model is not served by this endpoint")
    options = {}
    for answer, spellings in _SPELLINGS.items():
        ids = [server.encode(s) for s in spellings]
        if any(len(i) != 1 for i in ids):
            raise _Unavailable("yes/no are not single tokens for this model")
        options[answer] = sorted({i[0] for i in ids})
    turns = [{"role": "system",
              "content": (_PROMPTS / "jev_judge.md").read_text(encoding="utf-8").strip()}]
    for ex in json.loads((_PROMPTS / "jev_examples.json").read_text(encoding="utf-8")):
        turns += [{"role": "user", "content": _user(ex["need"], ex["item"])},
                  {"role": "assistant", "content": f"{LEAD} {ex['useful']}"}]
    # G and the answer header are cut from the template's own rendering: two
    # renderings that differ only in a one-character final message share
    # everything before it (G), and what follows it is the header.
    a = server.render(turns + [{"role": "user", "content": "\x01"}], add_generation_prompt=True)
    b = server.render(turns + [{"role": "user", "content": "\x02"}], add_generation_prompt=True)
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    tail = a[n + len(server.encode("\x01")):] + server.encode(LEAD)
    sample = _user(_PROBE_NEED, _PROBE_YES)
    reference = server.render(
        turns + [{"role": "user", "content": sample}, {"role": "assistant", "content": LEAD}],
        add_generation_prompt=False, continue_final_message=True)
    if a[:n] + server.encode(sample) + tail != reference:
        raise _Unavailable("the assembled prompt differs from the chat template")
    pad = -n % BLOCK
    judge_ = _Judge(server, a[:n] + server.encode("\n") * pad, tail, options)
    # The logprobs must come back and the model must get a clear-cut pair
    # right; these two requests also put G in the cache.
    p_yes, mass_yes, u_yes = judge_.ask(_PROBE_NEED, _PROBE_YES)
    p_no, mass_no, u_no = judge_.ask(_PROBE_NEED, _PROBE_NO)
    if p_yes is None or p_no is None or min(mass_yes, mass_no) < 0.5 or p_yes - p_no < 0.5:
        raise _Unavailable("the model does not answer the probe pair reliably")
    # The prompt files are sized so G lands just under whole blocks (the
    # measured prompt pads 143 tokens). A pad of a quarter block or more
    # means they were edited and should be re-sized: the pad is blank lines
    # the model reads before every need.
    logger.log(logging.WARNING if pad >= BLOCK // 4 else logging.INFO,
               "JEV on: fixed prefix of %d tokens + %d padding; the probe used %d uncached tokens",
               n, pad, sum(u["input"] - u["cached"] for u in (u_yes, u_no)))
    return judge_


_judge: _Judge | None = None
_off_until = 0.0
_probing: threading.Event | None = None  # set when the running probe ends
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="jev")


def _switch_off(error: Exception) -> None:
    global _judge, _off_until
    _off_until = time.monotonic() + REPROBE_S  # first, so no caller re-probes in between
    _judge = None
    reason = str(error) if isinstance(error, _Unavailable) else type(error).__name__
    logger.warning("JEV off for %d s: %s", REPROBE_S, reason)


def _run_probe(done: threading.Event) -> None:
    global _judge, _probing
    try:
        _judge = _probe()
    except Exception as e:  # noqa: BLE001 - any failure keeps JEV off, never a caller down
        _switch_off(e)
    finally:
        with _lock:
            _probing = None
        done.set()


def available(wait: float = 0.0) -> bool:
    """True when JEV is enabled on a self-hosted endpoint and the server has
    proved every capability it needs. The probe runs once, on first use, in
    its own thread; a caller waits for it at most `wait` seconds, so a slow
    server costs a bounded delay, never a stuck request. A failed probe logs
    one warning and keeps JEV off for REPROBE_S."""
    global _probing
    if not settings.jev_enabled or not settings.resolved_base_url:
        return False
    if _judge is not None:
        return True
    with _lock:
        if _judge is None and _probing is None and time.monotonic() >= _off_until:
            _probing = threading.Event()
            threading.Thread(target=_run_probe, args=(_probing,), name="jev-probe",
                             daemon=True).start()
        probing = _probing
    if probing is not None:
        probing.wait(wait)
    return _judge is not None


def _judge_all(judge_: _Judge, need: str, texts: list[str], usage: dict,
               deadline: float) -> list[float | None]:
    probs: list[float | None] = [None] * len(texts)

    def collect(i: int, future) -> None:
        try:
            p, _, u = future.result()
        except _ItemError:
            return  # refused: this item stays unjudged
        probs[i] = p
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]

    # The first request goes alone: on a cold cache it computes G once, and
    # the rest then reuse it instead of all computing it at the same time.
    first = _pool.submit(judge_.ask, need, texts[0])
    if not wait([first], timeout=max(0.0, deadline - time.monotonic())).done:
        first.cancel()
        raise TimeoutError("the first JEV request missed the deadline")
    collect(0, first)  # a transport failure here fails the whole call open
    rest = {_pool.submit(judge_.ask, need, t): i for i, t in enumerate(texts[1:], start=1)}
    done, pending = wait(rest, timeout=max(0.0, deadline - time.monotonic()))
    for f in pending:
        f.cancel()
    for f in done:
        try:
            collect(rest[f], f)
        except _Unavailable:
            raise
        except Exception:  # noqa: BLE001 - a lost item stays unjudged (kept)
            pass
    return probs


def judge(need: str, texts: list[str]) -> list[float | None] | None:
    """P(useful) for each item text (built with `judge_text`), in order, or
    None for an item left unjudged. Returns None when JEV is off, the need
    is blank, or the server failed — the caller then keeps exactly what it
    had. Token usage, fail-opens and unjudged items go to the request's
    usage ledger, so a run that silently fell back is visible. Requests
    still running past the deadline are not counted."""
    need = _clean(need or "")[:NEED_CHARS]
    if not settings.jev_enabled or not need or not texts:
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0}
    probs = None
    # A first call waits for the probe; that wait counts against its deadline.
    if available(wait=DEADLINE_S) and (judge_ := _judge) is not None:
        try:
            probs = _judge_all(judge_, need, texts, usage, started + DEADLINE_S)
        except _Unavailable as e:
            _switch_off(e)
        except Exception as e:  # noqa: BLE001 - the first request failed: unreachable or overloaded
            logger.warning("JEV failed open: %s", type(e).__name__)
    unjudged = sum(p is None for p in probs) if probs is not None else 0
    add_usage("jev", calls=1, failopen=int(probs is None), unjudged=unjudged, **usage)
    if probs is not None:
        logger.info("JEV judged %d/%d items in %.1fs (in=%d cached=%d out=%d)",
                    len(texts) - unjudged, len(texts), time.monotonic() - started,
                    usage["input"], usage["cached"], usage["output"])
    return probs
