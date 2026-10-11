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
_GUIDE = (_PROMPTS / "jev_judge.md").read_text(encoding="utf-8").strip()
_EXAMPLES = json.loads((_PROMPTS / "jev_examples.json").read_text(encoding="utf-8"))

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


class Unavailable(Exception):
    """The server cannot do JEV: off until re-probed. Its message is ours —
    never a URL, a key or stored text."""


class _ItemError(Exception):
    """The server refused ONE request (400/413/422): that item stays unjudged."""


def clean(text: str) -> str:
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
    text = clean(REF_RE.sub("", "\n".join(p for p in parts if p)))
    if len(text) > ITEM_CHARS:
        # The guide tells the judge a cut item says more after the cut: on
        # dev this kept long items whose answer lies past the window.
        text = text[:ITEM_CHARS] + " ...[continues]"
    return f"[{_TAG.get(entity_type, entity_type)}] {text}"


_THINKING_OFF = {"enable_thinking": False}

_session = requests.Session()
# Every JEV request counts toward one limit, whichever feature sends it (the
# judge, wiki triage, the wiki gate), so together they never put more than
# this many requests on a server that other work shares.
_in_flight = threading.BoundedSemaphore(WORKERS)


class Server:
    """The agent profile's own vLLM endpoint and model: JEV has no endpoint
    settings of its own."""

    def __init__(self) -> None:
        self.base = settings.resolved_base_url.rstrip("/")
        self.root = self.base.removesuffix("/v1")  # /tokenize lives beside /v1
        self.headers = {"Authorization": f"Bearer {settings.resolved_api_key}"}
        self.model = settings.resolved_agent_model

    def _send(self, method: str, url: str, body: dict | None) -> dict:
        for attempt in (1, 2):  # one retry on a dropped connection or a 5xx
            try:
                # Waiting for a free slot is bounded like the request itself,
                # so a full limit costs a caller its timeout, never a hang.
                if not _in_flight.acquire(timeout=REQUEST_TIMEOUT_S):
                    raise requests.Timeout("every JEV request slot stayed busy")
                try:
                    r = _session.request(method, url, json=body, headers=self.headers,
                                         timeout=REQUEST_TIMEOUT_S)
                finally:
                    _in_flight.release()
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 2:
                    raise
                continue
            if r.status_code >= 500 and attempt == 1:
                continue
            if r.status_code in (401, 403, 404):
                raise Unavailable(f"HTTP {r.status_code}")
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

    def render(self, messages: list[dict], template_kwargs: dict | None = _THINKING_OFF,
               **options) -> list[int]:
        """Token ids of `messages` through the server's own chat template,
        by default thinking off (the answer header then closes an empty think
        block); `template_kwargs=None` renders them as for a request that
        sends no template arguments."""
        body = {"model": self.model, "messages": messages}
        if template_kwargs is not None:
            body["chat_template_kwargs"] = template_kwargs
        return self._send("POST", self.root + "/tokenize", {**body, **options})["tokens"]


def _split(server: Server, turns: list[dict]) -> tuple[list[int], list[int]]:
    """`turns` plus one more user turn through the server's template, cut
    where that turn's content goes: the fixed tokens before it, and the
    tokens after it (the end of the turn and the answer header). Two
    renderings that differ only in a one-character final message share
    everything before it, and what follows it is the header."""
    a = server.render(turns + [{"role": "user", "content": "\x01"}], add_generation_prompt=True)
    b = server.render(turns + [{"role": "user", "content": "\x02"}], add_generation_prompt=True)
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return a[:n], a[n + len(server.encode("\x01")):]


def usage_of(out: dict) -> dict:
    """One response's token usage, as the usage ledger counts it."""
    u = out.get("usage") or {}
    return {"input": u.get("prompt_tokens") or 0,
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "output": u.get("completion_tokens") or 0}


def option_ids(server: Server, words: list[str]) -> dict[str, list[int]]:
    """Option word -> its single-token spellings (leading space; as given and capitalised)."""
    out = {}
    for w in words:
        ids = sorted({i[0] for s in (f" {w}", f" {w[:1].upper()}{w[1:]}")
                      if len(i := server.encode(s)) == 1})
        if not ids:
            raise Unavailable(f"option {w!r} is not one token for this model")
        out[w] = ids
    return out


def ask_ids(server: Server, prompt: list[int],
            options: dict[str, list[int]]) -> tuple[dict[str, float], float, dict]:
    """Each option's probability as the next token after `prompt` (token
    ids), the mass the options hold together, and the usage: ONE output
    token, read from the logprobs of the options' ids."""
    ids = sorted({i for v in options.values() for i in v})
    out = server.complete({"prompt": prompt, "max_tokens": 1, "temperature": 0,
                           "logprobs": 0, "logprob_token_ids": ids,
                           "return_tokens_as_token_ids": True})
    try:
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        raw = {o: sum(math.exp(top[f"token_id:{i}"]) for i in v) for o, v in options.items()}
    except (KeyError, IndexError, TypeError):
        raise Unavailable("the server did not return the requested logprobs") from None
    mass = sum(raw.values())
    if not math.isfinite(mass):  # a broken forward pass (NaN logits) answers nothing
        mass = 0.0
    return {o: (p / mass if mass > 0 else 0.0) for o, p in raw.items()}, mass, usage_of(out)


class _Judge:
    """What the probe proved, reused by every judgement: the padded prefix G,
    the answer header + lead, and the yes/no token ids."""

    def __init__(self, server: Server, g: list[int], tail: list[int],
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
            raise Unavailable("the server did not return the requested logprobs") from None
        total = mass["yes"] + mass["no"]
        if not math.isfinite(total):  # a broken forward pass (NaN logits) answers nothing
            total = 0.0
        return (mass["yes"] / total if total > 0 else None), total, usage_of(out)


def _probe() -> _Judge:
    server = Server()
    if server.model not in {m.get("id") for m in server.get("/models").get("data", [])}:
        raise Unavailable("the agent model is not served by this endpoint")
    options = {}
    for answer, spellings in _SPELLINGS.items():
        ids = [server.encode(s) for s in spellings]
        if any(len(i) != 1 for i in ids):
            raise Unavailable("yes/no are not single tokens for this model")
        options[answer] = sorted({i[0] for i in ids})
    turns = [{"role": "system", "content": _GUIDE}]
    for ex in _EXAMPLES:
        turns += [{"role": "user", "content": _user(ex["need"], ex["item"])},
                  {"role": "assistant", "content": f"{LEAD} {ex['useful']}"}]
    # G and the answer header are cut from the template's own rendering.
    fixed, header = _split(server, turns)
    tail = header + server.encode(LEAD)
    sample = _user(_PROBE_NEED, _PROBE_YES)
    reference = server.render(
        turns + [{"role": "user", "content": sample}, {"role": "assistant", "content": LEAD}],
        add_generation_prompt=False, continue_final_message=True)
    if fixed + server.encode(sample) + tail != reference:
        raise Unavailable("the assembled prompt differs from the chat template")
    n, pad = len(fixed), -len(fixed) % BLOCK
    judge_ = _Judge(server, fixed + server.encode("\n") * pad, tail, options)
    # The logprobs must come back and the model must get a clear-cut pair
    # right; these two requests also put G in the cache.
    p_yes, mass_yes, u_yes = judge_.ask(_PROBE_NEED, _PROBE_YES)
    p_no, mass_no, u_no = judge_.ask(_PROBE_NEED, _PROBE_NO)
    if p_yes is None or p_no is None or min(mass_yes, mass_no) < 0.5 or p_yes - p_no < 0.5:
        raise Unavailable("the model does not answer the probe pair reliably")
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
    reason = str(error) if isinstance(error, Unavailable) else type(error).__name__
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
    if not settings.jev_enabled or not settings.self_hosted:
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
        except Unavailable:
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
    need = clean(need or "")[:NEED_CHARS]
    if not settings.jev_enabled or not need or not texts:
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0}
    probs = None
    # A first call waits for the probe; that wait counts against its deadline.
    if available(wait=DEADLINE_S) and (judge_ := _judge) is not None:
        try:
            probs = _judge_all(judge_, need, texts, usage, started + DEADLINE_S)
        except Unavailable as e:
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


# ====================================================================== #
# Choices — one-token picks from a cached prompt family                   #
# ====================================================================== #
#
# The judge's mechanics for any closed question with named options. A prompt
# family's fixed text (a guide + worked examples as chat turns) is rendered
# once into a prefix G padded to whole cache blocks, so the server computes it
# once for every request; a request adds only its own text (encoded once and
# reused by every pick on it), the answer header (thinking off: the template
# writes an empty think block) and a short lead, and the answer is the NEXT
# token, read as probabilities over the options. One output token per
# decision; the model's words are never parsed. `write` is the one exception,
# for the rare piece of genuinely new content a decision needs (e.g. a new
# page's name): a short line under the same prefix.


class Choice:
    """A cached prompt family for one-token picks (see above), on the agent
    profile's own server."""

    def __init__(self, system: str, examples: list[tuple[str, str]]) -> None:
        self.server = server = Server()
        turns = [{"role": "system", "content": system}]
        for user, answer in examples:
            turns += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
        fixed, self.tail = _split(server, turns)   # tail: end of the user turn + answer header
        self.pad = -len(fixed) % BLOCK              # blank lines up to whole cache blocks
        self.g = fixed + server.encode("\n") * self.pad
        self._leads: dict[str, list[int]] = {}      # a family's leads are a few fixed strings

    def encode(self, text: str) -> list[int]:
        """A request's own text as tokens, for every pick on it."""
        return self.server.encode(clean(text))

    def options(self, words: list[str]) -> dict[str, list[int]]:
        """Option word -> its single-token spellings (leading space; as given and capitalised)."""
        return option_ids(self.server, words)

    def _prompt(self, body: list[int], lead: str) -> list[int]:
        if lead not in self._leads:
            self._leads[lead] = self.server.encode(lead)
        return self.g + body + self.tail + self._leads[lead]

    def ask(self, body: list[int], lead: str,
            options: dict[str, list[int]]) -> tuple[dict[str, float], float, dict]:
        """Each option's probability as the next token after the lead, for the
        request text `body` (from `encode`), the mass the options hold
        together, and the usage."""
        return ask_ids(self.server, self._prompt(body, lead), options)

    def write(self, body: list[int], lead: str, max_tokens: int = 16) -> tuple[str, dict]:
        """A short line of new content after the lead, thinking off; "" when
        the line did not end within `max_tokens` (a cut-off line is no answer)."""
        out = self.server.complete({"prompt": self._prompt(body, lead), "max_tokens": max_tokens,
                                    "temperature": 0, "stop": ["\n"]})
        choice = out["choices"][0]
        line = (choice.get("text") or "").strip() if choice.get("finish_reason") == "stop" else ""
        return line, usage_of(out)
