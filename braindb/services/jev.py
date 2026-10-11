"""
JEV — closed yes/no judgements read from one token's probability.

BrainDB's agents read a lot to decide a little: a recall returns up to 30
items, and the model then reasons at length about which of them matter.
JEV asks the served model the CLOSED question instead — "is this item
useful for this need?" — one item per request, and reads the answer as the
probability that the answer token is "yes" rather than "no". Code only
compares that probability with a threshold: the model still owns the
judgement (no keyword rule, no new similarity path); it states it in one
token instead of a paragraph.

Every request is a standard OpenAI chat completion through the openai
library, on the agent profile's own endpoint and model: thinking off with
`reasoning_effort="none"`, `temperature` 0, `logprobs` with `top_logprobs`.
The answer is the first option token the model writes, either first or
right after it restates the question's line (some models do, as the worked
examples write it); its probability comes from that token's top_logprobs,
or from its own logprob when the server returns no alternatives. A request
that repeats an earlier one's beginning is served from the server's prefix
cache, so the fixed guide and examples, or a conversation so far, are
computed once.

Any OpenAI-compatible server that returns logprobs and can turn thinking
off can serve it; `available()` proves that on first use (a clear-cut pair
the model must judge right) and keeps JEV off otherwise. Every failure
fails OPEN: the caller keeps exactly what it would have had without JEV.

Plain sync functions (the openai client + a small thread pool); async
callers wrap them in `asyncio.to_thread`.
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

import openai
from openai import OpenAI

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

# The line every judgement answers, as the worked examples write it.
LEAD = "useful (yes or no):"
YES_NO = ["yes", "no"]

# What the judge reads: the need and each item's own text, capped.
NEED_CHARS = 500
ITEM_CHARS = 1024

# Output tokens a decision may take: its answer, after the question's line
# when the model restates it first.
ANSWER_TOKENS = 24
# The most alternatives the standard returns per token.
TOP_LOGPROBS = 20
# Below this share of the answer token the options hold too little
# probability: the model meant to write something else (thinking, prose),
# and there is no answer. A reply's first token is free: a tool call, the
# end of the turn or the restated question line take some of it too, so a
# clear answer can hold under half of it (measured: 0.44 to 0.99 for clear
# decisions on a self-hosted Qwen; ~0 for a model that thinks first).
MIN_MASS = 0.3

# Measured on a self-hosted server: 30 items took 6.2 s with 4 workers and
# 3.7 s with 8. The pool is module-wide, so concurrent recalls together
# never put more than 8 JEV requests on a server that other work shares.
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
    text = re.sub(r"[ \t]+", " ", text.replace("<", "‹"))
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


# Every JEV request counts toward one limit, whichever feature sends it (the
# judge, wiki triage, the wiki gate, the reader, hops), so together they
# never put more than this many requests on a server that other work shares.
_in_flight = threading.BoundedSemaphore(WORKERS)


class Server:
    """The agent profile's own endpoint and model, through the openai
    library: JEV has no endpoint settings of its own."""

    def __init__(self) -> None:
        self.model = settings.resolved_agent_model
        self.client = OpenAI(base_url=settings.resolved_base_url,
                             api_key=settings.resolved_api_key or "missing-api-key",
                             timeout=REQUEST_TIMEOUT_S, max_retries=1)
        # Thinking off by the standard switch, until the server refuses it.
        self.effort = True

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = ANSWER_TOKENS, stop: list[str] | None = None):
        """One chat completion, thinking off, with the logprobs of what it
        writes. With `tools` (an agent's own, so the request starts as its
        requests do) the reply must be text."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "logprobs": True, "top_logprobs": TOP_LOGPROBS}
        if tools:
            body.update(tools=tools, tool_choice="none")
        if stop:
            body["stop"] = stop
        # Waiting for a free slot is bounded like the request itself, so a
        # full limit costs a caller its timeout, never a hang.
        if not _in_flight.acquire(timeout=REQUEST_TIMEOUT_S):
            raise TimeoutError("every JEV request slot stayed busy")
        try:
            return self._create(body)
        finally:
            _in_flight.release()

    def _create(self, body: dict):
        try:
            return self.client.chat.completions.create(
                **body, **({"reasoning_effort": "none"} if self.effort else {}))
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise Unavailable(f"HTTP {e.status_code}") from None
        except openai.APIStatusError as e:
            if e.status_code not in (400, 413, 422):
                raise
            if self.effort:
                # A server that rejects the standard switch: once more without
                # it. If the model then thinks, no option comes first and the
                # probe keeps JEV off.
                try:
                    out = self.client.chat.completions.create(**body)
                except openai.APIStatusError:
                    raise _ItemError(f"HTTP {e.status_code}") from None
                self.effort = False
                logger.info("JEV: the server refuses reasoning_effort; asking without it")
                return out
            raise _ItemError(f"HTTP {e.status_code}") from None


def usage_of(out) -> dict:
    """One response's token usage, as the usage ledger counts it."""
    u = getattr(out, "usage", None)
    details = getattr(u, "prompt_tokens_details", None)
    return {"input": getattr(u, "prompt_tokens", 0) or 0,
            "cached": getattr(details, "cached_tokens", 0) or 0,
            "output": getattr(u, "completion_tokens", 0) or 0}


def _option(token: str, options: list[str]) -> str | None:
    """The option a written token answers: the option it spells (case and
    spaces aside), or the one option it begins."""
    t = token.strip().lower()
    if not t:
        return None
    for o in options:
        if o.lower() == t:
            return o
    starts = [o for o in options if len(t) >= 2 and o.lower().startswith(t)]
    return starts[0] if len(starts) == 1 else None


def answer(out, options: list[str], lead: str) -> tuple[dict[str, float], float]:
    """Each option's probability at the answer in a reply, and the mass the
    options hold together there. The answer is the first option token
    written either first or right after the line `lead` ends with (a model
    may restate the question's line first). Its alternatives give the
    probabilities; a server that returns none gives the chosen token's own,
    the rest shared by the other options. No answer: all 0, mass 0."""
    choices = getattr(out, "choices", None) or []
    logprobs = getattr(choices[0], "logprobs", None) if choices else None
    tokens = getattr(logprobs, "content", None) or []
    line = lead.strip().splitlines()[-1].strip() if lead.strip() else ""
    said = ""
    for tok in tokens:
        if not said.strip() or (line and said.rstrip().endswith(line)):
            chosen = _option(tok.token, options)
            if chosen is not None:
                raw = dict.fromkeys(options, 0.0)
                if tok.top_logprobs:
                    for alt in tok.top_logprobs:
                        o = _option(alt.token, options)
                        if o is not None and alt.logprob is not None:
                            raw[o] += math.exp(alt.logprob)
                elif tok.logprob is not None:
                    p = math.exp(tok.logprob)
                    raw[chosen] = p
                    for o in options:
                        if o != chosen:
                            raw[o] = (1.0 - p) / (len(options) - 1)
                mass = raw[chosen] if not tok.top_logprobs else sum(raw.values())
                total = sum(raw.values())
                if not math.isfinite(total) or not math.isfinite(mass) or total <= 0:
                    return dict.fromkeys(options, 0.0), 0.0  # a broken forward pass answers nothing
                return {o: p / total for o, p in raw.items()}, mass
        said += tok.token
    return dict.fromkeys(options, 0.0), 0.0


def ask(server: Server, messages: list[dict], question: str, options: list[str],
        tools: list[dict] | None = None) -> tuple[dict[str, float], float, dict]:
    """`question` asked right after the last of `messages`: each option's
    probability at the answer, the mass the options hold there, and the
    usage. After a tool's output the question ends that output, nothing else
    added: a template that replays reasoning keys it to the last user
    message, so a new user turn would re-render the agent's earlier turns and
    lose the server's prefix cache. After a user's message (a job with no
    turns of the model's yet) it is its own user turn, which every model
    answers directly. Either way the question line comes after one line
    naming the options, which keeps a model that answers with tool calls
    from reaching for one. The reply stops at the end of its first line."""
    question = f"Answer with one word: {', '.join(options[:-1])} or {options[-1]}.\n{question}"
    last = messages[-1]
    if last.get("role") == "tool":
        text = last.get("content") or ""
        asked = messages[:-1] + [{**last, "content": f"{text}{'' if text.endswith(chr(10)) else chr(10)}{question}"}]
    else:
        asked = messages + [{"role": "user", "content": question}]
    out = server.chat(asked, tools=tools, stop=["\n"])
    probs, mass = answer(out, options, question)
    return probs, mass, usage_of(out)


class _Judge:
    """The guide and its worked examples as chat turns, reused by every
    judgement (the server caches them after the first)."""

    def __init__(self, server: Server) -> None:
        self.server = server
        self.turns = [{"role": "system", "content": _GUIDE}]
        for ex in _EXAMPLES:
            self.turns += [{"role": "user", "content": f"{_user(ex['need'], ex['item'])}\n{LEAD}"},
                           {"role": "assistant", "content": ex["useful"]}]

    def ask(self, need: str, text: str) -> tuple[float | None, float, dict]:
        """P(yes) for one item, the yes+no probability mass, and the usage."""
        out = self.server.chat(self.turns + [{"role": "user", "content": f"{_user(need, text)}\n{LEAD}"}],
                               stop=["\n"])
        probs, mass = answer(out, YES_NO, LEAD)
        return (probs["yes"] if mass > 0 else None), mass, usage_of(out)


def _probe() -> _Judge:
    judge_ = _Judge(Server())
    # The logprobs must come back and the model must get a clear-cut pair
    # right; these two requests also put the guide in the cache.
    p_yes, mass_yes, u_yes = judge_.ask(_PROBE_NEED, _PROBE_YES)
    p_no, mass_no, u_no = judge_.ask(_PROBE_NEED, _PROBE_NO)
    if p_yes is None or p_no is None or min(mass_yes, mass_no) < MIN_MASS or p_yes - p_no < 0.5:
        raise Unavailable("the model does not answer the probe pair reliably")
    logger.info("JEV on: the probe used %d uncached tokens",
                sum(u["input"] - u["cached"] for u in (u_yes, u_no)))
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
    """True when the recall filter is enabled and the server has proved
    every capability it needs. The probe runs once, on first use, in its own
    thread; a caller waits for it at most `wait` seconds, so a slow server
    costs a bounded delay, never a stuck request. A failed probe logs one
    warning and keeps JEV off for REPROBE_S."""
    global _probing
    if not settings.jev_enabled:
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

    # The first request goes alone: on a cold cache it computes the guide
    # once, and the rest then reuse it instead of all computing it at once.
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
# Choices — picks from a cached prompt family                             #
# ====================================================================== #
#
# The judge's mechanics for any closed question with named options. A prompt
# family's fixed text (a guide + worked examples as chat turns) starts every
# request, so the server computes it once. The model answers a request's own
# text the way the examples answer theirs — e.g. "action: attach\nwiki: A" —
# once, and each decision is read where its lead ends in that answer ("action:"
# then "wiki:"), as probabilities over its options. The model's words are
# never parsed but for `name`, the rare piece of genuinely new content a
# decision needs (a new page's name).


class Choice:
    """A cached prompt family for picks (see above), on the agent profile's
    own server."""

    def __init__(self, system: str, examples: list[tuple[str, str]], answer_tokens: int = 128) -> None:
        self.server = Server()
        self.turns = [{"role": "system", "content": system}]
        for user, reply in examples:
            self.turns += [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        self.answer_tokens = answer_tokens

    def encode(self, text: str) -> str:
        """A request's own text as the requests carry it."""
        return clean(text)

    def options(self, words: list[str]) -> dict[str, str]:
        """The options a decision reads, by word."""
        return {w: w for w in words}

    def reply(self, body: str):
        """The model's answer to the request text `body`, thinking off: read
        with `ask` and `name` for every decision on it."""
        return self.server.chat(self.turns + [{"role": "user", "content": body}],
                                max_tokens=self.answer_tokens)

    @staticmethod
    def ask(reply, lead: str, options: dict[str, str]) -> tuple[dict[str, float], float]:
        """Each option's probability where `lead` ends in `reply`, and the
        mass the options hold together there."""
        return answer(reply, list(options), lead)

    @staticmethod
    def name(reply, lead: str) -> str:
        """The line `reply` writes after `lead` (a new page's name), or ""
        when it writes none or the reply was cut off inside it."""
        if not reply.choices:
            return ""
        text = reply.choices[0].message.content or ""
        line = lead.strip().splitlines()[-1].strip()
        at = text.find(line)
        if at < 0:
            return ""
        rest = text[at + len(line):]
        if "\n" not in rest and reply.choices[0].finish_reason == "length":
            return ""  # a cut-off line is no answer
        return rest.split("\n", 1)[0].strip()
