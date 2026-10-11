"""
The adaptive reader, nested in the agent's own conversation: a long body is
read once, piece by piece, and only the pieces its question needs stay in
the agent's context.

Without it, `get_entity` on a body longer than one tool output returns its
first 8,000 chars, and the agent pages through the rest (a turn per slice)
or hands slices to subagents that read them again; whatever it read rides
through every later turn. With JEV_READER on, when an /agent/query request
(or one of its subagents) starts reading such a body, it is read the
adaptive reader's way: one piece at a time, each followed by ONE token,
"[s.p] | useful (yes or no):" — thinking off, the answer read from the
logprobs of its yes/no token. A piece judged useful stays, and reading
continues through it; one that is not is rolled back, and the next piece
follows what was kept before it. After a piece that stays, one more token,
"[s.p] | enough to answer (yes or no):": yes, and reading stops there (a
lookup whose answer is kept need not read on); the rest is listed as not
read. Nothing is summarised, and nothing kept is trimmed.

Nested: the tool output being built IS the reader's memory, inside the
agent's own conversation. Each decision request is the agent's next request
up to that output — the messages and tools the SDK will send — with the
question at the end of the output, where it never stays (services/jev.py
`ask`: a standard chat completion, thinking off, the answer read from the
logprobs of the yes/no token the model writes). Successive requests
share everything up to the pieces kept, so the server's prefix cache computes
each piece once (but for the tail past its last cache block), and the
agent's next turn continues from that cache.

The agent gets plain text (get_entity's JSON would escape every newline): a
header saying what was read and for which question, the kept pieces, and a
note of what was left out, as char spans it can still read with get_entity's
offset/limit — which this module never touches. Anything that goes wrong
gives exactly the output the agent would have had without it. The agents
that read carry worked examples of the question in their system prompt
(agent/prompts/reader_system.md): nothing else can be cached before it.

Replayed inside the real agent's turn on 20 long pages and documents of the
largest personal memories (4-53K chars, two needs each, blind two-judge
gold; measured with token-level requests on a self-hosted server, before the
requests moved to the standard chat API): on the held-out half the
top-judged piece was one the judges call essential for 20 of 20 needs;
lookups kept their answer at 30% of the page on average, gathers 32 of 33
essential pieces at 78%.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.services import jev
from braindb.services.wiki_sections import parse_sections

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_HEADER = (_PROMPTS / "reader_header.md").read_text(encoding="utf-8").strip()
# The probe asks inside the system prompt the agents that read carry (agent._reading_prompt).
_SYSTEM = ((_PROMPTS / "system_prompt.md").read_text(encoding="utf-8").rstrip() + "\n\n"
           + (_PROMPTS / "reader_system.md").read_text(encoding="utf-8"))

LEAD = "| useful (yes or no):"  # asked after each piece, behind the piece's anchor
ENOUGH_LEAD = "| enough to answer (yes or no):"  # asked after each piece that stays
STOP_TOOLS = ("final_answer", "handoff_to_successor")  # a turn that calls one has no next turn

KEEP = 0.5          # P(yes) from which a piece stays (chosen on dev needs over long pages)
ENOUGH = 0.95       # P(yes) from which what was kept answers the question: reading stops (lookups stop
                    # early on a related piece below it; chosen on dev needs over long pages)
MIN_MASS = jev.MIN_MASS  # below this the options hold too little probability: the piece stays
PIECE_CHARS = 3000  # a piece is about this long (about one cache block of a hybrid server)...
MIN_PIECE_CHARS = 600  # ...and a shorter tail joins the piece before it
QUESTION_CHARS = jev.NEED_CHARS  # a longer query is an instruction, not a reading question
WINDOW_TOKENS = 128_000  # the agent's prompt is kept below this (smaller than the servers' own windows)...
ROOM = 8192         # ...with this many tokens left for the agent's later turns
MAX_PIECES = 80     # a body longer than this many pieces is not read this way
DEADLINE_S = 120    # reading stops here; the rest is listed as not read
SLICE = 8000        # the note splits what was left out into spans of at most this many chars
OUTLINE_PARTS = 40  # parts named in the header's outline
CHARS_PER_TOKEN = 3.8  # a first estimate, refined on each body as it is read


@dataclass
class Turn:
    """The agent's next request up to one tool call's output, as plain data:
    its messages (the last one is that call's tool message, or the wiki
    gate's user message) and its tools."""
    messages: list[dict]
    tools: list[dict]


_lock = threading.Lock()
_state: dict = {"server": None, "off_until": 0.0}


def wanted() -> bool:
    """Whether the reader serves agent runs now."""
    return settings.jev_reader and time.monotonic() >= _state["off_until"]


def reading_question(need: str | None, query: str | None) -> str:
    """What a request's long reads are read for: its stated need, else its
    query when short enough to judge pieces against (QUESTION_CHARS); ""
    when neither (the ingest watcher's long extraction prompts are
    instructions, not questions)."""
    need = " ".join(jev.clean(need or "").split())  # one line: the header states it on one
    if need:
        return need[:QUESTION_CHARS]
    query = " ".join(jev.clean(query or "").split())
    return query if len(query) <= QUESTION_CHARS else ""


def first_call(response: list, call_id: str) -> bool:
    """Whether `call_id` is the first tool call of its turn, in a turn that a
    next turn follows: only the first call's output leads the next prompt
    (the later ones' outputs are still being made while it runs), and a
    turn that also finalises or hands off has no next turn."""
    calls = [it for it in response if isinstance(it, dict) and it.get("type") == "function_call"]
    return (bool(calls) and calls[0].get("call_id") == call_id
            and not any(c.get("name") in STOP_TOOLS for c in calls))


def _switch_off(error: Exception) -> None:
    with _lock:
        _state.update(off_until=time.monotonic() + jev.REPROBE_S, server=None)
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV reader off for %d s: %s", jev.REPROBE_S, reason)


def serves() -> bool:
    """Whether this server has passed the reader's probe, probing it now if
    it has not (agent._reading_prompt: only then does an agent carry the
    reading examples). Blocking on first use: call it off the event loop."""
    if time.monotonic() < _state["off_until"]:
        return False
    try:
        ready()
    except jev.Unavailable as e:
        _switch_off(e)
        return False
    return True


# ---------------------------------------------------------------- the turn


def with_content(turn: Turn, content: str) -> list[dict]:
    """The turn's messages with `content` as its last message's (the tool's
    output, or the wiki gate's user message)."""
    return turn.messages[:-1] + [{**turn.messages[-1], "content": content}]


def _lead(anchor: str) -> str:
    """The closed question asked after a piece, behind its anchor (the
    question it is asked for is stated once, in the header)."""
    return f"[{anchor}] {LEAD}"


# --------------------------------------------------------------- the body


_HEADING = re.compile(r"^#{1,6}[ \t]+\S.*$", re.M)


def _parts(entity_type: str | None, body: str) -> list[tuple[str, int]]:
    """(name, start) of each part of the body, in order, the first at 0: a
    wiki's header and `<!-- section -->` sections; any other body's markdown
    headings, else the whole body as one part."""
    out: list[tuple[str, int]] = []
    if entity_type == "wiki":
        header, sections = parse_sections(body)
        end = len(body)
        for s in reversed(sections):  # a section ends where the next one's marker starts
            end = body.rfind("<!--", 0, end - len(s.content))
            if end < 0:  # not the shape parse_sections reads: fall back to headings
                out = []
                break
            out.insert(0, (s.name, end))
        if out:
            out.insert(0, ("header", 0))
    if not out:
        out = [(" ".join(m.group(0).lstrip("#").split())[:60], m.start())
               for m in _HEADING.finditer(body)]
        if not out or out[0][1] > 0:
            out.insert(0, ("text", 0))
    starts: list[tuple[str, int]] = []
    for name, start in out:  # an empty part gives way to the one that starts where it does
        if starts and starts[-1][1] == start:
            starts[-1] = (name, start)
        else:
            starts.append((name, start))
    return starts


def _cuts(body: str, parts: list[tuple[str, int]]) -> list[list[int]]:
    """Where a piece may end, strongest first: where a part starts, where a
    paragraph starts, where a line starts, after a sentence, before a space."""
    return [[start for _, start in parts[1:]],
            [m.end() for m in re.finditer(r"\n[ \t]*\n", body)],
            [m.end() for m in re.finditer(r"\n", body)],
            [m.end() for m in re.finditer(r"[.!?][\"')\]]*(?=\s)", body)],
            [m.start() for m in re.finditer(r"(?<=\S)\s", body)]]


def _piece(frame: str, body: str, a: int, b: int) -> str:
    """The piece [a, b) as the tool output carries it: its frame, its text,
    one newline (so the next frame or the note starts a fresh token)."""
    return f"{frame}{body[a:b].strip()}\n"


def _cut(body: str, cuts: list[list[int]], a: int) -> int:
    """Where the piece starting at `a` ends: the strongest boundary from
    three quarters of PIECE_CHARS on, up to a quarter past it; the rest of
    the body when it ends inside that; else PIECE_CHARS on."""
    lo, hi = a + PIECE_CHARS * 3 // 4, a + PIECE_CHARS * 5 // 4
    if hi >= len(body):
        return len(body)
    for positions in cuts:
        i = bisect_left(positions, lo)
        if i < len(positions) and positions[i] <= hi:
            return positions[i]
    return a + PIECE_CHARS


def _part_at(parts: list[tuple[str, int]], pos: int) -> int:
    return bisect_right([start for _, start in parts], pos) - 1


def _header(row: dict, body: str, parts: list[tuple[str, int]], question: str) -> str:
    ends = [start for _, start in parts[1:]] + [len(body)]
    outline = "; ".join(f"{name} {start}-{end}" for (name, start), end
                        in zip(parts[:OUTLINE_PARTS], ends))
    if len(parts) > OUTLINE_PARTS:
        outline += f"; and {len(parts) - OUTLINE_PARTS} more"
    title = f' "{" ".join(row["title"].split())}"' if row.get("title") else ""
    return _HEADER.format(kind=row.get("entity_type") or "entity", id=row.get("id"), title=title,
                          chars=len(body), question=question, outline=outline) + "\n"


def _spans(spans: list[tuple[int, int]], parts: list[tuple[str, int]]) -> str:
    """Adjacent spans joined, then cut to at most SLICE chars, each with the
    part it starts in."""
    runs: list[list[int]] = []
    for a, b in spans:
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    out = []
    for a, b in runs:
        for s in range(a, b, SLICE):
            out.append(f"{s}-{min(b, s + SLICE)} ({parts[_part_at(parts, s)][0]})")
    return ", ".join(out)


def _note(body: str, parts: list[tuple[str, int]], kept: list[tuple[int, int]],
          dropped: list[tuple[int, int]], unread: int | None, why: str) -> str:
    """What the tool output ends with: what was kept, and every span left
    out, so the agent can still read any of it."""
    lines = [f"Kept {len(kept)} of {len(kept) + len(dropped)} pieces read "
             f"({sum(b - a for a, b in kept)} of {len(body)} chars)."]
    if dropped:
        lines.append(f"Not kept: {_spans(dropped, parts)}.")
    if unread is not None:
        lines.append(f"Not read ({why}): {_spans([(unread, len(body))], parts)}.")
    if dropped or unread is not None:
        lines.append("Read any of these with get_entity(entity_id, offset, limit).")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------- probe and read


_PROBE_QUESTION = "When did Kel Aster open the Rowan bakery?"
_PROBE_TURN = Turn(
    messages=[{"role": "system", "content": _SYSTEM},
              {"role": "user", "content": _PROBE_QUESTION},
              {"role": "assistant", "content": None, "tool_calls": [{
                  "id": "call_probe", "type": "function",
                  "function": {"name": "get_entity", "arguments": '{"entity_id": "probe"}'}}]},
              {"role": "tool", "tool_call_id": "call_probe", "content": ""}],
    tools=[{"type": "function", "function": {
        "name": "get_entity", "description": "Fetch one entity by ID.",
        "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}},
                       "required": ["entity_id"]}}}])
_PROBE_PIECES = ("Kel Aster opened the Rowan bakery on 2 June 2018, after two years of renovation.",
                 "The Rowan bakery sells rye bread on Saturdays and closes on Mondays.")


def _probe(server: jev.Server) -> None:
    """The model must keep the piece that answers a clear-cut question and
    drop the one that does not, asked the reader's way."""
    body = "\n\n".join(_PROBE_PIECES)
    row = {"entity_type": "datasource", "id": "probe", "title": "Rowan bakery", "content": body}
    header = _header(row, body, [("text", 0)], _PROBE_QUESTION)
    p = []
    for i, text in enumerate(_PROBE_PIECES):
        content = f"{header}[0.{i}] (text)\n{text}\n"
        probs, mass, _ = jev.ask(server, with_content(_PROBE_TURN, content), _lead(f"0.{i}"),
                                 jev.YES_NO, _PROBE_TURN.tools)
        if mass < MIN_MASS:
            raise jev.Unavailable("the options hold too little probability in the probe read")
        p.append(probs["yes"])
    if p[0] - p[1] < 0.5:
        raise jev.Unavailable("the model does not judge the probe pieces reliably")


def ready() -> jev.Server:
    """The agent model's server, probed once per process (again after
    REPROBE_S when a probe or a read switched it off). Shared with
    services/hops.py, which judges inside a tool output the same way."""
    with _lock:
        if time.monotonic() < _state["off_until"]:  # a read that waited here while the probe failed
            raise jev.Unavailable("the reader is off until it is probed again")
        if _state["server"] is None:
            server = jev.Server()
            try:
                _probe(server)
            except jev.Unavailable:
                raise
            except Exception as e:  # noqa: BLE001 - a probe that cannot finish keeps it off too
                raise jev.Unavailable(type(e).__name__) from None
            _state["server"] = server
            logger.info("JEV reader on")
        return _state["server"]


def read(turn: Turn, row: dict, question: str) -> str | None:
    """The body of `row` read for `question` inside the agent's turn `turn`:
    the text the tool returns, or None, and the caller then returns what it
    would have without the reader. Usage, fallbacks and pieces kept without
    an answer go to the request's usage ledger."""
    body = row.get("content") or ""
    if len(body) > MAX_PIECES * PIECE_CHARS:
        add_usage("jev", skipped=1)
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0, "unjudged": 0}
    out = None
    try:
        out = _read(turn, row, body, question, started, usage)
    except jev.Unavailable as e:
        _switch_off(e)
    except Exception as e:  # noqa: BLE001 - unreachable, refused or too slow: today's output
        logger.warning("JEV reader failed open: %s", type(e).__name__)
    add_usage("jev", calls=1, failopen=int(out is None), **usage)
    return out


def _read(turn: Turn, row: dict, body: str, question: str, started: float,
          usage: dict) -> str | None:
    server = ready()

    def ask(content: str, lead: str) -> tuple[dict, float, int]:
        """`lead` asked after `content` as the tool's output; the prompt's size too."""
        probs, mass, u = jev.ask(server, with_content(turn, content), lead, jev.YES_NO, turn.tools)
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]
        return probs, mass, u["input"]

    parts = _parts(row.get("entity_type"), body)
    cuts = _cuts(body, parts)
    memory = _header(row, body, parts, question)  # the reader's memory: the tool output so far
    size = 0  # tokens of the agent's prompt through it, as the server counted the last request
    kept: list[tuple[int, int]] = []
    dropped: list[tuple[int, int]] = []
    spans: list[list[tuple[int, int]]] = []  # kept or dropped, in reading order
    unread, why = None, ""
    pos = start = 0  # where the next piece's text and its span begin (whitespace before it is its own)
    numbers: dict[int, int] = {}
    while body[pos:].strip():
        if time.monotonic() > started + DEADLINE_S:
            unread, why = start, "the reading deadline"
            break
        si = _part_at(parts, pos)
        anchor = f"{si}.{numbers.get(si, 0)}"
        q = _cut(body, cuts, pos)
        if q < len(body) and (len(body) - q) < MIN_PIECE_CHARS:
            q = len(body)  # a short tail joins this piece
        if not body[pos:q].strip():  # only whitespace: the next piece's span takes it
            pos = q
            continue
        numbers[si] = numbers.get(si, 0) + 1
        if size and size + (q - pos) / CHARS_PER_TOKEN + ROOM > WINDOW_TOKENS:
            unread, why = start, "the context limit"
            break
        content = memory + _piece(f"[{anchor}] ({parts[si][0]})\n", body, pos, q)
        probs, mass, prompt = ask(content, _lead(anchor))
        if mass < MIN_MASS:
            usage["unjudged"] += 1
        stays = probs["yes"] >= KEEP or mass < MIN_MASS
        if stays:                                    # the content continues through it
            memory, size = content, prompt
            spans.append(kept)
        else:                                        # rolled back: the next piece follows `memory`
            spans.append(dropped)
        spans[-1].append((start, q))
        pos = start = q
        if stays and body[pos:].strip():
            # Does what was kept answer the question? Asked after the same
            # content, so it reuses the piece's request.
            probs, mass, _ = ask(content, f"[{anchor}] {ENOUGH_LEAD}")
            if mass >= MIN_MASS and probs["yes"] >= ENOUGH:
                unread, why = start, "enough for the question"
                break
    if not spans:
        return None  # nothing was read: the agent gets what it would have without the reader
    if unread is None and start < len(body):  # trailing whitespace joins the last span
        spans[-1][-1] = (spans[-1][-1][0], len(body))
    logger.info("JEV read %s %s: kept %d of %d pieces (%d of %d chars, %s) in %.1fs "
                "(in=%d cached=%d out=%d)", row.get("entity_type"), str(row.get("id"))[:8],
                len(kept), len(kept) + len(dropped), sum(b - a for a, b in kept), len(body),
                f"stopped at {unread}: {why}" if unread is not None else "read to the end",
                time.monotonic() - started, usage["input"], usage["cached"], usage["output"])
    return memory + _note(body, parts, kept, dropped, unread, why)
