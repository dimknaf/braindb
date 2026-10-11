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
logprobs of the yes/no ids. A piece judged useful stays, and reading
continues through it; one that is not is rolled back, and the next piece
follows what was kept before it. After a piece that stays, one more token,
"[s.p] | enough to answer (yes or no):": yes, and reading stops there (a
lookup whose answer is kept need not read on); the rest is listed as not
read. Nothing is summarised, and nothing kept is trimmed.

Nested: the tool output being built IS the reader's memory, inside the
agent's own conversation. Each decision request is the agent's next-turn
prompt up to that output — rendered by the server from the messages the SDK
will send, with the agent's own template arguments — then the end of the
tool turn and an answer turn opened with the lead. So the server computes
each piece once (but for the few tokens past a block edge, below), and the
agent's next turn continues from the cache of everything kept.

A hybrid model's server caches its state only at the last whole cache block
(jev.BLOCK tokens) a request ended in. So each piece is cut — at a part,
paragraph, line, sentence or word boundary — where its keep point lies a few
tokens past a block edge: its decision request then ends its last whole
block at that edge, and the next request, whether the piece stays or not,
and the agent's next turn reuse it. No padding enters the agent's context.

The agent gets plain text (get_entity's JSON would escape every newline): a
header saying what was read and for which question, the kept pieces, and a
note of what was left out, as char spans it can still read with get_entity's
offset/limit — which this module never touches. Anything that goes wrong
gives exactly the output the agent would have had without it. The agents
that read carry worked examples of the question in their system prompt
(agent/prompts/reader_system.md): nothing else can be cached before it.

Replayed inside the real agent's turn on 20 long pages and documents of the
largest personal memories (4-53K chars, two needs each, blind two-judge
gold): every request reused exactly the cache predicted, and the agent's
next turn the reader's. On the held-out half the top-judged piece was one
the judges call essential for 20 of 20 needs; lookups kept their answer at
30% of the page on average, gathers 32 of 33 essential pieces at 78%.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.services import jev
from braindb.services.wiki_sections import parse_sections

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).resolve().parent.parent / "agent" / "prompts"
_HEADER = (_PROMPTS / "reader_header.md").read_text(encoding="utf-8").strip()

LEAD = "| useful (yes or no):"  # asked after each piece, behind the piece's anchor
ENOUGH_LEAD = "| enough to answer (yes or no):"  # asked after each piece that stays
THINK = "<think>\n\n</think>\n\n"  # the closed empty think block the answer turn opens with
STOP_TOOLS = ("final_answer", "handoff_to_successor")  # a turn that calls one has no next turn

KEEP = 0.5          # P(yes) from which a piece stays (chosen on dev needs over long pages)
ENOUGH = 0.95       # P(yes) from which what was kept answers the question: reading stops (lookups stop
                    # early on a related piece below it; chosen on dev needs over long pages)
MIN_MASS = 0.5      # below this the options hold too little probability: the piece stays
MARGIN = 3          # a keep point lies at least this many tokens past its block edge...
WINDOW = 192        # ...and at most this many
MIN_PIECE = 160     # tokens: a piece is at least about this long
QUESTION_CHARS = jev.NEED_CHARS  # a longer query is an instruction, not a reading question
ROOM = 8192         # tokens of the served window left for the agent's later turns
MAX_PIECES = 80     # a body estimated longer than this many pieces is not read this way
DEADLINE_S = 120    # reading stops here; the rest is listed as not read
SLICE = 8000        # the note splits what was left out into spans of at most this many chars
OUTLINE_PARTS = 40  # parts named in the header's outline
CHARS_PER_TOKEN = 3.8  # a first estimate, refined on each body as it is read


@dataclass
class Turn:
    """The agent's next turn up to one tool call's output, as plain data: its
    messages (the last one is that call's tool message), its tools, and its
    chat-template arguments (None when the agent sends none)."""
    messages: list[dict]
    tools: list[dict]
    template_kwargs: dict | None


class Mismatch(Exception):
    """The request the reader would send is not the server's own rendering
    of the agent's turn: this read falls back, and the reader stays on."""


_lock = threading.Lock()
_state: dict = {"server": None, "options": None, "window": 0, "consts": None,
                "leads": {}, "off_until": 0.0}


def wanted() -> bool:
    """Whether the reader serves agent runs now."""
    return (settings.jev_reader and settings.self_hosted
            and time.monotonic() >= _state["off_until"])


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
        _state.update(off_until=time.monotonic() + jev.REPROBE_S, server=None, consts=None, leads={})
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


def _with(turn: Turn, content: str, answer: str) -> list[dict]:
    """The turn's messages with `content` as its last message's (the tool's
    output, or the wiki gate's user message), then an answer turn that starts
    with `answer`."""
    return turn.messages[:-1] + [{**turn.messages[-1], "content": content},
                                 {"role": "assistant", "content": answer}]


def _render(server: jev.Server, turn: Turn, content: str, answer: str) -> list[int]:
    """The server's own rendering of `_with(...)`, ending with `answer`."""
    return server.render(_with(turn, content, answer), template_kwargs=turn.template_kwargs,
                         tools=turn.tools, add_generation_prompt=False,
                         continue_final_message=True)


def split_turn(server: jev.Server, turn: Turn, consts: dict) -> tuple[list[int], list[int]]:
    """C, the agent's next-turn prompt up to the tool's output, and T, from
    the end of that output to the opened answer turn, cut from the server's
    own rendering with the agent's own template arguments. The output's last
    token and the template's newline after it are one token in a rendering,
    so the output is counted with its newline and T starts after it. The
    content may instead be a user message's (the wiki gate's): the template
    trims it and the end-of-turn marker follows it, so T starts there."""
    x = consts["x"] if turn.messages[-1]["role"] == "tool" else consts["x_user"]
    q, think = consts["q"], consts["think"]
    a, b = _render(server, turn, "x", "Q"), _render(server, turn, "y", "Q")
    n = next((i for i, (p, r) in enumerate(zip(a, b)) if p != r), None)
    if n is None or a[n:n + len(x)] != x or a[-len(q):] != q:
        raise Mismatch("the content is not where the reader expects it in the turn")
    t = a[n + len(x):len(a) - len(q)]
    if t[-len(think):] != think:
        raise jev.Unavailable("the template does not open the answer turn with a closed think block")
    return a[:n], t


def _lead(anchor: str) -> str:
    """The closed question asked after a piece, behind its anchor (the
    question it is asked for is stated once, in the header)."""
    return f"[{anchor}] {LEAD}"


def _lead_ids(server: jev.Server, text: str) -> list[int]:
    if text not in _state["leads"]:
        _state["leads"][text] = server.encode(text)
    return _state["leads"][text]


def check(server: jev.Server, turn: Turn, content: str, lead: str, prompt: list[int]) -> None:
    """The assembled request must be the server's own rendering of the turn."""
    if _render(server, turn, content, lead) != prompt:
        raise Mismatch("the assembled request differs from the server's rendering")


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


def _cut(server: jev.Server, body: str, cuts: list[list[int]], frame: str, a: int,
         lo: int, hi: int, ratio: float) -> tuple[int, list[int]]:
    """Where the piece starting at `a` ends, and its tokens: the strongest cut
    whose piece takes between `lo` and `hi` tokens; the rest of the body when
    that is shorter than `lo`."""
    end = len(body)

    def tokens(b: int) -> list[int]:
        return server.encode(_piece(frame, body, a, b))

    # Far enough for `lo` tokens: `ratio` only estimates chars per token, so
    # widen until it is (a body of digits or ids has fewer than estimated).
    reach = min(end, a + int(hi * ratio * 1.5) + 200)
    while len(ids := tokens(reach)) < lo and reach < end:
        reach = min(end, a + 2 * (reach - a))
    if len(ids) < lo:
        return end, ids  # the rest of the body is shorter than a piece: the last piece

    def first(positions: list[int]) -> tuple[int, list[int]] | None:
        """The first position whose piece reaches `lo` tokens (pieces grow
        with their end), or None."""
        found, left, right = None, 0, len(positions) - 1
        while left <= right:
            mid = (left + right) // 2
            ids = tokens(positions[mid])
            if len(ids) >= lo:
                found, right = (positions[mid], ids), mid - 1
            else:
                left = mid + 1
        return found

    for positions in cuts:
        found = first(positions[bisect_right(positions, a):bisect_right(positions, reach)])
        if found and len(found[1]) <= hi:
            return found
    # No boundary in the window (one very long word or line): cut between chars.
    return first(list(range(a + 1, reach + 1))) or (reach, ids)


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
    messages=[{"role": "user", "content": _PROBE_QUESTION},
              {"role": "assistant", "content": None, "tool_calls": [{
                  "id": "call_probe", "type": "function",
                  "function": {"name": "get_entity", "arguments": '{"entity_id": "probe"}'}}]},
              {"role": "tool", "tool_call_id": "call_probe", "content": ""}],
    tools=[{"type": "function", "function": {
        "name": "get_entity", "description": "Fetch one entity by ID.",
        "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}},
                       "required": ["entity_id"]}}}],
    template_kwargs=None)
_PROBE_PIECES = ("Kel Aster opened the Rowan bakery on 2 June 2018, after two years of renovation.",
                 "The Rowan bakery sells rye bread on Saturdays and closes on Mondays.")


def _probe(server: jev.Server, options: dict) -> None:
    """The model must keep the piece that answers a clear-cut question and
    drop the one that does not, read the reader's way; the requests must be
    the server's own renderings."""
    c, t = split_turn(server, _PROBE_TURN, _state["consts"])
    body = "\n\n".join(_PROBE_PIECES)
    row = {"entity_type": "datasource", "id": "probe", "title": "Rowan bakery", "content": body}
    header = _header(row, body, [("text", 0)], _PROBE_QUESTION)
    p = []
    for i, text in enumerate(_PROBE_PIECES):
        content, lead = f"{header}[0.{i}] (text)\n{text}\n", _lead(f"0.{i}")
        prompt = c + server.encode(content) + t + _lead_ids(server, lead)
        try:
            check(server, _PROBE_TURN, content, lead, prompt)
        except Mismatch as e:
            raise jev.Unavailable(str(e)) from None
        probs, mass, _ = jev.ask_ids(server, prompt, options)
        if mass < MIN_MASS:
            raise jev.Unavailable("the options hold too little probability in the probe read")
        p.append(probs["yes"])
    if p[0] - p[1] < 0.5:
        raise jev.Unavailable("the model does not judge the probe pieces reliably")


def connect() -> tuple[jev.Server, dict, int, dict]:
    """The agent model's server, its yes/no option ids, its served window and
    the rendering markers `split_turn` cuts by (jev.Unavailable when it states no
    window to read into)."""
    server = jev.Server()
    card = next((m for m in server.get("/models").get("data", [])
                 if m.get("id") == server.model), None)
    if card is None:
        raise jev.Unavailable("the agent model is not served by this endpoint")
    window = int(card.get("max_model_len") or 0)
    if window <= 2 * ROOM:
        raise jev.Unavailable("the server does not state a context window to read into")
    options = jev.option_ids(server, ["yes", "no"])
    consts = {"x": server.encode("x\n"), "x_user": server.encode("x"), "q": server.encode("Q"),
              "think": server.encode(THINK)}
    return server, options, window, consts


def ready() -> tuple[jev.Server, dict, int, dict]:
    """The server, the yes/no options, the served window and the rendering
    markers, probed once per process (again after REPROBE_S when a probe or a
    read switched it off). Shared with services/hops.py, which judges inside
    a tool output the same way."""
    with _lock:
        if time.monotonic() < _state["off_until"]:  # a read that waited here while the probe failed
            raise jev.Unavailable("the reader is off until it is probed again")
        if _state["server"] is None:
            try:
                server, options, window, _state["consts"] = connect()
                _probe(server, options)
            except jev.Unavailable:
                raise
            except Exception as e:  # noqa: BLE001 - a probe that cannot finish keeps it off too
                raise jev.Unavailable(str(e) if isinstance(e, Mismatch) else type(e).__name__) from None
            _state.update(server=server, options=options, window=window)
            logger.info("JEV reader on: a served window of %d tokens", window)
        return _state["server"], _state["options"], _state["window"], _state["consts"]


def read(turn: Turn, row: dict, question: str) -> str | None:
    """The body of `row` read for `question` inside the agent's turn `turn`:
    the text the tool returns, or None, and the caller then returns what it
    would have without the reader. Usage, fallbacks and pieces kept without
    an answer go to the request's usage ledger."""
    body = row.get("content") or ""
    if len(body) > MAX_PIECES * jev.BLOCK * CHARS_PER_TOKEN:
        add_usage("jev", skipped=1)
        return None
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0, "unjudged": 0}
    out = None
    try:
        out = _read(turn, row, body, question, started, usage)
    except jev.Unavailable as e:
        _switch_off(e)
    except Mismatch as e:
        logger.warning("JEV reader fell back: %s", e)
    except Exception as e:  # noqa: BLE001 - unreachable, refused or too slow: today's output
        logger.warning("JEV reader failed open: %s", type(e).__name__)
    add_usage("jev", calls=1, failopen=int(out is None), **usage)
    return out


def _read(turn: Turn, row: dict, body: str, question: str, started: float,
          usage: dict) -> str | None:
    server, options, window, consts = ready()
    c, t = split_turn(server, turn, consts)

    def spend(u: dict) -> None:
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]

    parts = _parts(row.get("entity_type"), body)
    cuts = _cuts(body, parts)
    memory = _header(row, body, parts, question)  # the reader's memory: the tool output so far
    m_ids = server.encode(memory)
    # The agent's prompt through the header, so that a piece dropped before
    # any is kept rolls back to a cached block near it.
    spend(jev.usage_of(server.complete({"prompt": c + m_ids, "max_tokens": 1, "temperature": 0})))
    kept: list[tuple[int, int]] = []
    dropped: list[tuple[int, int]] = []
    spans: list[list[tuple[int, int]]] = []  # kept or dropped, in reading order
    unread, why = None, ""
    pos = start = 0  # where the next piece's text and its span begin (whitespace before it is its own)
    numbers: dict[int, int] = {}
    read_chars, read_tokens = 0, 0
    checked = False
    while body[pos:].strip():
        if time.monotonic() > started + DEADLINE_S:
            unread, why = start, "the reading deadline"
            break
        # The keep point K of a content is its last token before the newline
        # the template ends it with (that one joins whatever follows). The
        # next piece must take K a few tokens past the next block edge, and
        # its decision request — K, that newline, T and the lead — must end
        # before the block after it: then the request's last whole block is
        # the edge, the one the next request and the agent's next turn reuse.
        # An edge closer than MIN_PIECE would make a tiny piece: the one after.
        k_m = len(c) + len(m_ids) - 1
        edge = (k_m // jev.BLOCK + 1) * jev.BLOCK
        if edge - k_m < MIN_PIECE:
            edge += jev.BLOCK
        si = _part_at(parts, pos)
        anchor = f"{si}.{numbers.get(si, 0)}"
        frame = f"[{anchor}] ({parts[si][0]})\n"
        lead = _lead(anchor)
        lead_ids = _lead_ids(server, lead)
        enough_ids = _lead_ids(server, f"[{anchor}] {ENOUGH_LEAD}")  # asked at the same point if it stays
        hi = min(edge + WINDOW, edge + jev.BLOCK - 2 - len(t) - max(len(lead_ids), len(enough_ids)))
        ratio = read_chars / read_tokens if read_tokens else CHARS_PER_TOKEN
        q, ids = _cut(server, body, cuts, frame, pos, edge + MARGIN - k_m, hi - k_m, ratio)
        if q < len(body) and (len(body) - q) < MIN_PIECE * ratio:
            q = len(body)  # a short tail joins this piece
        if not body[pos:q].strip():  # only whitespace: the next piece's span takes it
            pos = q
            continue
        numbers[si] = numbers.get(si, 0) + 1
        read_chars, read_tokens = read_chars + (q - pos), read_tokens + len(ids)
        content = memory + _piece(frame, body, pos, q)
        s_ids = server.encode(content)  # the whole output so far, as the server tokenizes it
        if len(c) + len(s_ids) + ROOM > window:
            unread, why = start, "the context limit"
            break
        prompt = c + s_ids + t + lead_ids
        if not checked:
            check(server, turn, content, lead, prompt)
            checked = True
        probs, mass, u = jev.ask_ids(server, prompt, options)
        spend(u)
        if mass < MIN_MASS:
            usage["unjudged"] += 1
        stays = probs["yes"] >= KEEP or mass < MIN_MASS
        if stays:                                    # the content continues through it
            memory, m_ids = content, s_ids
            spans.append(kept)
        else:                                        # rolled back: the next piece follows `memory`
            spans.append(dropped)
        spans[-1].append((start, q))
        pos = start = q
        if stays and body[pos:].strip():
            # Does what was kept answer the question? Asked at the same point,
            # so its request ends in the same block as the piece's.
            probs, mass, u = jev.ask_ids(server, c + s_ids + t + enough_ids, options)
            spend(u)
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
