"""
Hops: JEV follows BrainDB's relations from what recall showed, inside the
agent's own conversation; a neighbour reaches the agent only when JEV keeps
it, and only the neighbours kept are followed further.

BrainDB's recall ranks its seeds and their graph neighbours by stored scores
(similarity, relation weights, decay) and shows the top items as previews. A
neighbour of a shown item that ranked below that cut never reaches the agent
unless the agent explores around the item itself (view_tree or
view_entity_relations: a turn each, returning every neighbour, needed or
not). With JEV_HOPS on, a recall_memory call in an /agent/query request with
a reading question (services/reader.py's: the need, else a short query) is
extended the adaptive reader's way, nested in the agent's own conversation:
the neighbours of the shown items (relations only: no keyword tags, no
retired pages, nothing already shown), those most of the shown items link
to first, are appended one at a time to the tool's output, each followed by
ONE token, "[h1.2] | useful (yes or no):", thinking off, read from the
logprobs of its yes/no token. A page is offered by its name, summary line and
section names, a document also with the shown item it is linked from, and
anything else by its content. Yes (P >= KEEP): the output continues through
it, and its own neighbours are the next hop's candidates. No, or too little
probability on the options: it is rolled back and never followed. Up to
HOPS hops of at most CANDIDATES neighbours each. Nothing recall showed is
removed, and anything that goes wrong gives recall's output as it was.

Each decision request is the agent's next request up to the tool's output,
then the question, as the reader's are (services/reader.py, jev.ask): every
neighbour is computed once, and the agent's next turn continues from the
cache of what was kept. The agents that read carry worked examples of the
question in their system prompt (agent/prompts/hops_system.md).

The wiki gate takes the job members' neighbours as extra candidates the same
way (services/wiki_gate.py, `neighbours`).
"""
from __future__ import annotations

import logging
import threading
import time

import psycopg2.extras

from braindb.agent.run_state import add_usage
from braindb.config import settings
from braindb.db import get_conn
from braindb.services import jev, reader
from braindb.services.wiki_jobs import SUMMARY_RE
from braindb.services.wiki_sections import parse_sections

logger = logging.getLogger(__name__)

LEAD = "| useful (yes or no):"  # asked after each neighbour, behind its anchor
HOPS = 2            # hops followed from what recall showed
CANDIDATES = 6      # neighbours offered per hop, those most of the shown items link to first
KEEP = 0.9          # P(yes) from which a neighbour stays, and is followed
MIN_MASS = jev.MIN_MASS  # below this the options hold too little probability: rolled back
PREVIEW = 400       # chars of a neighbour's content it is offered with
SECTIONS = 12       # a page's section names it is offered with
DEADLINE_S = 60     # for all of a recall's hops; what is left is not offered
# Pages and documents first among neighbours equally linked: they are where
# the facts recall shows were summarised or came from.
_TYPE_ORDER = {"wiki": 0, "datasource": 0, "source": 1}

_state: dict = {"off_until": 0.0}
_lock = threading.Lock()

# Each neighbour once, with how many of the given items link to it, the
# first (best-ranked) of them, and that link's relation.
_NEIGHBOURS_SQL = """
WITH links AS (
    SELECT t.id AS tid, f.id AS via, f.ord, r.relation_type, (r.from_entity_id = f.id) AS outgoing,
           COALESCE(r.relevance_score, 0.5) * COALESCE(r.importance_score, 0.5) AS weight
    FROM unnest(%s::uuid[]) WITH ORDINALITY AS f(id, ord)
    JOIN relations r ON r.from_entity_id = f.id OR r.to_entity_id = f.id
    JOIN entities t ON t.id = CASE WHEN r.from_entity_id = f.id THEN r.to_entity_id ELSE r.from_entity_id END
    LEFT JOIN wikis_ext w ON w.entity_id = t.id
    WHERE r.relation_type <> 'tagged_with' AND t.entity_type <> 'keyword'
      AND (t.entity_type <> 'wiki' OR w.retired_at IS NULL)
      AND NOT (t.id = ANY(%s::uuid[]))
)
SELECT t.id::text AS id, t.entity_type, t.title, t.content, t.keywords,
       count(DISTINCT l.via) AS links, min(l.ord) AS ord, max(l.weight) AS weight,
       (array_agg(l.relation_type ORDER BY l.ord, l.weight DESC))[1] AS relation_type,
       (array_agg(l.outgoing ORDER BY l.ord, l.weight DESC))[1] AS outgoing,
       (array_agg(l.via::text ORDER BY l.ord, l.weight DESC))[1] AS via,
       (array_agg(v.content ORDER BY l.ord, l.weight DESC))[1] AS via_content
FROM links l
JOIN entities t ON t.id = l.tid
JOIN entities v ON v.id = l.via
GROUP BY t.id, t.entity_type, t.title, t.content, t.keywords
"""


def wanted() -> bool:
    """Whether hops serve agent runs now."""
    return settings.jev_hops and time.monotonic() >= _state["off_until"]


def neighbours(conn, ids: list[str], exclude: set[str], limit: int,
               types: tuple[str, ...] | None = None) -> list[dict]:
    """The relation neighbours of `ids` not in `exclude` (keyword tags,
    keywords and retired pages left out), of `types` when given, at most
    `limit`: those most of `ids` link to first, then pages and documents,
    then those of the earliest of `ids` (recall's order is the question's).
    Replayed on 30 needs over a personal memory, this order brought the
    document a need was answered from within the first five neighbours for
    7 of the 10 needs where it was one relation away (stored relation weight:
    3 of 10)."""
    if not ids or limit <= 0:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_NEIGHBOURS_SQL, (list(ids), sorted({*exclude, *ids})))
        rows = [dict(r) for r in cur.fetchall()]
    return _first(rows, limit, types)


def _first(rows: list[dict], limit: int, types: tuple[str, ...] | None = None) -> list[dict]:
    """`neighbours`' order and cut, on its rows."""
    rows = [r for r in rows if not types or r["entity_type"] in types]
    rows.sort(key=lambda r: (-r["links"], _TYPE_ORDER.get(r["entity_type"], 2), r["ord"],
                             -float(r["weight"] or 0)))
    return rows[:limit]


def _switch_off(error: Exception) -> None:
    with _lock:
        _state["off_until"] = time.monotonic() + jev.REPROBE_S
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV hops off for %d s: %s", jev.REPROBE_S, reason)


def _clip(text: str | None, n: int) -> str:
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + f"...(+{len(s) - n} chars; get_entity for full)"


def _view(n: dict) -> str:
    """What a neighbour is judged by. A page by its name, summary line and
    section names (its body opens with machine metadata), a document by its
    title and opening and the shown item it is linked from, anything else
    by its content."""
    if n["entity_type"] == "wiki":
        body = n.get("content") or ""
        summary = SUMMARY_RE.search(body)
        _, sections = parse_sections(body)
        names = "; ".join(s.name for s in sections[:SECTIONS]) + ("; ..." if len(sections) > SECTIONS else "")
        return (f"  page: {n.get('title') or ''}" + (f" - {_clip(summary.group(1), 300)}" if summary else "")
                + (f"\n  sections: {names}" if names else "") + "\n")
    text = _clip(f"{n.get('title') or ''} {n.get('content') or ''}", PREVIEW)
    if n["entity_type"] == "datasource" and n.get("via_content"):
        return f"  content: {text}\n  linked from: {_clip(n['via_content'], 200)}\n"
    return f"  content: {text}\n"


def _item(anchor: str, n: dict) -> str:
    """One neighbour as the tool's output carries it: its anchor, type,
    relation and how many shown items link to it, then its id and view, and
    a newline (the template closes a tool output on one)."""
    # The relation as seen from the item it hangs on: "derived_from from 3f2a1b0c"
    # (that item derives from this one), "summarises to 3f2a1b0c" (this one summarises it).
    way = "from" if n["outgoing"] else "to"
    also = f" (and {n['links'] - 1} more of the items above)" if n.get("links", 1) > 1 else ""
    return (f"[{anchor}] [{n['entity_type']}] {n['relation_type']} {way} {n['via'][:8]}{also}\n"
            f"  id: {n['id']}\n{_view(n)}")


def extend(turn: reader.Turn, shown: list[str], text: str, question: str) -> str | None:
    """Recall's output `text`, whose items are `shown`, with the neighbours
    JEV kept for `question` appended inside the agent's turn `turn`; or None,
    and the caller returns `text` as it was. Usage goes to the request's
    ledger."""
    started = time.monotonic()
    usage = {"requests": 0, "input": 0, "cached": 0, "output": 0}
    out, failed = None, True
    try:
        out = _extend(turn, shown, text, question, started + DEADLINE_S, usage)
        failed = False  # nothing kept is an answer too: recall's output as it was
    except jev.Unavailable as e:
        _switch_off(e)
    except Exception as e:  # noqa: BLE001 - unreachable, refused or too slow: recall's output
        logger.warning("JEV hops failed open: %s", type(e).__name__)
    add_usage("jev", calls=1, failopen=int(failed), **usage)
    return out


def _extend(turn: reader.Turn, shown: list[str], text: str, question: str, deadline: float,
            usage: dict) -> str | None:
    server = reader.ready()

    def spend(u: dict) -> None:
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]

    # The tool's output so far: recall's, then what the hops keep.
    memory = (f"{text}\n\nNeighbours of the items above, one hop at a time, each kept only if useful for: "
              f"{question}\n")
    size = 0  # tokens of the agent's prompt through it, as the server counted the last request
    seen, frontier = set(shown), list(shown)
    offered, kept, done = 0, 0, False
    for hop in range(1, HOPS + 1):
        with get_conn() as conn:  # its own short transaction
            cands = neighbours(conn, frontier, seen, CANDIDATES)
        seen |= {n["id"] for n in cands}
        frontier = []
        for i, n in enumerate(cands, 1):
            anchor = f"h{hop}.{i}"
            item = _item(anchor, n)
            # The deadline, or the room the agent keeps for its later turns:
            # the rest is not offered.
            if time.monotonic() > deadline or (
                    size and size + len(item) / reader.CHARS_PER_TOKEN + reader.ROOM > reader.WINDOW_TOKENS):
                done = True
                break
            content = memory + item
            probs, mass, u = jev.ask(server, reader.with_content(turn, content), f"[{anchor}] {LEAD}",
                                     jev.YES_NO, turn.tools, turn.extra)
            spend(u)
            offered += 1
            if mass >= MIN_MASS and probs["yes"] >= KEEP:  # stays, and is followed next hop
                memory, size = content, u["input"]
                frontier.append(n["id"])
                kept += 1
        if done or not frontier:
            break
    logger.info("JEV hops for %d shown items: kept %d of %d neighbours offered in %.1fs (in=%d cached=%d out=%d)",
                len(shown), kept, offered, time.monotonic() - (deadline - DEADLINE_S),
                usage["input"], usage["cached"], usage["output"])
    if not kept:
        return None  # recall's output as it was: the agent's next turn still continues from its cache
    return memory + f"(Kept {kept} of {offered} neighbours offered; get_entity reads any in full.)\n"
