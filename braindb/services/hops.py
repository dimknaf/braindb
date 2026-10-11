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
retired pages, nothing already shown), strongest relation first, are
appended one at a time to the tool's output, each followed by ONE token,
"[h1.2] | useful (yes or no):", thinking off, read from the logprobs of the
yes/no ids. Yes: the output continues through it, and its own neighbours are
the next hop's candidates. No, or too little probability on the options: it
is rolled back and never followed. Up to HOPS hops of at most CANDIDATES
neighbours each. Nothing recall showed is removed, and anything that goes
wrong gives recall's output as it was.

Each decision request is the agent's next turn up to the tool's output, as
the reader's are (reader.split_turn and reader.check): every neighbour is
computed once, and the agent's next turn continues from the cache of what
was kept. The agents that read carry worked examples of the question in
their system prompt (agent/prompts/hops_system.md).

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

logger = logging.getLogger(__name__)

LEAD = "| useful (yes or no):"  # asked after each neighbour, behind its anchor
HOPS = 2            # hops followed from what recall showed
CANDIDATES = 10     # neighbours offered per hop, strongest relation first
KEEP = 0.5          # P(yes) from which a neighbour stays, and is followed
MIN_MASS = 0.5      # below this the options hold too little probability: rolled back
PREVIEW = 400       # chars of a neighbour's content it is offered with
DEADLINE_S = 60     # for all of a recall's hops; what is left is not offered

_state: dict = {"off_until": 0.0}
_lock = threading.Lock()

_NEIGHBOURS_SQL = """
SELECT DISTINCT ON (t.id)
       t.id::text AS id, t.entity_type, t.title, t.content, t.keywords,
       r.relation_type, (r.from_entity_id = f.id) AS outgoing, f.id::text AS via, f.ord,
       COALESCE(r.relevance_score, 0.5) * COALESCE(r.importance_score, 0.5) AS weight
FROM unnest(%s::uuid[]) WITH ORDINALITY AS f(id, ord)
JOIN relations r ON r.from_entity_id = f.id OR r.to_entity_id = f.id
JOIN entities t ON t.id = CASE WHEN r.from_entity_id = f.id THEN r.to_entity_id ELSE r.from_entity_id END
LEFT JOIN wikis_ext w ON w.entity_id = t.id
WHERE r.relation_type <> 'tagged_with' AND t.entity_type <> 'keyword'
  AND (t.entity_type <> 'wiki' OR w.retired_at IS NULL)
  AND NOT (t.id = ANY(%s::uuid[]))
ORDER BY t.id, weight DESC, f.ord
"""


def wanted() -> bool:
    """Whether hops serve agent runs now."""
    return (settings.jev_hops and settings.self_hosted
            and time.monotonic() >= _state["off_until"])


def neighbours(conn, ids: list[str], exclude: set[str], limit: int,
               types: tuple[str, ...] | None = None) -> list[dict]:
    """The relation neighbours of `ids` not in `exclude` (keyword tags,
    keywords and retired pages left out), strongest relation first, then
    the order of `ids`: at most `limit`, of `types` when given."""
    if not ids or limit <= 0:
        return []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_NEIGHBOURS_SQL, (list(ids), sorted({*exclude, *ids})))
        rows = [dict(r) for r in cur.fetchall()]
    if types:
        rows = [r for r in rows if r["entity_type"] in types]
    rows.sort(key=lambda r: (-float(r["weight"] or 0), r["ord"]))
    return rows[:limit]


def _switch_off(error: Exception) -> None:
    with _lock:
        _state["off_until"] = time.monotonic() + jev.REPROBE_S
    reason = str(error) if isinstance(error, jev.Unavailable) else type(error).__name__
    logger.warning("JEV hops off for %d s: %s", jev.REPROBE_S, reason)


def _item(anchor: str, n: dict) -> str:
    """One neighbour as the tool's output carries it: its anchor, type and
    relation, then id and content as recall shows its items, and a newline
    (the template closes a tool output on one)."""
    text = " ".join(((n.get("title") or "") + " " + (n.get("content") or "")).split())
    if len(text) > PREVIEW:
        text = text[:PREVIEW] + f"...(+{len(text) - PREVIEW} chars; get_entity for full)"
    # The relation as seen from the item it hangs on: "derived_from from 3f2a1b0c"
    # (that item derives from this one), "summarises to 3f2a1b0c" (this one summarises it).
    way = "from" if n["outgoing"] else "to"
    return (f"[{anchor}] [{n['entity_type']}] {n['relation_type']} {way} {n['via'][:8]}\n"
            f"  id: {n['id']}\n  content: {text}\n")


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
    except reader.Mismatch as e:
        logger.warning("JEV hops fell back: %s", e)
    except Exception as e:  # noqa: BLE001 - unreachable, refused or too slow: recall's output
        logger.warning("JEV hops failed open: %s", type(e).__name__)
    add_usage("jev", calls=1, failopen=int(failed), **usage)
    return out


def _extend(turn: reader.Turn, shown: list[str], text: str, question: str, deadline: float,
            usage: dict) -> str | None:
    server, options, window, consts = reader.ready()
    c, t = reader.split_turn(server, turn, consts)
    leads: dict[str, list[int]] = {}

    def spend(u: dict) -> None:
        usage["requests"] += 1
        for k in ("input", "cached", "output"):
            usage[k] += u[k]

    # The tool's output so far: recall's, then what the hops keep.
    memory = (f"{text}\n\nNeighbours of the items above, one hop at a time, each kept only if useful for: "
              f"{question}\n")
    # Recall's output through this line, so that a neighbour rolled back
    # before any is kept returns to a cached block near it.
    spend(jev._usage(server.complete({"prompt": c + server.encode(memory), "max_tokens": 1, "temperature": 0})))
    seen, frontier = set(shown), list(shown)
    offered, kept, checked, done = 0, 0, False, False
    for hop in range(1, HOPS + 1):
        with get_conn() as conn:  # its own short transaction
            cands = neighbours(conn, frontier, seen, CANDIDATES)
        seen |= {n["id"] for n in cands}
        frontier = []
        for i, n in enumerate(cands, 1):
            anchor = f"h{hop}.{i}"
            content, lead = memory + _item(anchor, n), f"[{anchor}] {LEAD}"
            s_ids = server.encode(content)  # the whole output so far, as the server tokenizes it
            # The deadline, or the room the agent keeps for its later turns:
            # the rest is not offered.
            if time.monotonic() > deadline or len(c) + len(s_ids) + reader.ROOM > window:
                done = True
                break
            if lead not in leads:
                leads[lead] = server.encode(lead)
            prompt = c + s_ids + t + leads[lead]
            if not checked:
                reader.check(server, turn, content, lead, prompt)
                checked = True
            probs, mass, u = jev.ask_ids(server, prompt, options)
            spend(u)
            offered += 1
            if mass >= MIN_MASS and probs["yes"] >= KEEP:  # stays, and is followed next hop
                memory = content
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
