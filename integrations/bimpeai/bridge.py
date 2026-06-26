"""
bridge.py — the local fulfiller for the agent-driven BrainDB memory integration.

The BimpeAI cloud agent holds the whole conversation. When IT decides to save or
recall, it emits a memory command in its reply:

    [[MEMORY: <plain-English instruction>]]

This bridge polls the agent's conversations (SDK, outbound — no tunnel), and for
each such command runs it through BrainDB's own agent (/api/v1/agent/query, which
saves OR recalls in natural language), then posts the answer back into the
conversation as:

    [[MEMORY_RESULT: <answer>]]

…which triggers the agent's final reply to the user. The bridge does NOT ingest
messages and does NOT decide anything — the AGENT decides; the bridge only fulfils
its memory commands.

Config: this folder's .env (BIMPEAI_API_KEY, BIMPEAI_AGENT_ID, BRAINDB_BASE_URL).
Run:    integrations/bimpeai/.venv/Scripts/python  integrations/bimpeai/bridge.py
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import requests
from bimpeai import BimpeAI

HERE = Path(__file__).resolve().parent
ENV_PATH = HERE / ".env"
STATE_DIR = HERE / ".state"
SEEN_FILE = STATE_DIR / "seen.txt"

# Matches the command [[MEMORY: q]] but NOT the result [[MEMORY_RESULT: ...]]
# (the colon comes right after MEMORY only in the command).
MEM_RE = re.compile(r"\[\[MEMORY:\s*(.+?)\]\]", re.IGNORECASE | re.DOTALL)

# Make stdout UTF-8 so the agent's em-dashes / smart quotes can't crash logging on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bimpeai-bridge] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger("bimpeai-bridge")
logging.getLogger("httpx").setLevel(logging.WARNING)


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def load_seen() -> set[str]:
    try:
        return set(SEEN_FILE.read_text(encoding="utf-8").split())
    except Exception:
        return set()


def save_seen(seen: set[str], cap: int = 5000) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text("\n".join(list(seen)[-cap:]), encoding="utf-8")


def braindb_query(base: str, q: str, timeout: int = 180) -> str:
    """Plain-English save/recall via BrainDB's own agent."""
    r = requests.post(f"{base}/api/v1/agent/query", json={"query": q}, timeout=timeout)
    r.raise_for_status()
    return ((r.json() or {}).get("answer") or "").strip()


def main() -> None:
    env = load_env()
    api_key = env.get("BIMPEAI_API_KEY", "").strip()
    agent_id = env.get("BIMPEAI_AGENT_ID", "").strip()
    braindb = env.get("BRAINDB_BASE_URL", "http://localhost:24700").strip().rstrip("/")
    poll = int(env.get("BIMPEAI_POLL_SECONDS", "3"))

    if not api_key or not agent_id:
        log.info("dormant: set BIMPEAI_API_KEY and BIMPEAI_AGENT_ID (run setup_agent.py). Sleeping.")
        while True:
            time.sleep(3600)

    client = BimpeAI(api_key=api_key)
    seen = load_seen()
    prime = len(seen) == 0    # first run: learn current message ids, don't replay history
    log.info("bridge up. agent=%s braindb=%s poll=%ss%s", agent_id, braindb, poll,
             " (priming)" if prime else "")

    while True:
        try:
            for conv in client.conversations.list(agent_id):
                cid = conv.id
                try:
                    msgs = list(client.conversations.messages.list(agent_id, cid, sort="created_at"))
                except requests.RequestException:
                    continue
                except Exception as e:
                    log.warning("messages.list(%s): %s", cid, _short(e))
                    continue
                for m in msgs:
                    if m.id in seen:
                        continue
                    seen.add(m.id)
                    if prime:
                        continue
                    _handle(client, braindb, agent_id, cid, m)
            if prime:
                prime = False
                log.info("primed %d message(s); watching for the agent's [[MEMORY: ...]] commands", len(seen))
            save_seen(seen)
        except Exception as e:
            log.warning("poll error: %s", _short(e))   # transient (e.g. DNS) — keep going, no traceback
        time.sleep(poll)


def _handle(client: BimpeAI, braindb: str, agent_id: str, cid: str, m) -> None:
    # Only the AGENT's memory commands matter. Ignore user messages and our own results.
    if (getattr(m, "role", "") or "").lower() != "assistant":
        return
    text = getattr(m, "message", "") or ""
    cmd = MEM_RE.search(text)
    if not cmd:
        return
    q = cmd.group(1).strip()
    log.info("MEMORY command -> %r", q[:120])
    try:
        answer = braindb_query(braindb, q) or "Nothing found for that yet."
    except Exception as e:
        answer = f"(memory service error: {_short(e)})"
    answer = answer.replace("\n", " ").strip()
    try:
        client.conversations.messages.send(agent_id, cid, message=f"[[MEMORY_RESULT: {answer}]]", role="user")
        log.info("posted MEMORY_RESULT (%d chars)", len(answer))
    except Exception as e:
        log.warning("could not post result: %s", _short(e))


def _short(e: object) -> str:
    s = str(e).strip().splitlines()
    return (s[0] if s else repr(e))[:160]


if __name__ == "__main__":
    main()
