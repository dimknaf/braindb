"""
Hermes situational ingestor (hermes profile) — Strait of Hormuz monitor.

Launched by `braindb.profile_runner` when CUSTOM_PROFILE includes `hermes`. On a
schedule it asks an external Hermes Agent (its headless OpenAI-compatible gateway)
the question in this folder's `query.md`, and drops the answer into `data/sources/`
as one datasource, where the existing watcher ingests + extracts it. The
accumulating facts form the central "Hormuz status" wiki (shaped by
`wiki_writer.add.md`), while default rules build wikis for the other entities
mentioned — a situation-aware memory.

DORMANT BY DEFAULT: with no HERMES_API_KEY (and no HERMES_SAMPLE_FILE) it logs
"idle" and NEVER calls Hermes. Each ask is a FRESH, STATELESS discussion (no
session header), so no context bleeds between polls.

Config — from this folder's own `.env` (copy `.env.example` -> `.env`):
    HERMES_ASK_URL        Hermes gateway base URL (default http://127.0.0.1:8642)
    HERMES_API_KEY        bearer token (REQUIRED to run live; unset => dormant)
    HERMES_MODEL          model name sent in the request (default hermes-agent)
    HERMES_ASK_TIMEOUT    per-ask timeout seconds (default 600; agent loop is slow)
    HERMES_POLL_SECONDS   loop cadence (default 1800 = 30 min; 3600 = hourly)
    HERMES_DEDUP          skip re-ingesting an unchanged answer (default true)
    HERMES_SAMPLE_FILE    DRY-RUN: use this file's text as the "answer" instead of
                          calling Hermes (validate the pipeline before Hermes is live)
    HERMES_PRUNE_DAYS     delete ingested hermes-*.md older than this (default 0=keep)

The Hormuz prompt itself lives in `query.md` (committed, profile-owned), not here.
"""
import hashlib
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]              # custom-profiles/hermes -> repo root
_SOURCES = _REPO_ROOT / "data" / "sources"
_INGESTED = _SOURCES / "ingested"
_STATE = _HERE / ".state"
_LAST = _STATE / "last.txt"                # substance-hash of the last dropped answer
_QUERY_FILE = _HERE / "query.md"

_SUBJECT = "Hormuz status"                 # baked in — the whole profile is Hormuz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hermes-ingestor] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("hermes-ingestor")


def _load_env(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Container/host env wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)[:80]


def _resolve(p: str) -> Path:
    """Resolve a configured path against repo root / profile dir if relative."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    for base in (_REPO_ROOT, _HERE):
        cand = base / pp
        if cand.is_file():
            return cand
    return _REPO_ROOT / pp


def _load_question() -> str:
    try:
        return _QUERY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _ask_hermes(question: str, url: str, key: str, model: str, timeout: int) -> str:
    """One FRESH, STATELESS ask to the Hermes OpenAI-compatible gateway.

    No `X-Hermes-Session-Id` header is sent, so each call is an independent
    discussion with no carried-over context."""
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    r = requests.post(endpoint, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # OpenAI-compatible; tolerate minor shape differences.
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        if isinstance(data, dict):
            return (data.get("answer") or data.get("content") or "").strip()
        return ""


def _substance(answer: str) -> str:
    """Hash the answer's substance (whitespace-collapsed) for change-detection."""
    return hashlib.sha256(re.sub(r"\s+", " ", answer).strip().encode("utf-8")).hexdigest()


def _load_last() -> str:
    try:
        return _LAST.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _save_last(h: str) -> None:
    _STATE.mkdir(parents=True, exist_ok=True)
    _LAST.write_text(h, encoding="utf-8")


def _write_report(answer: str, as_of: str) -> bool:
    """Drop one dated situational report into data/sources/. The subject is made
    salient so the extractor clusters facts into the central "Hormuz status" wiki;
    the framing tells the extractor that dates are authoritative and conflicts /
    uncertainty must be preserved (not flattened)."""
    name = f"hermes-hormuz-{_slug(as_of)}.md"
    dest = _SOURCES / name
    if dest.exists() or (_INGESTED / name).exists():
        return False
    header = [
        f'<!-- hermes:report subject="{_SUBJECT}" -->',
        f"# {_SUBJECT} - situational report",
        "",
        f"**Subject:** {_SUBJECT}",
        f"**Fetched (poll time):** {as_of}",
        "**Source:** external Hermes agent (web-search). This is an incoming situational report; its "
        "web sources are inline below. The DATES INSIDE the report are authoritative for when each "
        "thing was observed - 'Fetched' above is only when we polled. Record dates precisely; where "
        "reports conflict, prefer the more credible and note it; where certainty is not available, "
        "say so rather than guessing.",
        "",
        answer.strip(),
        "",
    ]
    dest.write_text("\n".join(header), encoding="utf-8")
    return True


def _prune(days: int) -> None:
    if days <= 0 or not _INGESTED.is_dir():
        return
    cutoff = time.time() - days * 86400
    for f in _INGESTED.glob("hermes-*.md"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def main() -> None:
    _load_env(_HERE / ".env")
    url = os.getenv("HERMES_ASK_URL", "http://127.0.0.1:8642").strip()
    key = os.getenv("HERMES_API_KEY", "").strip()
    model = os.getenv("HERMES_MODEL", "hermes-agent").strip()
    timeout = int(os.getenv("HERMES_ASK_TIMEOUT", "600"))
    poll = int(os.getenv("HERMES_POLL_SECONDS", "1800"))
    dedup = os.getenv("HERMES_DEDUP", "true").strip().lower() == "true"
    sample = os.getenv("HERMES_SAMPLE_FILE", "").strip()
    prune_days = int(os.getenv("HERMES_PRUNE_DAYS", "0"))

    question = _load_question()
    if not question:
        log.error("query.md is empty/missing - no question to ask. Exiting.")
        sys.exit(1)

    _SOURCES.mkdir(parents=True, exist_ok=True)

    # DORMANT guard: never call Hermes unless a key is configured OR a dry-run
    # sample is provided. Until then, idle quietly so the supervisor doesn't churn.
    if not key and not sample:
        log.info("idle (not configured): set HERMES_API_KEY to enable, or HERMES_SAMPLE_FILE to "
                 "dry-run. Sleeping; will NOT call Hermes.")
        while True:
            time.sleep(max(poll, 300))

    mode = "DRY-RUN (sample)" if sample else "live"
    log.info("hermes-ingestor ready (%s, subject=%s, poll=%ss, url=%s)",
             mode, _SUBJECT, poll, "(sample)" if sample else url)

    last = _load_last()
    while True:
        try:
            now = datetime.now(timezone.utc)
            as_of = now.isoformat()
            answer = ""
            if sample:
                try:
                    answer = _resolve(sample).read_text(encoding="utf-8").strip()
                except Exception as e:
                    log.warning("sample read error: %s", e)
            else:
                try:
                    log.info("asking hermes (fresh discussion)...")
                    dated_question = f"Today is {now:%Y-%m-%d %H:%M} UTC.\n\n{question}"
                    answer = _ask_hermes(dated_question, url, key, model, timeout)
                except requests.RequestException as e:
                    log.warning("hermes ask error: %s", e)

            if not answer:
                log.info("empty answer; will retry next poll")
            else:
                h = _substance(answer)
                if dedup and h == last:
                    log.info("answer unchanged since last poll; skipping "
                             "(set HERMES_DEDUP=false to record every tick)")
                elif _write_report(answer, as_of):
                    last = h
                    _save_last(h)
                    log.info("wrote hermes report into data/sources/ (as of %s)", as_of)
            _prune(prune_days)
        except Exception as e:  # noqa: BLE001 — one bad poll must not kill the loop
            log.exception("loop error: %s", e)
        time.sleep(poll)


if __name__ == "__main__":
    main()
