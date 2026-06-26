"""
Hacker News ingestor (hackernews profile) — a KEYLESS worked example for the
custom-profiles SKILL. Launched by `braindb.profile_runner` when CUSTOM_PROFILE
includes `hackernews`. No API key, no auth.

It polls the public Hacker News Firebase API and writes one Markdown file per new
top story into `data/sources/`, where the existing watcher ingests + extracts it.
The story title + source domain are made salient so the extraction step tags
facts with the company/project/technology named, which clusters them into entity
wikis.

Config (all optional, NO key) — from this folder's own `.env`:
    HN_LIMIT          how many top stories to scan per poll (default 30)
    HN_POLL_SECONDS   loop cadence (default 600)
    HN_PRUNE_DAYS     delete ingested hn-*.md older than this (default 14; 0=keep)
"""
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]          # custom-profiles/hackernews -> repo root
_SOURCES = _REPO_ROOT / "data" / "sources"
_INGESTED = _SOURCES / "ingested"
_STATE = _HERE / ".state" / "seen.txt"

_HN = "https://hacker-news.firebaseio.com/v0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [hn-ingestor] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("hn-ingestor")


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


def _domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _fetch_json(path: str):
    r = requests.get(f"{_HN}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _story(item_id: int) -> dict | None:
    """Fetch one HN item; return normalised story fields, or None to skip
    (non-story / dead / no title). Lets requests errors propagate so the caller
    can retry next poll instead of marking the id seen."""
    it = _fetch_json(f"item/{item_id}.json")
    if not isinstance(it, dict) or it.get("type") != "story":
        return None
    if it.get("dead") or it.get("deleted") or not it.get("title"):
        return None
    ts = it.get("time")
    posted = (datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
              if isinstance(ts, (int, float)) else "")
    url = it.get("url") or ""
    return {
        "id": str(it["id"]),
        "title": str(it["title"]).strip(),
        "url": url,
        "source": _domain(url) or "Hacker News (self post)",
        "by": it.get("by", ""),
        "posted": posted,
        "score": it.get("score", 0),
        "text": (it.get("text") or "").strip(),
    }


def _write_story(s: dict) -> bool:
    """Write one hn-*.md into data/sources/. Returns True if a file was written."""
    name = f"hn-{_slug(s['id'])}.md"
    dest = _SOURCES / name
    if dest.exists() or (_INGESTED / name).exists():
        return False
    header = [
        f"**Source:** {s['source']}",
        f"**By:** {s['by']}",
        f"**Posted:** {s['posted']}",
        f"**HN score:** {s['score']}",
    ]
    if s["url"]:
        header.append(f"**URL:** {s['url']}")
    body = f"# {s['title']}\n\n" + "\n".join(header) + f"\n\n{s['title']}\n"
    if s["text"]:
        body += f"\n{s['text']}\n"
    dest.write_text(body, encoding="utf-8")
    return True


def _load_seen() -> set[str]:
    try:
        return set(_STATE.read_text(encoding="utf-8").split())
    except Exception:
        return set()


def _save_seen(seen: set[str], cap: int = 20000) -> None:
    _STATE.parent.mkdir(parents=True, exist_ok=True)
    _STATE.write_text("\n".join(list(seen)[-cap:]), encoding="utf-8")


def _prune(days: int) -> None:
    if days <= 0 or not _INGESTED.is_dir():
        return
    cutoff = time.time() - days * 86400
    for f in _INGESTED.glob("hn-*.md"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def main() -> None:
    _load_env(_HERE / ".env")
    limit = int(os.getenv("HN_LIMIT", "30"))
    poll = int(os.getenv("HN_POLL_SECONDS", "600"))
    prune_days = int(os.getenv("HN_PRUNE_DAYS", "14"))

    _SOURCES.mkdir(parents=True, exist_ok=True)
    log.info("hn-ingestor ready (top %d, poll=%ss, sources=%s)", limit, poll, _SOURCES)

    seen = _load_seen()
    while True:
        try:
            written = 0
            try:
                ids = _fetch_json("topstories.json")[:limit]
            except requests.RequestException as e:
                log.warning("topstories fetch error: %s", e)
                ids = []
            for sid in ids:
                key = str(sid)
                if key in seen:
                    continue
                try:
                    s = _story(sid)
                except requests.RequestException as e:
                    log.warning("item %s fetch error: %s", sid, e)
                    continue   # transient — retry next poll, don't mark seen
                seen.add(key)
                if s and _write_story(s):
                    written += 1
            if written:
                log.info("wrote %d new story file(s) into data/sources/", written)
                _save_seen(seen)
            _prune(prune_days)
        except Exception as e:
            log.exception("loop error: %s", e)
        time.sleep(poll)


if __name__ == "__main__":
    main()
