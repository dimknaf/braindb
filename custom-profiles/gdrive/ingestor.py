"""
Google Drive folder-follower ingestor (gdrive profile).

Launched by `braindb.profile_runner` when CUSTOM_PROFILE includes `gdrive`. It
does NOT touch BrainDB internals: it lists a Google Drive folder, exports native
Google Docs/Sheets/Slides to text, and writes files into `data/sources/`, where
the existing watcher ingests + extracts them. Generic + env-driven: every user
points it at their OWN folder and credentials via this folder's `.env` — no code
edit needed.

Incremental by design — it ingests ONLY new files and ONLY the changed parts of
edited docs:
  * a manifest (`.state/manifest.json`, `{file_id: change_key}`) decides which
    docs are new/changed/unchanged. change_key = md5Checksum (uploads) or
    modifiedTime (native Docs, which have no md5). Unchanged docs are skipped
    with no download.
  * a per-doc snapshot (`.state/snapshots/<file_id>.md`) lets us, in `diff` mode
    (default), emit only the changed/new markdown sections of an edited doc, each
    wrapped in a self-describing breadcrumb (what part it is + where it belongs),
    instead of re-ingesting the whole document (which would duplicate facts and
    waste LLM time). First sight of a doc ingests the full baseline.

Config — from this folder's own `.env` (copy `.env.example` -> `.env`):
    GDRIVE_FOLDER_ID                 (required) the Drive folder to follow
    GDRIVE_CREDENTIALS_FILE          service-account JSON (default service-account.json here)
    GDRIVE_INGEST_MODE               diff (default) | full
    GDRIVE_RECURSIVE                 walk subfolders (default true)
    GDRIVE_POLL_SECONDS              loop cadence (default 300)
    GDRIVE_EXPORT_DOC/SHEET/SLIDES   export mime types (defaults md / csv / plain)
    GDRIVE_PRUNE_DAYS                delete ingested gdrive-*.* older than this (default 0 = keep)
    GDRIVE_SKIP_EXISTING_ON_FIRST_RUN  seed state without ingesting existing docs (default false)

IMPORTANT: a service account only sees what is SHARED with it. Share the target
folder with the service-account email (…iam.gserviceaccount.com) as Viewer.
"""
import difflib
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]          # custom-profiles/gdrive -> repo root
_SOURCES = _REPO_ROOT / "data" / "sources"
_INGESTED = _SOURCES / "ingested"
_STATE = _HERE / ".state"
_MANIFEST = _STATE / "manifest.json"
_SNAP_DIR = _STATE / "snapshots"

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_FOLDER_MIME = "application/vnd.google-apps.folder"
_FIELDS = ("nextPageToken, files(id,name,mimeType,modifiedTime,md5Checksum,"
           "webViewLink,trashed)")
# Extensions the BrainDB watcher accepts as plain text (binary docx/pdf are not
# ingestible, so we skip them rather than drop a file the watcher would reject).
_TEXT_EXTS = {"md", "txt", "json", "yaml", "yml", "csv", "log", "html", "xml"}
_MIME_EXT = {"text/markdown": "md", "text/plain": "txt", "text/csv": "csv"}
_MAX_DEPTH = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gdrive-ingestor] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("gdrive-ingestor")


# --------------------------------------------------------------------------- #
# env + small helpers (mirrors custom-profiles/hackernews/ingestor.py)
# --------------------------------------------------------------------------- #
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


def _resolve(p: str) -> str:
    """Resolve a configured path against repo root / profile dir if relative."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    for base in (_REPO_ROOT, _HERE):
        cand = base / pp
        if cand.is_file():
            return str(cand)
    return str(_REPO_ROOT / pp)


def _ensure_deps() -> None:
    """Self-bootstrap the Google client libs so the BrainDB image stays untouched."""
    try:
        import googleapiclient  # noqa: F401
        import google.oauth2.service_account  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    pkgs = ["google-api-python-client", "google-auth"]
    log.info("installing Google Drive client libs: %s", ", ".join(pkgs))
    for extra in ([], ["--user"]):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "--quiet", *extra, *pkgs])
            return
        except subprocess.CalledProcessError as e:
            log.warning("pip install %sfailed: %s", "(--user) " if extra else "", e)
    log.error("could not install Google Drive libs; exiting so the supervisor retries")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Drive access (auth + list + export) — pattern from ir-evaluation
# --------------------------------------------------------------------------- #
def _build_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds_path = _resolve(os.getenv("GDRIVE_CREDENTIALS_FILE",
                                    str(_HERE / "service-account.json")))
    if not Path(creds_path).is_file():
        log.error("service-account JSON not found at %s "
                  "(set GDRIVE_CREDENTIALS_FILE). Exiting.", creds_path)
        sys.exit(1)
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_children(service, folder_id: str) -> list[dict]:
    files, page = [], None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=_FIELDS,
            pageSize=1000,
            pageToken=page,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page = resp.get("nextPageToken")
        if not page:
            break
    return files


def _walk(service, folder_id: str, recursive: bool, depth: int = 0,
          path: str = "") -> list[dict]:
    """Return non-folder files under folder_id, each tagged with its `_drivePath`."""
    out = []
    for f in _list_children(service, folder_id):
        if f.get("mimeType") == _FOLDER_MIME:
            if recursive and depth < _MAX_DEPTH:
                sub = f"{path}/{f['name']}" if path else f["name"]
                out.extend(_walk(service, f["id"], recursive, depth + 1, sub))
        else:
            f["_drivePath"] = path
            out.append(f)
    return out


def _content_for(service, f: dict) -> tuple[bytes | None, str | None]:
    """Export/download a file's content as text bytes; (None, None) to skip."""
    from googleapiclient.http import MediaIoBaseDownload
    mime = f.get("mimeType", "")
    fid = f["id"]
    exp_doc = os.getenv("GDRIVE_EXPORT_DOC", "text/markdown")
    exp_sheet = os.getenv("GDRIVE_EXPORT_SHEET", "text/csv")
    exp_slides = os.getenv("GDRIVE_EXPORT_SLIDES", "text/plain")

    def _export(targets: list[str]) -> tuple[bytes | None, str | None]:
        for target in targets:
            try:
                data = service.files().export(fileId=fid, mimeType=target).execute()
                return data, _MIME_EXT.get(target, "txt")
            except Exception as e:  # noqa: BLE001 — try the fallback target
                log.warning("export %s as %s failed: %s", f.get("name"), target, e)
        return None, None

    if mime == "application/vnd.google-apps.document":
        return _export([exp_doc, "text/plain"])
    if mime == "application/vnd.google-apps.spreadsheet":
        return _export([exp_sheet])
    if mime == "application/vnd.google-apps.presentation":
        return _export([exp_slides])
    if mime.startswith("application/vnd.google-apps"):
        log.info("skip native type %s (%s) - no text export", mime, f.get("name"))
        return None, None

    # Uploaded file: only ingest text-ish ones (the watcher is text-only).
    ext = f["name"].rsplit(".", 1)[-1].lower() if "." in f["name"] else ""
    if ext in _TEXT_EXTS:
        try:
            buf, dl, done = io.BytesIO(), None, False
            dl = MediaIoBaseDownload(buf, service.files().get_media(fileId=fid))
            while not done:
                _, done = dl.next_chunk()
            return buf.getvalue(), ext
        except Exception as e:  # noqa: BLE001
            log.warning("download %s failed: %s", f.get("name"), e)
            return None, None
    log.info("skip binary/unsupported %s (%s)", f.get("name"), mime or "?")
    return None, None


# --------------------------------------------------------------------------- #
# section diff (stdlib only)
# --------------------------------------------------------------------------- #
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_sections(md: str) -> list[dict]:
    """Split markdown into ordered sections, each with its heading breadcrumb
    `path` (chain of enclosing headings) and `text` (heading line + body)."""
    sections, stack = [], []
    cur = {"path": "(preamble)", "lines": []}
    for line in md.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append(cur)
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = {"path": " > ".join(t for _, t in stack), "lines": [line]}
        else:
            cur["lines"].append(line)
    sections.append(cur)
    out = []
    for s in sections:
        text = "\n".join(l.rstrip() for l in s["lines"]).strip("\n")
        if text:
            out.append({"path": s["path"], "text": text})
    return out


def _keyed(sections: list[dict]) -> tuple[dict, list[tuple]]:
    """Return ({key: text}, [(key, section)]) with duplicate paths disambiguated."""
    seen, mp, keyed = {}, {}, []
    for s in sections:
        seen[s["path"]] = seen.get(s["path"], 0) + 1
        key = s["path"] if seen[s["path"]] == 1 else f"{s['path']} #{seen[s['path']]}"
        mp[key] = s["text"]
        keyed.append((key, s))
    return mp, keyed


def _diff_header(meta: dict, removed: list[str]) -> str:
    lines = [
        f"<!-- gdrive:diff doc_id={meta['id']} -->",
        "This file is an INCREMENTAL DIFF of a Google Doc: it contains only "
        "changed/new parts, NOT the full document. Treat each part below as an "
        "update to the named document; the full document is already in memory "
        "under its title, and the Source URL is the original.",
        f"**Document:** {meta['title']}",
        f"**Doc id:** {meta['id']}",
    ]
    if meta.get("url"):
        lines.append(f"**Source:** {meta['url']}")
    if meta.get("modified"):
        lines.append(f"**Modified:** {meta['modified']}")
    if removed:
        lines.append("_Parts removed since last sync (noted only; facts not "
                     f"auto-retracted): {'; '.join(removed)}_")
    return "\n".join(lines)


def _breadcrumb(meta: dict, path: str, n: int, total: int) -> str:
    return (f'> [DIFF | part: "{path}" | position {n}/{total} | belongs to: '
            f'"{meta["title"]}" ({meta["id"]}) | full doc in memory under '
            f'"{meta["title"]}" | source: {meta.get("url", "")}]')


def _build_diff(old_md: str, new_md: str, meta: dict) -> str | None:
    """Self-describing delta of changed/new sections, or None if nothing changed."""
    new_secs = _split_sections(new_md)
    if len(new_secs) < 2:                      # no usable headings -> line diff
        return _build_line_diff(old_md, new_md, meta)
    old_map, _ = _keyed(_split_sections(old_md))
    new_map, new_keyed = _keyed(new_secs)
    total = len(new_keyed)
    changed = [(i, s) for i, (key, s) in enumerate(new_keyed, 1)
               if old_map.get(key) != s["text"]]
    removed = [k for k in old_map if k not in new_map]
    if not changed:
        return None
    parts = [_diff_header(meta, removed)]
    for i, s in changed:
        parts.append(_breadcrumb(meta, s["path"], i, total))
        parts.append(s["text"])
    return "\n\n".join(parts) + "\n"


def _build_line_diff(old_md: str, new_md: str, meta: dict) -> str | None:
    """Fallback for heading-less docs (e.g. exported sheets): emit added/changed
    lines as plain text under a breadcrumb."""
    diff = list(difflib.unified_diff(old_md.splitlines(), new_md.splitlines(),
                                     lineterm="", n=0))
    added = [l[1:] for l in diff if l.startswith("+") and not l.startswith("+++")]
    added_text = "\n".join(l for l in added if l.strip())
    if not added_text:
        return None
    crumb = (f'> [DIFF | line-level changes | belongs to: "{meta["title"]}" '
             f'({meta["id"]}) | full doc in memory under "{meta["title"]}" | '
             f'source: {meta.get("url", "")}]')
    return f"{_diff_header(meta, [])}\n\n{crumb}\n\n{added_text}\n"


def _full_header(meta: dict) -> str:
    lines = [
        f"<!-- gdrive:full doc_id={meta['id']} -->",
        f"**Document:** {meta['title']}",
        f"**Doc id:** {meta['id']}",
    ]
    if meta.get("drive_path"):
        lines.append(f"**Drive path:** {meta['drive_path']}")
    if meta.get("url"):
        lines.append(f"**Source:** {meta['url']}")
    if meta.get("modified"):
        lines.append(f"**Modified:** {meta['modified']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def _load_manifest() -> dict:
    try:
        return json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(m: dict) -> None:
    _STATE.mkdir(parents=True, exist_ok=True)
    _MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_snapshot(fid: str) -> str:
    try:
        return (_SNAP_DIR / f"{_slug(fid)}.md").read_text(encoding="utf-8")
    except Exception:
        return ""


def _save_snapshot(fid: str, text: str) -> None:
    _SNAP_DIR.mkdir(parents=True, exist_ok=True)
    (_SNAP_DIR / f"{_slug(fid)}.md").write_text(text, encoding="utf-8")


def _change_key(f: dict) -> str:
    return f.get("md5Checksum") or f.get("modifiedTime") or ""


def _write_source(name: str, content: str) -> bool:
    dest = _SOURCES / name
    if dest.exists() or (_INGESTED / name).exists():
        return False
    dest.write_text(content, encoding="utf-8")
    return True


def _prune(days: int) -> None:
    if days <= 0 or not _INGESTED.is_dir():
        return
    cutoff = time.time() - days * 86400
    for f in _INGESTED.glob("gdrive-*.*"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# per-file processing
# --------------------------------------------------------------------------- #
def _process(service, f: dict, manifest: dict, mode: str, counts: dict,
             seed_only: bool) -> None:
    fid, cur = f["id"], _change_key(f)
    prev = manifest.get(fid)
    if prev is not None and prev == cur:
        counts["skipped"] += 1
        return                                  # unchanged: no download, no write

    data, ext = _content_for(service, f)
    if data is None:                            # unsupported/native/error: record + move on
        manifest[fid] = cur
        return
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)

    if seed_only:                               # first run, --skip-existing: snapshot only
        _save_snapshot(fid, text)
        manifest[fid] = cur
        counts["seeded"] += 1
        return

    meta = {"id": fid, "title": f["name"], "url": f.get("webViewLink", ""),
            "modified": f.get("modifiedTime", ""), "drive_path": f.get("_drivePath", "")}
    is_new = prev is None

    if is_new or mode == "full":
        out_text, out_ext, kind = _full_header(meta) + "\n\n" + text, ext, \
            ("new" if is_new else "changed")
    else:
        out_text = _build_diff(_load_snapshot(fid), text, meta)
        out_ext, kind = "md", "changed"
        if out_text is None:                    # no real content change (whitespace/meta)
            _save_snapshot(fid, text)
            manifest[fid] = cur
            counts["skipped"] += 1
            return

    h8 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    name = f"gdrive-{_slug(f['name'])}-{h8}.{out_ext}"
    if _write_source(name, out_text):
        counts[kind] += 1
    _save_snapshot(fid, text)
    manifest[fid] = cur


def main() -> None:
    _load_env(_HERE / ".env")
    _ensure_deps()

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        log.error("GDRIVE_FOLDER_ID not set (custom-profiles/gdrive/.env). Exiting.")
        sys.exit(1)
    mode = os.getenv("GDRIVE_INGEST_MODE", "diff").strip().lower()
    recursive = os.getenv("GDRIVE_RECURSIVE", "true").strip().lower() == "true"
    poll = int(os.getenv("GDRIVE_POLL_SECONDS", "300"))
    prune_days = int(os.getenv("GDRIVE_PRUNE_DAYS", "0"))
    skip_existing = os.getenv("GDRIVE_SKIP_EXISTING_ON_FIRST_RUN",
                              "false").strip().lower() == "true"

    _SOURCES.mkdir(parents=True, exist_ok=True)
    service = _build_service()
    log.info("gdrive-ingestor ready (folder=%s, mode=%s, recursive=%s, poll=%ss)",
             folder_id, mode, recursive, poll)

    manifest = _load_manifest()
    first_run = not manifest
    while True:
        try:
            seed_only = first_run and skip_existing
            counts = {"new": 0, "changed": 0, "skipped": 0, "seeded": 0}
            try:
                files = _walk(service, folder_id, recursive)
            except Exception as e:  # noqa: BLE001 — transient Drive error, retry next poll
                log.warning("drive list error: %s", e)
                files = []
            for f in files:
                try:
                    _process(service, f, manifest, mode, counts, seed_only)
                except Exception as e:  # noqa: BLE001 — one bad doc must not kill the loop
                    log.warning("process %s error: %s", f.get("name"), e)
            _save_manifest(manifest)
            if seed_only:
                log.info("first run: seeded %d doc(s) (changes-only from now)",
                         counts["seeded"])
            log.info("found %d / new %d / changed %d / skipped %d",
                     len(files), counts["new"], counts["changed"], counts["skipped"])
            _prune(prune_days)
            first_run = False
        except Exception as e:  # noqa: BLE001
            log.exception("loop error: %s", e)
        time.sleep(poll)


if __name__ == "__main__":
    main()
