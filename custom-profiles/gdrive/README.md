# gdrive profile (Google Drive folder follower)

Follows a Google Drive folder and feeds BrainDB **only new files and only the changed parts of
edited docs** — generic and **env-driven**, so any user points it at their *own* folder and
credentials via `.env`, with no code change. See [`../SKILL.md`](../SKILL.md) for the profile
mechanism.

## What it does

- `ingestor.py` lists a Drive folder (recursively), exports native **Google Docs → markdown**,
  **Sheets → CSV**, **Slides → text**, and drops the result into `data/sources/`, where the
  existing watcher ingests + extracts it. (Binary `.docx`/`.pdf` are skipped — the pipeline is
  text-only.)
- **Incremental.** A manifest (`.state/manifest.json`, `{file_id: change_key}`) skips unchanged
  docs entirely. `change_key` = `md5Checksum` (uploads) or `modifiedTime` (native Docs).
- **Diff-aware (default `GDRIVE_INGEST_MODE=diff`).** First sight of a doc ingests the full
  baseline; on later edits it emits **only the changed/new markdown sections**, each wrapped in a
  self-describing breadcrumb (*what part it is* — heading path + position — and *where it belongs*
  — doc title/id/URL, "full doc already in memory"). Set `GDRIVE_INGEST_MODE=full` to re-ingest the
  whole doc on every change instead.

This profile shapes ingestion only — it ships no wiki prompt fragments, so the wiki maintainer/
writer keep their defaults. (Add `wiki_*.add.md` here later if you want shaped pages.)

## One-time Google setup (any user, their own Drive)

1. In Google Cloud, create (or reuse) a **service account** and **enable the Drive API** for its
   project. Create a **JSON key** and save it as `service-account.json` in this folder (gitignored),
   or point `GDRIVE_CREDENTIALS_FILE` at it.
2. **Share the folder with the service account.** A service account only sees what is shared with
   it: open the Drive folder → Share → add the service-account email
   (`…iam.gserviceaccount.com`) as **Viewer**.
   *(For personal docs you don't want to share to a service account, an OAuth user flow is the
   alternative — a future option, not wired here.)*

## Run it

1. `cp .env.example .env`, then set `GDRIVE_FOLDER_ID` (the last segment of the folder URL
   `https://drive.google.com/drive/folders/<ID>`) and place the JSON key.
2. In the repo-root `.env`: `CUSTOM_PROFILE=gdrive` (or append, e.g. `gdrive,hackernews`).
3. `docker compose up -d`.
4. Watch it flow:
   ```bash
   docker compose logs -f profile_runner     # "launching ingestor for profile 'gdrive'",
                                             # then "found N / new M / changed K / skipped U"
   ls data/sources/                          # gdrive-*.md / gdrive-*.csv
   curl -s -X POST http://localhost:8000/api/v1/memory/context \
     -H "Content-Type: application/json" -d '{"queries":["<a doc title>"]}'
   ```

### Changing the tracked folder

Edit `GDRIVE_FOLDER_ID` in `.env`, share the new folder with the service account, and
`docker compose up -d` (or restart `profile_runner`). To switch credentials, point
`GDRIVE_CREDENTIALS_FILE` at a different JSON key. No code change either way.

## Honest notes / limits

- A service account sees **only folders shared with it** (or that it owns).
- **Diff mode** anchors a delta's facts to the doc's existing in-memory cluster via the shared doc
  title + the breadcrumb (the extractor can't live-fetch the source URL — it's preserved as
  provenance). Editing a section may restate a prior fact (small near-dup surface the wiki
  maintainer consolidates).
- **Deletions/trashing** are noted in the diff header but do **not** auto-retract already-extracted
  facts.
- The Google client libs (`google-api-python-client`, `google-auth`) aren't in the BrainDB image;
  the ingestor **self-installs** them on first run (the image stays untouched).
