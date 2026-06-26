# Custom profiles

Optional, self-contained overlays that shape BrainDB's wiki prompts and (optionally)
feed it a custom ingestion source — **with zero effect on default behaviour when none is
active**. One switch, one folder.

A profile is a folder under this directory. You activate it with a single env var in the
repo-root `.env`:

```
CUSTOM_PROFILE=cityfalcon_news            # one profile
CUSTOM_PROFILE=company_base,cityfalcon_news   # several, composed left-to-right
```

When `CUSTOM_PROFILE` is empty (the default), every mechanism below is a strict no-op:
the prompts are the baked-in defaults, byte-for-byte, and the ingestor supervisor sleeps.

Only this `README.md` and the `example/` profile are committed. **Every real profile is
gitignored** (see `.gitignore`) — its prompt fragments are domain-specific and its `.env`
holds credentials.

## What a profile folder may contain

```
custom-profiles/<name>/
├── wiki_maintainer.add.md     # appended to the maintainer prompt
├── wiki_maintainer.replace.md # OR: replaces the maintainer prompt entirely
├── wiki_writer.add.md         # appended to the writer prompt
├── wiki_writer.replace.md     # OR: replaces the writer prompt entirely
├── ingestor.py                # optional: a standalone polling loop (see below)
└── .env                       # optional: the ingestor's own secrets/config
```

All files are optional. A profile that only ships `wiki_writer.add.md` touches nothing
else.

## Prompt shaping — `add` vs `replace`

Two filename conventions, two behaviours (`<target>` ∈ `wiki_maintainer`, `wiki_writer`):

- **`<target>.add.md` — append.** Your text is glued onto the end of the base prompt,
  *after* the base's placeholders are filled in. The base stays intact, so all the
  machine-contract rules it carries (citation tokens, section markers, decision schema)
  are preserved. This is the safe default and covers almost every need.
- **`<target>.replace.md` — full swap.** Your file *becomes* the prompt (the base is
  discarded). It is substituted for the template *before* placeholders are filled, so you
  keep the dynamic tokens — but you also inherit responsibility for the machine contract:
  the maintainer's `{seeds}` / `{wiki_catalog}`, the writer's `%%MODE%%` / `%%CANONICAL%%`
  / `%%WIKI_ID%%` / `%%MEMBERS%%` / `%%CURRENT_BODY%%` / `%%DUPLICATES%%`, the
  `[[ref:UUID]]` citation tokens, the `<!-- section:NAME -->` markers, and the
  `<!-- wiki:meta keywords=… -->` header. Drop these and the pipeline silently breaks.
  Prefer `add` unless you truly need to rewrite the base.

With several active profiles, `add` fragments are concatenated in list order; the last
profile that supplies a `replace` for a target wins.

## Custom ingestion — `ingestor.py`

A profile may ship an `ingestor.py`: a standalone, long-running script (its own poll
loop) that feeds BrainDB. The committed `profile_runner` sidecar
(`braindb/profile_runner.py`) launches each active profile's `ingestor.py` as an isolated
subprocess and restarts it on exit. With no active profile it sleeps — so the base
`docker-compose.yml` carries only a neutral, dormant-by-default service, and the same
`CUSTOM_PROFILE` switch brings up both the prompt shaping and the ingestor.

The simplest, least-coupled ingestor just **writes files into `data/sources/`** and lets
the existing watcher do ingestion + fact-extraction unchanged (the verbatim source is
stored in the DB at ingest, so the file is only a carrier and can be pruned afterwards).

`ingestor.py` should load its own secrets from a sibling `.env`:

```python
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
```

This keeps each profile self-contained and means no profile-specific variable is ever
named in the public `docker-compose.yml`.

## The `example/` profile

`example/` is a harmless, committed reference. Activate it with `CUSTOM_PROFILE=example`
to see the `add` mechanism in action against the writer prompt; it ships no ingestor.
