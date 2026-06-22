# example profile

A harmless reference profile that documents the contract. Activate it with:

```
CUSTOM_PROFILE=example
```

It ships a single `wiki_writer.add.md` fragment, so activating it only *appends* a short
note to the wiki writer prompt — the base prompt, the maintainer prompt, and all default
behaviour are otherwise untouched. It has no `ingestor.py`, so the `profile_runner`
sidecar stays idle.

Copy this folder to start a real profile (e.g. `custom-profiles/my_profile/`); real
profiles are gitignored. See `../README.md` for the full contract.
