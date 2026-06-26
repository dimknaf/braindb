"""
Generic custom-profile ingestor supervisor — ONE loop, same posture as
ingest_watcher.py / wiki_scheduler.py. DORMANT unless CUSTOM_PROFILE is set.

For each active profile (CUSTOM_PROFILE, comma-separated) that ships an
`ingestor.py`, this launches `python <profile>/ingestor.py` as an ISOLATED
subprocess and keeps it alive (restart with backoff on exit). It holds no
profile-specific knowledge and never imports profile code: a broken ingestor
can only crash its own subprocess, never the api / watcher / wiki.

No active profile, or no active profile ships an ingestor.py -> the runner
sleeps and makes zero calls (exactly like WIKI_ENABLED=false). This is what
lets ONE switch (CUSTOM_PROFILE) bring up prompt-shaping AND a profile's
ingestor together, while the base docker-compose.yml carries only a neutral,
dormant-by-default service. See custom-profiles/README.md.
"""
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # braindb/.. -> repo root

# Liveness-check cadence and the minimum spacing between relaunch attempts for a
# crashing ingestor (prevents a tight crash-loop). Both are seconds.
POLL_INTERVAL = int(os.getenv("PROFILE_RUNNER_INTERVAL", "15"))
RESTART_BACKOFF = int(os.getenv("PROFILE_RUNNER_BACKOFF", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [profile-runner] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("profile-runner")


def _profiles_root() -> Path:
    base = os.getenv("CUSTOM_PROFILE_DIR", "custom-profiles").strip() or "custom-profiles"
    p = Path(base)
    return p if p.is_absolute() else _REPO_ROOT / p


def _active_profiles() -> list[str]:
    raw = os.getenv("CUSTOM_PROFILE", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()]


def _ingestors() -> dict[str, Path]:
    """{profile_name: ingestor.py path} for active profiles that ship one."""
    root = _profiles_root()
    out: dict[str, Path] = {}
    for name in _active_profiles():
        script = root / name / "ingestor.py"
        if script.is_file():
            out[name] = script
    return out


def main() -> None:
    ingestors = _ingestors()
    if not ingestors:
        log.info("no active profile ingestor (CUSTOM_PROFILE=%r). Idle.",
                 os.getenv("CUSTOM_PROFILE", ""))
        # Sleep forever — container stays Up, zero calls. Toggle via env + restart.
        while True:
            time.sleep(3600)

    log.info("supervising %d profile ingestor(s): %s",
             len(ingestors), ", ".join(ingestors))

    procs: dict[str, subprocess.Popen] = {}
    launched_at: dict[str, float] = {}

    def _launch(name: str, script: Path) -> None:
        log.info("launching ingestor for profile '%s': %s", name, script)
        # Isolated subprocess; cwd at repo root so the script's relative paths
        # (its own folder, data/sources/) resolve exactly as in the containers.
        procs[name] = subprocess.Popen([sys.executable, str(script)], cwd=str(_REPO_ROOT))
        launched_at[name] = time.time()

    for name, script in ingestors.items():
        _launch(name, script)

    while True:
        try:
            for name, script in ingestors.items():
                proc = procs.get(name)
                if proc is None:
                    _launch(name, script)
                    continue
                if proc.poll() is None:
                    continue  # still running
                # Exited — relaunch once the backoff window (since last launch)
                # has elapsed, so a fast crash-loop is throttled while a
                # long-lived process that dies restarts promptly.
                if time.time() - launched_at[name] >= RESTART_BACKOFF:
                    log.warning("ingestor '%s' exited (code=%s); restarting",
                                name, proc.returncode)
                    _launch(name, script)
        except Exception as e:
            log.exception("supervisor loop error: %s", e)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
