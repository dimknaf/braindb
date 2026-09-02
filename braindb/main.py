import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from braindb.db import get_conn
from braindb.routers import agent, entities, integrations, memory, relations, wiki
from braindb.services import wiki_jobs
from braindb.services.activity_log import log_activity
from braindb.services.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Captured at import = process start. Startup uses it to release wiki jobs
# whose claims died with the PREVIOUS process (see release_stale_assigned).
_PROCESS_START = datetime.now(timezone.utc)

app = FastAPI(
    title="BrainDB",
    description="Memory database and REST API for LLM agents",
    version="0.9.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(entities.router)
app.include_router(relations.router)
app.include_router(memory.router)
app.include_router(agent.router)
app.include_router(wiki.router)
# External-integration endpoints (Hermes memory provider + similar clients).
# Additive only; see braindb/routers/integrations.py.
app.include_router(integrations.router)


@app.on_event("startup")
def startup():
    """Release restart-orphaned wiki claims, then load embeddings."""
    # Agent runs execute only inside this process (scheduler/watcher are
    # HTTP clients), so any wiki_job still `assigned` from before this
    # process started belongs to a run that no longer exists. Release them
    # now instead of letting them sit dark for the full lease. Runs before
    # the slow embedding load so recovery is immediate; the schema is
    # guaranteed present because migrations run in the container command
    # before uvicorn (see docker-compose*.yml).
    try:
        with get_conn() as conn:
            released = wiki_jobs.release_stale_assigned(conn, _PROCESS_START)
            if released:
                log_activity(conn, "wiki_jobs_release", None, None,
                             details={"released": released,
                                      "reason": "assigned before process start"})
                logging.getLogger(__name__).info(
                    "released %d restart-orphaned wiki job claim(s)", released)
    except Exception:
        # A DB hiccup here must never block the API from starting — the
        # lease remains the fallback for anything not released.
        logging.getLogger(__name__).exception(
            "startup release of stale assigned jobs failed; lease will cover")
    emb = get_embedding_service()
    emb.initialize()


@app.get("/health")
def health():
    emb = get_embedding_service()
    return {
        "status": "ok",
        "embeddings": emb.is_available(),
    }
