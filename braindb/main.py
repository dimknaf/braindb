import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from braindb.routers import agent, entities, integrations, memory, relations, wiki
from braindb.services.embedding_service import get_embedding_service

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    """Initialize the embedding service on startup."""
    emb = get_embedding_service()
    emb.initialize()


@app.get("/health")
def health():
    emb = get_embedding_service()
    return {
        "status": "ok",
        "embeddings": emb.is_available(),
    }
