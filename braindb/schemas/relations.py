from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


RELATION_TYPES = Literal[
    "supports",
    "contradicts",
    "elaborates",
    "refers_to",
    "derived_from",
    "similar_to",
    "is_example_of",
    "challenges",
    "tagged_with",
]


class RelationCreate(BaseModel):
    from_entity_id: UUID
    to_entity_id: UUID
    relation_type: RELATION_TYPES
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_bidirectional: bool = False
    description: str | None = None   # why this relation exists
    notes: str | None = None         # evolving commentary


class RelationRead(BaseModel):
    id: UUID
    from_entity_id: UUID
    to_entity_id: UUID
    relation_type: str
    relevance_score: float
    importance_score: float
    is_bidirectional: bool
    description: str | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class RelationUpdate(BaseModel):
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    is_bidirectional: bool | None = None
    description: str | None = None
    notes: str | None = None
