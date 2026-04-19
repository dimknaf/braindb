from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    entity_types: list[str] | None = None   # filter to specific types
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)


class ContextRequest(BaseModel):
    query: str | None = Field(default=None, min_length=1)
    queries: list[str] | None = None        # multi-query: runs each, merges seeds, unified ranking
    entity_types: list[str] | None = None
    max_results: int = Field(default=30, ge=1, le=100)
    max_depth: int = Field(default=3, ge=1, le=3)
    min_relevance: float = Field(default=0.05, ge=0.0, le=1.0)
    include_always_on_rules: bool = True
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def at_least_one_query(self):
        if not self.query and not self.queries:
            raise ValueError("Provide 'query' or 'queries'")
        return self


class SearchResultItem(BaseModel):
    id: UUID
    entity_type: str
    title: str | None
    content: str
    summary: str | None
    keywords: list[str]
    importance: float
    source: str | None = None     # provenance
    notes: str | None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    accessed_at: datetime | None = None
    access_count: int = 0
    search_score: float           # raw text/fuzzy match score
    effective_importance: float   # after temporal decay + reinforcement
    depth: int = 0                # 0 = direct match, 1-3 = graph hops
    accumulated_relevance: float = 1.0
    final_rank: float             # search_score * effective_importance * accumulated_relevance
    # type-specific extras
    ext: dict[str, Any] = {}


class ContextResponse(BaseModel):
    query: str                                  # first query (backward-compat)
    queries: list[str] = []                     # all queries used
    items: list[SearchResultItem]
    always_on_rules: list[SearchResultItem] = []
    total_found: int
