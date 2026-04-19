from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------------ #
# Shared base                                                         #
# ------------------------------------------------------------------ #

class EntityBase(BaseModel):
    title: str | None = None
    content: str
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str | None = None       # provenance: "user-stated", "agent-inference", "document", "third-party"
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRead(EntityBase):
    id: UUID
    entity_type: str
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime | None = None
    access_count: int

    model_config = {"from_attributes": True}


# ------------------------------------------------------------------ #
# THOUGHT                                                             #
# ------------------------------------------------------------------ #

class ThoughtCreate(EntityBase):
    certainty: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str | None = None
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0)


class ThoughtRead(EntityRead):
    entity_type: Literal["thought"] = "thought"
    certainty: float
    context: str | None
    emotional_valence: float


class ThoughtUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    certainty: float | None = Field(default=None, ge=0.0, le=1.0)
    context: str | None = None
    emotional_valence: float | None = Field(default=None, ge=-1.0, le=1.0)


# ------------------------------------------------------------------ #
# FACT                                                                #
# ------------------------------------------------------------------ #

class FactCreate(EntityBase):
    certainty: float = Field(default=0.8, ge=0.0, le=1.0)
    is_verified: bool = False
    source_entity_id: UUID | None = None


class FactRead(EntityRead):
    entity_type: Literal["fact"] = "fact"
    certainty: float
    is_verified: bool
    source_entity_id: UUID | None


class FactUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    certainty: float | None = Field(default=None, ge=0.0, le=1.0)
    is_verified: bool | None = None
    source_entity_id: UUID | None = None


# ------------------------------------------------------------------ #
# SOURCE                                                              #
# ------------------------------------------------------------------ #

class SourceCreate(EntityBase):
    url: str
    domain: str | None = None
    http_status: int | None = None


class SourceRead(EntityRead):
    entity_type: Literal["source"] = "source"
    url: str
    domain: str | None
    http_status: int | None
    last_checked_at: str | None


class SourceUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    url: str | None = None
    domain: str | None = None
    http_status: int | None = None


# ------------------------------------------------------------------ #
# DATASOURCE                                                          #
# ------------------------------------------------------------------ #

class DatasourceCreate(EntityBase):
    file_path: str | None = None
    url: str | None = None
    content_hash: str | None = None
    word_count: int | None = None
    language: str = "en"

    @model_validator(mode="after")
    def require_file_or_url(self) -> "DatasourceCreate":
        if not self.file_path and not self.url:
            raise ValueError("Either file_path or url must be provided")
        return self


class DatasourceRead(EntityRead):
    entity_type: Literal["datasource"] = "datasource"
    file_path: str | None
    url: str | None
    content_hash: str | None
    word_count: int | None
    language: str


class DatasourceUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    file_path: str | None = None
    url: str | None = None
    content_hash: str | None = None
    word_count: int | None = None
    language: str | None = None

    @model_validator(mode="after")
    def prevent_clearing_both_identifiers(self) -> "DatasourceUpdate":
        data = self.model_fields_set
        if "file_path" in data and "url" in data and not self.file_path and not self.url:
            raise ValueError("Cannot clear both file_path and url — at least one must remain")
        return self


# ------------------------------------------------------------------ #
# RULE                                                                #
# ------------------------------------------------------------------ #

class RuleCreate(EntityBase):
    always_on: bool = False
    category: str | None = None
    priority: int = Field(default=50, ge=1, le=100)
    is_active: bool = True


class RuleRead(EntityRead):
    entity_type: Literal["rule"] = "rule"
    always_on: bool
    category: str | None
    priority: int
    is_active: bool


class RuleUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    summary: str | None = None
    keywords: list[str] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None
    always_on: bool | None = None
    category: str | None = None
    priority: int | None = Field(default=None, ge=1, le=100)
    is_active: bool | None = None


# ------------------------------------------------------------------ #
# Generic entity read (union) used in list endpoints                  #
# ------------------------------------------------------------------ #

AnyEntityRead = ThoughtRead | FactRead | SourceRead | DatasourceRead | RuleRead
