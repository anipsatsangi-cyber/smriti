"""Pydantic models for Smriti API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory node from the Smriti store."""

    id: str = Field(description="UUID of the memory")
    text: str = Field(description="The memory content")
    tags: list[str] = Field(default_factory=list, description="User-supplied tags")
    kind: str = Field(default="fact", description="Memory kind: decision, fact, event, preference")
    importance: float = Field(default=0.5, description="Importance weight 0.0-1.0")
    created_at: Optional[datetime] = Field(default=None, description="When the memory was created")
    last_accessed_at: Optional[datetime] = Field(default=None, description="When last recalled")
    access_count: int = Field(default=0, description="Number of times recalled")
    token_count: int = Field(default=0, description="Estimated token count")
    superseded_by: Optional[str] = Field(default=None, description="UUID of replacement memory")
    supersedes: Optional[str] = Field(default=None, description="UUID of memory this replaces")


class RecallHit(BaseModel):
    """A single hit from a recall query."""

    node: Memory = Field(description="The recalled memory")
    final_score: float = Field(description="Combined retrieval score")
    fingerprint_sim: float = Field(default=0.0, description="HDC fingerprint similarity")
    ppr_score: float = Field(default=0.0, description="Personalized PageRank score")
    decay_factor: float = Field(default=1.0, description="Temporal decay factor")
    from_hippocampus: bool = Field(default=False, description="True if from recent buffer")


class RecallResult(BaseModel):
    """Result of a recall query."""

    hits: list[RecallHit] = Field(default_factory=list, description="Ranked recall hits")
    tokens_used: int = Field(default=0, description="Tokens consumed by returned memories")
    tokens_budget: int = Field(default=0, description="Token budget requested")
    candidates_considered: int = Field(default=0, description="Total candidates evaluated")


class StoreStats(BaseModel):
    """Storage-level statistics."""

    total_memories: int = 0
    active_memories: int = 0
    superseded_memories: int = 0
    total_edges: int = 0
    total_tokens: int = 0


class Stats(BaseModel):
    """Full Smriti engine statistics."""

    store: StoreStats = Field(default_factory=StoreStats)
    hippocampus_size: int = 0
    hippocampus_capacity: int = 0
    neocortex_size: int = 0
    neocortex_edges: int = 0


class ConsolidationResult(BaseModel):
    """Result of a consolidation pass."""

    processed: int = 0
    promoted: int = 0
    reinforced: int = 0
    dropped: int = 0
    edges_created: int = 0
