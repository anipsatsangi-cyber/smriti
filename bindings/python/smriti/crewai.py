"""CrewAI memory adapter for Smriti.

Provides a Smriti-backed storage backend for CrewAI's memory system,
giving your crew persistent, structured memory across runs.

Usage:
    >>> from smriti.crewai import SmritiCrewMemory
    >>> memory = SmritiCrewMemory(base_url="http://localhost:4000")
    >>> # Pass to your CrewAI Crew
    >>> crew = Crew(agents=[...], tasks=[...], memory=memory)
"""

from __future__ import annotations

from typing import Any, Optional

from smriti.client import SmritiClient


class SmritiCrewMemory:
    """CrewAI-compatible memory storage backed by Smriti.

    Implements the interface expected by CrewAI's memory system,
    storing task outputs and agent observations as structured memories.

    Attributes:
        base_url: Smriti server URL.
        budget: Token budget for recall queries.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        budget: int = 2000,
        agent: Optional[str] = None,
    ):
        self.client = SmritiClient(base_url=base_url, agent=agent or "crewai")
        self.budget = budget

    def save(
        self,
        value: str,
        metadata: Optional[dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> str:
        """Save a memory from a CrewAI agent.

        Args:
            value: The content to remember.
            metadata: Optional metadata dict (converted to tags).
            agent: Optional agent name for scoping.

        Returns:
            UUID of the stored memory.
        """
        tags = []
        kind = "fact"
        importance = 0.5

        if metadata:
            tags = [f"{k}:{v}" for k, v in metadata.items() if isinstance(v, str)]
            kind = metadata.get("kind", "fact")
            importance = float(metadata.get("importance", 0.5))

        if agent:
            tags.append(f"agent:{agent}")

        return self.client.remember(
            value,
            tags=tags,
            kind=kind,
            importance=importance,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search memories relevant to a query.

        Args:
            query: Search query.
            limit: Maximum results (approximated via budget).
            score_threshold: Minimum score to include.

        Returns:
            List of dicts with 'content', 'score', and 'metadata' keys.
        """
        # Approximate: each memory ~100 tokens
        budget = limit * 100
        result = self.client.recall(query, budget=budget)

        output = []
        for hit in result.hits:
            if hit.final_score < score_threshold:
                continue
            output.append({
                "content": hit.node.text,
                "score": hit.final_score,
                "metadata": {
                    "id": hit.node.id,
                    "kind": hit.node.kind,
                    "tags": hit.node.tags,
                    "importance": hit.node.importance,
                    "from_hippocampus": hit.from_hippocampus,
                },
            })
        return output[:limit]

    def reset(self) -> None:
        """Reset is a no-op — Smriti uses natural decay."""
        pass
