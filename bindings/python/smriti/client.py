"""Smriti HTTP client — thin wrapper around the Smriti REST API.

Usage:
    >>> from smriti import SmritiClient
    >>> client = SmritiClient("http://localhost:4000")
    >>> mid = client.remember("The auth module uses JWT RS256", tags=["auth"])
    >>> result = client.recall("authentication")
    >>> print(result.hits[0].node.text)
    'The auth module uses JWT RS256'
"""

from __future__ import annotations

from typing import Optional

import requests

from smriti.types import (
    ConsolidationResult,
    Memory,
    RecallHit,
    RecallResult,
    Stats,
    StoreStats,
)


class SmritiError(Exception):
    """Raised when a Smriti API call fails."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Smriti API error ({status_code}): {message}")


class SmritiClient:
    """HTTP client for the Smriti memory engine.

    Args:
        base_url: Base URL of the Smriti HTTP server (default: http://localhost:4000).
        timeout: Request timeout in seconds (default: 30).
        agent: Agent scope identifier for multi-tenant isolation.
        user: User scope identifier for multi-tenant isolation.
        session: Session scope identifier.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        timeout: int = 30,
        agent: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "smriti-python/0.1.0",
        })
        self._scope = self._build_scope(agent=agent, user=user, session=session)

    @staticmethod
    def _build_scope(
        *,
        agent: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
    ) -> dict[str, str]:
        """Build the canonical HTTP scope payload."""
        scope: dict[str, str] = {}
        if agent:
            scope["agent_id"] = agent
        if user:
            scope["user_id"] = user
        if session:
            scope["session_id"] = session
        return scope

    def _scope_payload(self) -> dict[str, str]:
        """Normalize scope payloads for backward compatibility."""
        if not self._scope:
            return {}

        scope = dict(self._scope)
        normalized = {
            "agent_id": scope.get("agent_id") or scope.get("agent"),
            "user_id": scope.get("user_id") or scope.get("user"),
            "session_id": scope.get("session_id") or scope.get("session"),
        }
        return {k: v for k, v in normalized.items() if v}

    def set_scope(
        self,
        *,
        agent: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
    ) -> None:
        """Set the default scope used by future operations."""
        self._scope = self._build_scope(agent=agent, user=user, session=session)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        resp = self._session.request(
            method, self._url(path), timeout=self.timeout, **kwargs
        )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise SmritiError(resp.status_code, str(detail))
        return resp

    # ── Core Operations ────────────────────────────────────────────

    def remember(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
        kind: str = "fact",
        importance: float = 0.5,
    ) -> str:
        """Store a memory. Returns the memory UUID.

        Args:
            text: The memory content. Be specific for better recall.
            tags: Optional tags for filtering and auto-linking.
            kind: Memory kind — 'decision', 'fact', 'event', or 'preference'.
            importance: Weight 0.0-1.0 (higher = prioritized in snapshots).

        Returns:
            UUID string of the stored memory.
        """
        payload = {
            "text": text,
            "kind": kind,
            "importance": importance,
        }
        if tags:
            payload["tags"] = tags
        scope = self._scope_payload()
        if scope:
            payload["scope"] = scope

        resp = self._request("POST", "/api/remember", json=payload)
        data = resp.json()
        return data.get("id", "")

    def recall(
        self,
        query: str,
        *,
        budget: int = 2000,
        tags: Optional[list[str]] = None,
    ) -> RecallResult:
        """Recall memories matching a query within a token budget.

        Uses HDC fingerprint similarity + Personalized PageRank + decay
        scoring to find the most relevant memories.

        Args:
            query: Natural language query or keywords.
            budget: Maximum tokens to return (default: 2000).
            tags: Optional tag filter.

        Returns:
            RecallResult with ranked hits and token accounting.
        """
        payload = {"query": query, "budget": budget}
        if tags:
            payload["tags"] = tags
        scope = self._scope_payload()
        if scope:
            payload["scope"] = scope

        resp = self._request("POST", "/api/recall", json=payload)
        data = resp.json()

        hits = []
        for h in data.get("hits", []):
            node_data = h.get("node", h)
            node = Memory(**node_data)
            hits.append(
                RecallHit(
                    node=node,
                    final_score=h.get("final_score", 0.0),
                    fingerprint_sim=h.get("fingerprint_sim", 0.0),
                    ppr_score=h.get("ppr_score", 0.0),
                    decay_factor=h.get("decay_factor", 1.0),
                    from_hippocampus=h.get("from_hippocampus", False),
                )
            )

        return RecallResult(
            hits=hits,
            tokens_used=data.get("tokens_used", 0),
            tokens_budget=data.get("tokens_budget", budget),
            candidates_considered=data.get("candidates_considered", 0),
        )

    def forget(self, memory_id: str) -> None:
        """Soft-delete a memory by UUID.

        The memory is superseded by a tombstone, preserving the audit trail.
        """
        self._request("POST", "/api/forget", json={"id": memory_id})

    def supersede(self, old_id: str, new_text: str, **kwargs) -> str:
        """Replace a memory with a new version.

        Creates a new memory and marks the old one as superseded.

        Args:
            old_id: UUID of the memory to replace.
            new_text: Text for the replacement memory.
            **kwargs: Additional args passed to remember().

        Returns:
            UUID of the new memory.
        """
        new_id = self.remember(new_text, **kwargs)
        self._request(
            "POST",
            "/api/supersede",
            json={"old_id": old_id, "new_id": new_id},
        )
        return new_id

    def link(
        self,
        from_id: str,
        to_id: str,
        edge: str = "relates_to",
    ) -> None:
        """Link two memories with a typed edge.

        Edge types: relates_to, contradicts, supports, derived_from,
                    supersedes, before, after, caused_by.
        """
        self._request(
            "POST",
            "/api/link",
            json={"from": from_id, "to": to_id, "edge": edge},
        )

    def consolidate(self) -> ConsolidationResult:
        """Force a consolidation pass (hippocampus → neocortex migration).

        During consolidation, recent memories are evaluated against the
        long-term store: novel ones are promoted, similar ones reinforce
        existing memories, and redundant ones are dropped.

        Returns:
            ConsolidationResult with promotion/reinforcement/drop counts.
        """
        resp = self._request("POST", "/api/consolidate")
        return ConsolidationResult(**resp.json())

    def stats(self) -> Stats:
        """Get memory store statistics."""
        resp = self._request("GET", "/api/stats")
        data = resp.json()
        return Stats(
            store=StoreStats(**data.get("store", {})),
            hippocampus_size=data.get("hippocampus_size", 0),
            hippocampus_capacity=data.get("hippocampus_capacity", 0),
            neocortex_size=data.get("neocortex_size", 0),
            neocortex_edges=data.get("neocortex_edges", 0),
        )

    def health(self) -> bool:
        """Health check — returns True if the server is alive."""
        try:
            resp = self._request("GET", "/api/health")
            return resp.status_code == 200
        except Exception:
            return False

    # ── Context Manager ────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()
