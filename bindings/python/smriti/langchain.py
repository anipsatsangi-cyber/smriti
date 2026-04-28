"""LangChain BaseMemory adapter for Smriti.

Drop-in replacement for LangChain's built-in memory classes. Persists
conversation context to the Smriti engine and retrieves it using
PPR-ranked graph traversal.

Usage with LangChain:
    >>> from smriti.langchain import SmritiMemory
    >>> memory = SmritiMemory(base_url="http://localhost:4000")
    >>> # Use with any LangChain chain
    >>> chain = ConversationChain(memory=memory, llm=llm)

Usage with LangGraph:
    >>> memory = SmritiMemory(base_url="http://localhost:4000")
    >>> # Manual load/save in your graph nodes
    >>> context = memory.load_memory_variables({"input": user_msg})
    >>> memory.save_context({"input": user_msg}, {"output": ai_msg})
"""

from __future__ import annotations

from typing import Any, Optional

from smriti.client import SmritiClient

try:
    from langchain_core.memory import BaseMemory
except ImportError:
    raise ImportError(
        "LangChain is required for this adapter. "
        "Install it with: pip install smriti[langchain]"
    )


class SmritiMemory(BaseMemory):
    """LangChain-compatible memory backed by the Smriti engine.

    Automatically stores conversation turns as memories and retrieves
    relevant context on each new turn using Smriti's HDC + PPR recall.

    Attributes:
        base_url: Smriti server URL (default: http://localhost:4000).
        memory_key: Key in the memory variables dict (default: "smriti_context").
        human_prefix: Prefix for human messages (default: "Human").
        ai_prefix: Prefix for AI messages (default: "AI").
        budget: Token budget for recall (default: 2000).
        auto_tags: Tags automatically added to all memories.
    """

    base_url: str = "http://localhost:4000"
    memory_key: str = "smriti_context"
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    budget: int = 2000
    auto_tags: list[str] = []
    _client: Optional[SmritiClient] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def client(self) -> SmritiClient:
        if self._client is None:
            self._client = SmritiClient(base_url=self.base_url)
        return self._client

    @property
    def memory_variables(self) -> list[str]:
        """Keys this memory injects into the chain."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load relevant memories for the current input.

        Extracts the user's input and uses it as a recall query.
        Returns formatted memory context within the token budget.
        """
        # Find the user input — check common keys
        user_input = ""
        for key in ("input", "question", "query", "human_input"):
            if key in inputs:
                user_input = str(inputs[key])
                break

        if not user_input:
            return {self.memory_key: ""}

        try:
            result = self.client.recall(user_input, budget=self.budget)
            if not result.hits:
                return {self.memory_key: ""}

            lines = ["Relevant memories from previous sessions:"]
            for hit in result.hits:
                tags_str = f" [{', '.join(hit.node.tags)}]" if hit.node.tags else ""
                lines.append(f"- {hit.node.text}{tags_str}")

            return {self.memory_key: "\n".join(lines)}

        except Exception:
            # Fail silently — memory is enhancement, not critical path
            return {self.memory_key: ""}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the current conversation turn to Smriti.

        Stores both the human input and AI output as separate memories
        with appropriate tags.
        """
        human_input = ""
        for key in ("input", "question", "query", "human_input"):
            if key in inputs:
                human_input = str(inputs[key])
                break

        ai_output = ""
        for key in ("output", "response", "answer", "text"):
            if key in outputs:
                ai_output = str(outputs[key])
                break

        try:
            tags = list(self.auto_tags)

            if human_input:
                self.client.remember(
                    f"{self.human_prefix}: {human_input}",
                    tags=tags + ["conversation", "human"],
                    kind="event",
                    importance=0.4,
                )

            if ai_output:
                self.client.remember(
                    f"{self.ai_prefix}: {ai_output}",
                    tags=tags + ["conversation", "ai"],
                    kind="event",
                    importance=0.3,
                )
        except Exception:
            # Fail silently
            pass

    def clear(self) -> None:
        """Clear is a no-op — Smriti uses decay instead of hard deletion."""
        pass
