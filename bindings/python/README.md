# smriti 🧠

> **The SQLite of AI memory — now in Python.**

Python SDK for [Smriti](https://github.com/fork-demon/codegraph) (स्मृति), the neuroscience-inspired memory engine for AI agents. Zero external dependencies, sub-millisecond recall, grounded in Complementary Learning Systems theory.

## Install

```bash
pip install smriti
```

With LangChain support:
```bash
pip install smriti[langchain]
```

With CrewAI support:
```bash
pip install smriti[crewai]
```

## Quick Start

```python
from smriti import SmritiClient

# Connect to Smriti server (start with: codegraph smriti serve)
client = SmritiClient("http://localhost:4000")

# Store memories
client.remember(
    "The auth module uses JWT RS256 with 1-hour expiry",
    tags=["auth", "security"],
    kind="decision",
    importance=0.9,
)

client.remember(
    "Database is Postgres 15 on RDS",
    tags=["infra", "database"],
    kind="fact",
)

# Recall with PPR-ranked graph traversal
result = client.recall("authentication", budget=2000)
for hit in result.hits:
    print(f"[{hit.final_score:.2f}] {hit.node.text}")

# Link memories with temporal edges
client.link(deploy_id, incident_id, edge="caused_by")
client.link(deploy_id, incident_id, edge="before")

# Stats
stats = client.stats()
print(f"Memories: {stats.store.active_memories}")
print(f"Neocortex: {stats.neocortex_size} nodes")
```

## LangChain Integration

Drop-in replacement for LangChain's memory:

```python
from smriti.langchain import SmritiMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = SmritiMemory(base_url="http://localhost:4000")
chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory,
)

# Memories persist across sessions automatically
response = chain.predict(input="What auth method does our API use?")
```

## CrewAI Integration

```python
from smriti.crewai import SmritiCrewMemory

memory = SmritiCrewMemory(base_url="http://localhost:4000")

# Store agent observations
memory.save("Found SQL injection in /api/users", metadata={"kind": "event"})

# Search across all agent memories
results = memory.search("security vulnerabilities", limit=5)
```

## Edge Types

| Edge | Description | Use Case |
|------|-------------|----------|
| `relates_to` | General association | Auto-created by shared tags |
| `contradicts` | Conflicting facts | "We use Postgres" vs "We use MySQL" |
| `supports` | Evidence | Fact backing a decision |
| `derived_from` | Causal derivation | Summary from raw data |
| `supersedes` | Replacement | Updated fact |
| `before` | Temporal ordering | Event sequencing |
| `after` | Temporal ordering | Event sequencing |
| `caused_by` | Causal link | Incident → root cause |

## How It Works

Smriti uses a dual-store architecture inspired by neuroscience:

1. **Hippocampus** — Fast episodic buffer for recent memories (HDC fingerprints)
2. **Neocortex** — Semantic knowledge graph for consolidated long-term memory (PPR)
3. **Consolidation** — "Sleep replay" that promotes valuable memories and drops redundant ones

Recall is **zero-ML**: no embedding API calls, no external vector DB. Similarity is computed via XOR/popcount on 2048-bit hypervectors in ~10ns.

## Requirements

- Python ≥ 3.9
- Running Smriti server: `codegraph smriti serve`
- Or via Docker: `docker run -p 4000:4000 ghcr.io/fork-demon/smriti`

## License

MIT
