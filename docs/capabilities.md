# Smriti Capabilities — Reference Guide

A complete map of what Smriti does, with a Rust and an MCP/JSON example for every capability. This is the document that proves the moat: every capability listed here works **without an LLM in the loop**. The MCP examples are an ergonomic surface for agents that already have an LLM; the Rust examples are the load-bearing primitives.

> **Smriti is a memory engine, not a memory model.** The LLM is your agent's reasoning layer, not Smriti's. Replace your LLM, the engine is unchanged. Replace Smriti, your agent regresses to nearest-neighbor lookup.

---

## Table of contents

1. [Dual-store CLS recall](#1-dual-store-cls-recall) — the canonical operation
2. [Confidence verdicts](#2-confidence-verdicts) — the engine knows when it doesn't know
3. [Tiered confident truncation](#3-tiered-confident-truncation) — token economy at production scale
4. [Generic typed attribute filters](#4-generic-typed-attribute-filters) — schema-less, type-safe, composable
5. [Salience network — Routine / Important / Critical](#5-salience-network) — decay bypass
6. [Spreading activation (semantic priming)](#6-spreading-activation-semantic-priming) — cross-query continuity
7. [Goal-driven persistent priming](#7-goal-driven-persistent-priming) — motivational cortex
8. [Predictive coding (surprise mechanics)](#8-predictive-coding-surprise-mechanics) — Friston-style anomaly auto-flagging
9. [Causal trajectory replay](#9-causal-trajectory-replay) — episodic narrative reconstruction
10. [Supersede chains](#10-supersede-chains) — graceful contradiction with audit trail
11. [Reconsolidation](#11-reconsolidation) — usage-driven plasticity
12. [Sleep summarization (`suggest_clusters` + `merge`)](#12-sleep-summarization) — cognitive compression
13. [Vacuum + tombstone GC](#13-vacuum) — keep the graph dense
14. [Multi-agent scopes + federated HDC gossip](#14-scopes-and-federated-gossip)
15. [P2P sync (Last-Write-Wins)](#15-p2p-sync) — multi-device memory fabric
16. [Token-budgeted packing + MMR diversity](#16-token-budgeted-packing) — no echo-chamber returns
17. [SMRP/1.0 wire protocol + MCP](#17-smrp-and-mcp) — protocol-first integration
18. [What Smriti deliberately does NOT do](#what-smriti-deliberately-does-not-do)

---

## 1. Dual-store CLS recall

**What it is.** Smriti splits memory into a fast episodic *Hippocampus* and a slow semantic *Neocortex*, with a consolidation pass between them. Recall queries both stores together, fuses signals via Reciprocal Rank Fusion (Cormack et al. 2009), and packs the result into a token budget.

**Why it matters.** A single-store memory either keeps everything (no decay, ever-growing context) or evicts on a fixed window (loses just-learned facts). CLS gives you both: recent memories live in the hippocampus and stay accessible immediately, then consolidate into the neocortex where they're filtered for redundancy and linked into the semantic graph.

**Rust:**

```rust
use smriti::{Smriti, MemoryKind};

let mut s = Smriti::open(":memory:")?;
s.remember("Auth uses JWT RS256")
    .kind(MemoryKind::Fact)
    .tag("auth").tag("security")
    .commit()?;
s.consolidate()?;

let r = s.recall("how does authentication work")
    .budget(500)
    .execute()?;
println!("{}", r.render_text());
```

**MCP:**

```json
{ "name": "smriti_recall",
  "arguments": { "query": "how does authentication work", "budget": 500, "tags": ["auth"] } }
```

**Where to dig deeper:** `core::recall::recall_with_dense` is the full pipeline; `core::hippocampus` and `core::neocortex` are the two stores; `core::consolidation` is the redundancy filter that sits between them.

---

## 2. Confidence verdicts

**What it is.** Every `recall()` returns a `RecallVerdict`:

| Verdict             | Meaning                                                                  |
| ------------------- | ------------------------------------------------------------------------ |
| `Confident`         | Top hit clears the absolute floor, has a clear margin, and lexical/dense support. |
| `AmbiguousLeader`   | Top score is high enough but the runner-up is too close — there's a tie. |
| `UnsupportedTop`    | Top hit has no term overlap and weak dense similarity. Possible hallucination. |
| `LowConfidence`     | Top score is below the absolute floor. Engine isn't confident.           |
| `Abstained`         | Empty hit set. The engine refused to answer.                             |

**Why it matters.** Vector stores return their nearest neighbor regardless of how far away it is. Smriti can refuse to answer. **On bench-500 with 12 adversarial queries (gold not in corpus), Smriti correctly abstains 91.7% of the time.** No retrieval product I'm aware of measures this.

**Rust:**

```rust
use smriti::RecallVerdict;

let r = s.recall("which database").budget(500).execute()?;
match r.verdict {
    RecallVerdict::Confident       => println!("✓ {}", r.hits[0].node.text),
    RecallVerdict::AmbiguousLeader => println!("? top-2 disagree, returning both"),
    RecallVerdict::UnsupportedTop  => println!("⚠ pure-graph match, no lexical evidence"),
    RecallVerdict::LowConfidence   => println!("⚠ best score below floor"),
    RecallVerdict::Abstained       => println!("✗ corpus has no answer"),
}
```

**MCP:** Verdict is included on every `smriti_recall` response under `verdict`. Agents can branch on it.

```json
// Response from smriti_recall
{ "hits": [...], "verdict": "Confident", "top_score": 0.092, "top_margin": 0.018 }
```

---

## 3. Tiered confident truncation

**What it is.** When the engine is *very* sure (high score AND high margin AND strong lexical support), it returns just the single top hit. When ordinarily sure, it returns the top 2 as a hedge. When `AmbiguousLeader`, it returns up to 4. When `LowConfidence`, it returns nothing.

**Why it matters.** Agents pay tokens per memory in their context window. A confident answer doesn't need a context buffet. **bench-500 average tokens dropped from 489 → 79 (6× cut) at 95.7% intrinsic hit rate.**

**Rust:**

```rust
let r = s.recall("what hashing algorithm")
    .budget(500)
    .confident_truncation(/* confident */ 2, /* ambiguous */ 2, /* low */ 0)
    .confident_solo(/* score floor */ 0.085, /* margin floor */ 0.015)
    .execute()?;
// r.hits.len() may be 1 (extremely sure), 2 (ordinary sure), 4 (ambiguous), or 0 (low confidence).
```

**MCP:** All four knobs are exposed as optional integers in `smriti_recall` arguments.

```json
{ "name": "smriti_recall",
  "arguments": {
    "query": "what hashing algorithm",
    "budget": 500,
    "truncate_when_confident": 2,
    "truncate_when_ambiguous": 2,
    "confident_solo_score_floor": 0.085,
    "confident_solo_margin_floor": 0.015
  } }
```

---

## 4. Generic typed attribute filters

**What it is.** Memories carry an optional `attributes: HashMap<String, AttributeValue>`. `AttributeValue` is a typed enum (`Boolean`, `Number`, `Text`, `List`). Filter primitives:

| Filter                   | Semantics                                                                |
| ------------------------ | ------------------------------------------------------------------------ |
| `Eq(v)`                  | Exact equality. Epsilon-aware on `Number` (1e-9). Cross-type → `TypeMismatch`. |
| `Gt(Number)` / `Lt(Number)` | Strict numeric comparison. Non-numeric → `TypeMismatch`.              |
| `Range(Number, Number)`  | Inclusive numeric range `[min, max]`.                                   |
| `Contains(v)`            | List membership. **Not text substring.** For text use `Substring`.      |
| `Substring(String)`      | Substring search inside `Text`. Case-sensitive.                          |
| `All(Vec<AttrFilter>)`   | AND composition. Empty `All` is vacuous truth.                           |
| `Any(Vec<AttrFilter>)`   | OR composition. Empty `Any` is `NoMatch`.                                |

The match function returns `MatchResult { Match, NoMatch, TypeMismatch }`. `TypeMismatch` (e.g. caller stores `"50"` as `Text` but filter is `Gt(Number(50))`) excludes the candidate **and** logs one warning per query so the bug is visible.

**Why it matters.** This is the surface that makes spatial / quantitative / temporal / relational queries possible **without baking those dimensions into the engine schema**. Tomorrow's dimension (sentiment, provenance, intent) is just data.

**Rust:**

```rust
use smriti::{AttrFilter, AttributeValue};

s.remember("Bought camera lens at $250, downtown shop, March 2024")
    .attr("price", AttributeValue::Number(250.0))
    .attr("location", AttributeValue::Text("downtown".into()))
    .attr("tags", AttributeValue::List(vec![
        AttributeValue::Text("photography".into()),
        AttributeValue::Text("equipment".into()),
    ]))
    .commit()?;

// "Show me purchases between $100 and $300, downtown OR airport"
let r = s.recall("recent purchases")
    .where_attr("price", AttrFilter::Range(
        AttributeValue::Number(100.0), AttributeValue::Number(300.0)))
    .where_attr("location", AttrFilter::Any(vec![
        AttrFilter::Eq(AttributeValue::Text("downtown".into())),
        AttrFilter::Eq(AttributeValue::Text("airport".into())),
    ]))
    .execute()?;
```

**MCP:**

```json
{ "name": "smriti_recall",
  "arguments": {
    "query": "recent purchases",
    "attr_filters": {
      "price":    { "Range": [{"Number": 100.0}, {"Number": 300.0}] },
      "location": { "Any":   [
                      { "Eq": {"Text": "downtown"} },
                      { "Eq": {"Text": "airport"} }
                  ]}
    } } }
```

---

## 5. Salience network

**What it is.** Each memory has a `salience: Salience` field with three values:

- `Routine` (default) — standard kind-based decay.
- `Important` — half-life is `kind.half_life_days() * 3.0`. Slower forgetting.
- `Critical` — bypasses decay entirely; `effective_importance` returns `importance + 1.0`. Auto-injected as a PPR seed in **every** recall (the *amygdala fast-track*).

**Why it matters.** Some memories must never decay regardless of access frequency. Life-safety guardrails, hard-set user preferences, security-critical facts. Vector stores have no equivalent — they decay (or not) uniformly across all embeddings.

**Rust:**

```rust
use smriti::node::Salience;

s.remember("User has a peanut allergy. Never recommend peanut-containing dishes.")
    .kind(MemoryKind::Decision)
    .salience(Salience::Critical)
    .commit()?;
// Forever, every recall — no matter the topic — auto-seeds PPR with this node.
```

**MCP:**

```json
{ "name": "smriti_remember",
  "arguments": {
    "text": "User has a peanut allergy. Never recommend peanut-containing dishes.",
    "kind": "decision",
    "salience": "critical"
  } }
```

---

## 6. Spreading activation (semantic priming)

**What it is.** Each PPR run leaves residual activation on the nodes it visited. The next recall reads that residual into its seed vector, so a follow-up query inherits the structural priming of the previous one. Decay is two-component: a per-call pulse (0.5×) drains residual on tight loops, and a 30-second wall-clock half-life adds additional drain on human-scale pauses.

**Why it matters.** Real conversations are *correlated*. The user asks about deploys, then asks about logs, then asks about CPU. Priming makes each follow-up cheaper and more accurate by carrying the relevant subgraph forward.

**Honest scope.** On a 5-node-per-cluster synthetic corpus the residual lift is ~0 (PPR converges over the small subgraph regardless of seed weight). The mechanism is more visible on denser graphs and is the substrate for goal-pinning (next section), which *does* demonstrably steer recall.

**Rust:**

```rust
// Q1 primes the database subgraph.
let _ = s.recall("which database do we use").budget(500).execute()?;
// Q2 inherits the residual priming.
let _ = s.recall("how is connection pooling configured").budget(500).execute()?;
// Topic switch — wipe activation.
s.clear_activation();
let _ = s.recall("what is our marketing strategy").budget(500).execute()?;
```

**MCP:**

```json
// Topic switch
{ "name": "smriti_clear_activation", "arguments": {} }
```

---

## 7. Goal-driven persistent priming

**What it is.** A memory tagged with `MemoryKind::Goal` has its activation pinned at 1.0 and never decays until the goal is explicitly superseded (i.e. completed). It is auto-injected as a PPR seed in every recall regardless of the explicit query seeds. This implements the *motivational cortex* pattern from cognitive architecture (SOAR, ACT-R).

**Why it matters.** When an agent has a long-running task, ambiguous queries should be interpreted in light of that task. "What's the main problem?" means something different when the agent is debugging Postgres versus when it's writing CSS. Goal-pinning makes that bias explicit, persistent, and easy to reset.

**Rust:**

```rust
let goal = s.remember("Primary objective: troubleshoot and optimize PostgreSQL performance")
    .kind(MemoryKind::Goal)
    .tag("database").tag("optimization")
    .commit()?;

s.remember("CPU spikes during write-heavy periods").kind(MemoryKind::Event).commit()?;
s.remember("Bloat is hitting user_activity_logs").kind(MemoryKind::Event).commit()?;
s.consolidate()?;

// Ambiguous query — but the goal pins recall to the database subgraph.
let r = s.recall("what's the main problem").budget(150).execute()?;
println!("{}", r.hits[0].node.text); // → "Bloat is hitting user_activity_logs"

// When the goal is complete:
s.supersede(goal, /* completion memory id */ done_id)?;
```

This is the integration test in `core/tests/agi_integration_test.rs`. It passes on every build.

**MCP:**

```json
{ "name": "smriti_remember",
  "arguments": {
    "text": "Primary objective: troubleshoot PostgreSQL performance",
    "kind": "goal",
    "tags": ["database", "optimization"]
  } }
```

---

## 8. Predictive coding (surprise mechanics)

**What it is.** During consolidation, Smriti computes the HDC fingerprint similarity between the candidate memory and its nearest existing neighbor. If `max_similarity < SURPRISE_THRESHOLD` (0.05) AND the corpus has at least `SURPRISE_MIN_CORPUS` (30) nodes, the new memory is auto-promoted to `Salience::Critical` with `importance = 1.0`. This is Karl Friston's free-energy principle made operational: the agent learns hardest from prediction errors.

**Why it matters.** Most retrieval systems treat all incoming memories equally. Smriti notices when a new fact is structurally orthogonal to what it already knows — exactly the moments an agent should pay close attention to.

**Calibration story.** This was originally `SURPRISE_THRESHOLD = 0.2`, which was too loose for diverse 500-memory corpora and caused a real bench regression by auto-promoting most consolidated memories to Critical (polluting the PPR seed pool). The fix tightened to 0.05 and added the minimum-corpus guard. The mechanism is intact; the calibration is honest about needing context to detect anomalies.

**Rust:** Surprise fires automatically. Inspect after consolidation:

```rust
s.remember("normal database log entry").commit()?;
// ...many normal entries...
s.remember("retry storm: 150,000 rows/sec from a frontend bug").commit()?;
s.consolidate()?;

let nodes = s.export_sync_state()?.0;
for n in &nodes {
    if n.salience == smriti::node::Salience::Critical {
        println!("auto-flagged anomaly: {}", n.text);
    }
}
```

**MCP:** No tool required — the surprise check runs as part of `smriti_consolidate`. Agents can read the salience of any memory by inspecting recall results.

---

## 9. Causal trajectory replay

**What it is.** Given a starting memory id, Smriti traverses `CausedBy` / `Before` / `After` / `DerivedFrom` edges in BFS order to return an ordered narrative of events. This is *episodic replay* — DeepMind's hippocampal replay primitive made first-class.

**Why it matters.** Vector stores cannot reconstruct narratives. They can return memories that look similar to a query, but they have no edge structure to walk. Smriti's typed edges + bounded BFS gives agents a deterministic primitive for "show me the chain of events leading to this."

**Rust:**

```rust
use smriti::MemoryEdge;

let bug = s.remember("retry storm pushed DB to 150k rows/sec").commit()?;
let cpu = s.remember("Postgres primary CPU saturated").commit()?;
let outage = s.remember("API errors spiked, users locked out").commit()?;

s.link(bug, cpu, MemoryEdge::CausedBy)?;
s.link(cpu, outage, MemoryEdge::CausedBy)?;

for (i, n) in s.recall_trajectory(bug, 5)?.iter().enumerate() {
    println!("step {}: {}", i + 1, n.text);
}
// step 1: retry storm pushed DB to 150k rows/sec
// step 2: Postgres primary CPU saturated
// step 3: API errors spiked, users locked out
```

**MCP:**

```json
{ "name": "smriti_recall_trajectory",
  "arguments": { "start_id": "uuid-of-the-bug", "limit": 5 } }
```

---

## 10. Supersede chains

**What it is.** When a fact gets corrected, the agent calls `supersede(old_id, new_id)`. The old memory is hidden from recall but stays on disk. The new memory carries a `supersedes: Some(old_id)` reference. Audit trail preserved.

**Why it matters.** A user moves from Seattle to Austin. Vector stores either delete (history lost) or keep both (LLM gets confused). Smriti hides the old, surfaces the new, and the audit chain is reconstructable forever.

**Validated.** On LongMemEval-S `knowledge-update` category, Smriti scores 62.5% (substring) — the canonical multi-session knowledge-update benchmark working at scale.

**Rust:**

```rust
let old = s.remember("User lives in Seattle, WA")
    .kind(MemoryKind::Fact).tag("user").commit()?;

let new = s.remember("User lives in Austin, TX (moved March 2025)")
    .kind(MemoryKind::Fact).tag("user")
    .supersedes(old)
    .commit()?;

// Future recalls return only the Austin fact.
let r = s.recall("where does the user live").execute()?;
assert!(r.hits[0].node.text.contains("Austin"));
```

**MCP:**

```json
{ "name": "smriti_supersede",
  "arguments": {
    "old_id": "uuid-seattle",
    "new_text": "User lives in Austin, TX (moved March 2025)",
    "tags": ["user"]
  } }
```

---

## 11. Reconsolidation

**What it is.** Retrieved memories can have new tags appended in light of how they were just used. Smriti recomputes the HDC fingerprint after the tag change so the graph's structural index stays consistent. This implements the neuroscience phenomenon (Nader 2003) where retrieval makes a memory chemically labile and modifiable.

**Why it matters.** Memory becomes adaptive. When a fact is useful for a query type, tag it with that query type and it surfaces faster next time. The engine *learns from retrieval*.

**Rust:**

```rust
// Earlier: stored a fact about JWT.
let fact_id = s.remember("Auth uses JWT RS256 with 1-hour expiry").commit()?;

// Agent retrieves it for a security-audit question. Tag accordingly.
s.reconsolidate(fact_id, vec!["security-audit-2025".into()])?;
// Next time the agent asks about security audits, this fact surfaces faster.
```

**MCP:**

```json
{ "name": "smriti_reconsolidate",
  "arguments": {
    "id": "uuid-of-the-jwt-fact",
    "new_tags": ["security-audit-2025"]
  } }
```

---

## 12. Sleep summarization

**What it is.** `suggest_clusters(limit)` finds dense subgraphs (≥3 mutually-related memories joined by `RelatesTo` / `Supports` / `DerivedFrom` / `CausedBy` edges) and ranks them by `redundancy_score = nodes × internal_edge_count`. The agent uses an LLM to summarize each cluster, then `merge(old_ids, summary)` replaces the cluster with a single summary node.

**Why it matters.** Long-running agents accumulate hairballs of redundant memories. Manual cleanup is unscalable. Smriti finds the candidates; the agent (or its LLM) writes the summary; the merge is atomic.

**Rust:**

```rust
let clusters = s.suggest_clusters(3);
for c in &clusters {
    println!("cluster of {} nodes, {} internal edges, score {}",
        c.nodes.len(), c.internal_edge_count, c.redundancy_score);
    for n in &c.nodes { println!("  {}", n.text); }
}
// Agent writes the summary externally (LLM call), then atomically replaces
// the cluster with a single summary node by superseding each member:
let summary = s.remember("Summary text covering all three facts").commit()?;
for old_id in [id1, id2, id3] {
    s.supersede(old_id, summary)?;
}
```

(`smriti_merge` over MCP wraps this pattern in one tool call.)

**MCP:**

```json
// Step 1
{ "name": "smriti_suggest_clusters", "arguments": { "limit": 3 } }
// Step 2 (after LLM writes summary)
{ "name": "smriti_merge",
  "arguments": {
    "old_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "new_text": "Three-fact summary"
  } }
```

---

## 13. Vacuum

**What it is.** `vacuum()` rebuilds the active neocortex graph, copying only active (non-superseded) nodes and valid edges into a new `petgraph::DiGraph`. Activation state is remapped through the new node indices. Auto-runs as part of `consolidate()`; can be invoked explicitly.

**Why it matters.** Without vacuum, superseded nodes remain as tombstones in the graph, slowing PPR over time. With vacuum, the active graph stays dense and traversal stays fast.

**Engineering note.** Currently runs unconditionally inside `consolidate()`. There's a TODO at `smriti.rs` to gate this with `if tombstone_count() > THRESHOLD` — under heavy supersede traffic the unconditional rebuild can spike p95 latency.

**Rust:**

```rust
s.vacuum();  // typically you let consolidate() call this
```

**MCP:**

```json
{ "name": "smriti_vacuum", "arguments": {} }
```

---

## 14. Scopes and federated gossip

**What it is.** Every memory has a `scope: Scope { agent, user, session, shared_with }`. Recall enforces scope at every step — a query in scope `(agent="B", user="bob")` cannot retrieve memories scoped to `(agent="A", user="alice")` unless `alice`'s memory has explicit `shared_with` permitting it.

The federated *HDC gossip* pattern: an agent in scope A can broadcast a 2048-bit HDC fingerprint of its query to an agent in scope B's Smriti instance. B does an XOR similarity check against its own neocortex. If a match exists, B replies "I have high-relevance memories about this — ask the user for explicit consent before sharing." **No raw text is ever leaked during the search phase**, because HDC fingerprints are one-way hashes.

**Why it matters.** Multi-agent swarms need to know if they're working on the same problem without exposing each other's private memory. This is the FIDO of agent collaboration.

**Rust:**

```rust
use smriti::Scope;

let alice_scope = Scope::agent("agent-a").with_user("alice");
s.remember("Alice's medical records: peanut allergy, blood type O-")
    .scope(alice_scope)
    .commit()?;

// Agent B in a different scope cannot retrieve Alice's data.
let bob_scope = Scope::agent("agent-b").with_user("bob");
let r = s.recall("medical history").scope(bob_scope).execute()?;
assert!(r.hits.iter().all(|h| !h.node.text.contains("Alice")));
```

**MCP:**

```json
{ "name": "smriti_remember",
  "arguments": {
    "text": "Alice's medical records",
    "scope": { "agent": "agent-a", "user": "alice" }
  } }
```

---

## 15. P2P sync

**What it is.** `export_sync_state()` returns `(Vec<MemoryNode>, Vec<(Uuid, Uuid, MemoryEdge)>)`. `import_sync_state(nodes, edges)` applies them with **Last-Write-Wins** conflict resolution on `(version, last_accessed_at)`. Nodes are processed in chronological order so multi-update chains converge correctly.

**Why it matters.** Hosted memory products own your memory. Smriti is the foundation for **multi-device, multi-agent memory federation**: an agent on a laptop can sync to one in a browser tab, no central server required.

**Rust:**

```rust
let mut s1 = Smriti::open("device-a.db")?;
let mut s2 = Smriti::open("device-b.db")?;

s1.remember("a fact learned on device A").commit()?;
s1.consolidate()?;

let (nodes, edges) = s1.export_sync_state()?;
s2.import_sync_state(nodes, edges)?;
// device B now has the fact, with its full version history and edges.
```

**MCP:** Sync isn't exposed via MCP today — it's an engine-to-engine primitive. Agents shouldn't be calling sync directly.

---

## 16. Token-budgeted packing

**What it is.** Every recall takes a `budget` (in tokens). The MMR-diversified candidate list is greedily packed into the budget via knapsack — but biased by Maximal Marginal Relevance (Carbonell & Goldstein 1998) so the pack diversifies on tag-Jaccard. No five copies of the same fact in different wording.

**Why it matters.** Agents pay tokens. A naïve top-K returns redundancy. MMR returns coverage.

**Validated.** On bench-500, the duplicate hit rate is 0.0% across both zero-ML and embeddings modes.

**Rust:**

```rust
let r = s.recall("auth").budget(500).lambda(0.7).execute()?;
// lambda=1.0 → pure relevance (no diversity), lambda=0.0 → pure diversity.
```

**MCP:**

```json
{ "name": "smriti_recall",
  "arguments": { "query": "auth", "budget": 500, "mmr_lambda": 0.7 } }
```

---

## 17. SMRP and MCP

**What it is.** Two integration surfaces over the same Rust primitives:

- **SMRP/1.0** (Smriti Memory Request Protocol) — versioned, line-delimited JSON-RPC for direct integration. Debuggable with `nc localhost 4000`.
- **MCP** (Model Context Protocol) — adapter for Claude Code, Cursor, Zed. Exposes 13 tools including everything documented above.

Both are translation layers; neither does cognitive work. The engine's primitives are what they speak.

**HTTP server:**

```bash
cargo run --release --features http --bin smriti-http -- --db ~/.smriti/global.db --port 4000
```

**Endpoints:**

- `POST /api/remember` — stores a memory
- `POST /api/recall` — token-budgeted recall
- `POST /api/supersede` / `/api/forget` / `/api/link` / `/api/consolidate`
- `GET /api/stats`
- `POST /smrp` — SMRP/1.0 line-delimited

---

## What Smriti deliberately does NOT do

The boundary that keeps Smriti from becoming "just another LLM-backed memory product."

### 1. Natural-language temporal phrases ("last Tuesday")

**The boundary.** The engine is deterministic and WASM-buildable. It refuses to embed an LLM to parse dates, locations, or other natural-language structures.

**Where it lives instead.** The agent (or your application's LLM router) is responsible for translating "last Tuesday" into a concrete `AttrFilter::Range` over Unix timestamps. Once the filter is structured, Smriti executes deterministically.

```rust
// Bad: agent passes string → Smriti can't parse it
// (Smriti will not implement this. It would couple the engine to an LLM.)

// Good: agent computes the timestamp range, passes typed filter
let one_week_ago = chrono::Utc::now() - chrono::Duration::days(7);
let r = s.recall("what did I deploy")
    .where_attr("timestamp", AttrFilter::Gt(AttributeValue::Number(one_week_ago.timestamp() as f64)))
    .execute()?;
```

### 2. Endogenous summarization

**The boundary.** Smriti finds redundant clusters (`suggest_clusters`) and atomically replaces them (`merge`), but writing the summary text is a generative-LM task.

**Where it lives instead.** The agent reads the cluster, asks its LLM to summarize, then calls `merge(old_ids, summary_text)`.

### 3. Free-form numeric aggregation

**The boundary.** Smriti is a cognitive memory engine, not a relational analytics engine. It will not compute "average price of flights" from your memories.

**Where it lives instead.** The agent recalls relevant memories with attribute filters, reads them into its own context, and computes the aggregate. Smriti's job is to surface the right memories; the agent's job is to do math on them.

### 4. Automatic semantic conflict detection

**The boundary.** If you store "User loves Rust" and then "User hates Rust," Smriti returns both. It does not parse English to detect contradiction.

**Where it lives instead.** Before storing a new preference, the agent recalls existing preferences. If a conflict is detected (by the agent's LLM), the agent calls `supersede` or `link(MemoryEdge::Contradicts)` to encode the relationship explicitly.

### 5. LLM-driven query rewriting

**The boundary.** Smriti will not transparently rewrite "tell me about my notes" into a more specific query.

**Where it lives instead.** This is exactly what an LLM router *should* do above Smriti — and it's a one-call ergonomic layer, not a load-bearing primitive. The router translates the user's natural-language sentence into a concrete `RecallBuilder` call (query string, scope, tags, attribute filters, budget). Smriti executes deterministically below it.

---

## Honest scope, summarized

| Capability                       | Verified working                                                                  |
| -------------------------------- | --------------------------------------------------------------------------------- |
| Dual-store CLS recall            | bench-500 zero-ML 95.7% intrinsic hit                                             |
| Confidence verdicts + abstention | bench-500: 91.7% correct abstention on 12 adversarial queries                     |
| Tiered confident truncation      | bench-500: 489 → 79 tokens (6× cut), no quality loss                              |
| Generic typed attribute filters  | 16 unit tests; type-mismatch diagnostics; 5 filter primitives + AND/OR composition |
| Salience::Critical decay bypass  | Decay returns `importance + 1.0`; auto-PPR-seed; `agi_integration_test`           |
| Spreading activation             | Mechanism intact; quantitative lift requires denser graphs (continuity bench null on 5-node clusters) |
| Goal-driven priming              | `agi_integration_test`: ambiguous query → goal-relevant fact returned             |
| Predictive coding                | Auto-Critical promotion at `similarity < 0.05` and `corpus ≥ 30`                  |
| Causal trajectory replay         | `agi_integration_test`: reconstructed bug → CPU-spike chain                       |
| Supersede chains                 | LongMemEval `knowledge-update` 62.5%                                              |
| Reconsolidation                  | Tag append + HDC fingerprint update unit-tested                                   |
| Sleep summarization              | `suggest_clusters` returns ranked clusters; `merge` is atomic                     |
| Vacuum                           | Active graph rebuild after consolidation; no orphan tombstones                    |
| Scopes + federated gossip        | Scope isolation enforced at every recall step                                     |
| P2P sync (LWW)                   | `sync_state_roundtrip_lww` test                                                   |
| Token-budget MMR                 | bench-500: 0.0% duplicate hit rate                                                |

**83 unit tests + 1 integration test, all green.** WASM build at 127 KB gzipped. p95 recall at 1.6 ms on 500 memories.

For the live numbers and the cold-vs-primed continuity benchmark, see [`benchmarks/results/REAL_DATASETS_REPORT.md`](../benchmarks/results/REAL_DATASETS_REPORT.md) and run `cargo run --release --bin smriti-bench-500`.
