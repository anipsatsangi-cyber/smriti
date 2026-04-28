//! **codegraph-memory** — graph-structured agent memory engine.
//!
//! Applies CodeGraph's PPR + token-budget compression philosophy to agent memory:
//! facts, entities, and experiences stored as a typed graph, retrieved via
//! personalized PageRank rather than raw vector similarity.
//!
//! ## Core operations
//!
//! ```rust,ignore
//! use smriti::MemoryEngine;
//!
//! let mut engine = MemoryEngine::open("./memory.db")?;
//! engine.remember("JWT RS256 is used for auth", &["auth", "security"], 0.9)?;
//! let ctx = engine.recall("authentication", 2000)?;
//! println!("{}", ctx.format_context());
//! ```

pub mod graph;
pub mod index;
pub mod recall;
pub mod snapshot;
pub mod store;

pub use graph::{MemoryEdge, MemoryGraph, MemoryNode};
pub use recall::{recall, RecallResult};
pub use snapshot::MemorySnapshot;
pub use store::MemoryStore;

use anyhow::Result;
use std::path::Path;
use uuid::Uuid;

/// High-level facade over the graph, store, and index.
///
/// This is the primary entry point for the MCP tools.
pub struct MemoryEngine {
    pub graph: MemoryGraph,
    store: MemoryStore,
    index: index::MemoryIndex,
}

impl MemoryEngine {
    /// Open or create a memory engine backed by `db_path` (SQLite) and
    /// `index_dir` (Tantivy). Loads all persisted memories into the in-memory graph.
    pub fn open(db_path: &Path, index_dir: &Path) -> Result<Self> {
        let store = MemoryStore::open(db_path)?;
        let mut idx = index::MemoryIndex::open(index_dir)?;
        let mut graph = MemoryGraph::new();

        // Hydrate graph from SQLite
        let nodes = store.load_all_nodes()?;
        let edges = store.load_all_edges()?;

        for node in nodes {
            graph.add_node(node);
        }
        for (from, to, kind) in edges {
            graph.add_edge(from, to, kind);
        }

        // Rebuild index from graph (idempotent — existing docs are overwritten)
        for node in graph.all_nodes() {
            idx.add(&node.id, &node.text, &node.tags)?;
        }
        idx.commit()?;

        Ok(Self { graph, store, index: idx })
    }

    /// Open an in-memory engine (no persistence — useful for testing).
    pub fn open_ephemeral() -> Result<Self> {
        Ok(Self {
            graph: MemoryGraph::new(),
            store: MemoryStore::open_in_memory()?,
            index: index::MemoryIndex::open_in_ram()?,
        })
    }

    /// Store a new memory and return its id.
    pub fn remember(&mut self, text: &str, tags: &[&str], importance: f32) -> Result<Uuid> {
        let token_count = count_tokens(text);
        let node = MemoryNode {
            id: Uuid::new_v4(),
            text: text.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            importance: importance.clamp(0.0, 1.0),
            created_at: chrono::Utc::now().to_rfc3339(),
            token_count,
        };
        let id = node.id;
        self.store.insert_node(&node)?;
        self.index.add(&id, text, &node.tags)?;
        self.index.commit()?;
        self.graph.add_node(node);
        Ok(id)
    }

    /// Link two memories with a typed relationship.
    pub fn link(&mut self, from: Uuid, to: Uuid, kind: MemoryEdge) -> Result<()> {
        self.store.insert_edge(&from, &to, kind)?;
        self.graph.add_edge(from, to, kind);
        Ok(())
    }

    /// Remove a memory by id. Returns true if it existed.
    pub fn forget(&mut self, id: &Uuid) -> Result<bool> {
        let existed = self.store.delete_node(id)?;
        self.index.remove(id)?;
        self.index.commit()?;
        self.graph.remove_node(id);
        Ok(existed)
    }

    /// PPR-ranked recall within a token budget.
    pub fn recall(&self, query: &str, token_budget: usize) -> Result<RecallResult> {
        recall::recall(&self.graph, &self.index, query, token_budget, 20)
    }

    /// Surface contradictions related to `query`.
    pub fn think(&self, query: &str) -> Result<Vec<(MemoryNode, MemoryNode)>> {
        let candidate_ids = self.index.search(query, 10)?;
        let indices: Vec<_> = candidate_ids
            .iter()
            .filter_map(|id| self.graph.index_of(id))
            .collect();
        let pairs = self.graph.find_contradictions(&indices);
        Ok(pairs.into_iter().map(|(a, b)| (a.clone(), b.clone())).collect())
    }

    /// Compressed full memory snapshot within a token budget.
    pub fn snapshot(&self, token_budget: usize) -> Result<MemorySnapshot> {
        MemorySnapshot::generate(&self.graph, token_budget)
    }

    /// Statistics about the current memory state.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_memories: self.graph.node_count(),
            total_edges: self.graph.edge_count(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub total_edges: usize,
}

/// Lightweight token counter using simple word splitting (no ML dependency).
/// Approximation: ~1.3 tokens per word for English text.
fn count_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    ((words as f64) * 1.3).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_remember_recall() {
        let mut engine = MemoryEngine::open_ephemeral().unwrap();
        engine.remember("JWT RS256 is used for API authentication", &["auth"], 0.9).unwrap();
        engine.remember("PostgreSQL is the primary database", &["db"], 0.7).unwrap();

        let result = engine.recall("authentication", 2000).unwrap();
        assert!(!result.memories.is_empty());
        assert!(result.memories.iter().any(|m| m.text.contains("JWT")));
    }

    #[test]
    fn test_engine_forget() {
        let mut engine = MemoryEngine::open_ephemeral().unwrap();
        let id = engine.remember("Temporary fact", &[], 0.5).unwrap();
        assert!(engine.forget(&id).unwrap());
        assert!(!engine.forget(&id).unwrap()); // second forget returns false
    }

    #[test]
    fn test_engine_think_contradictions() {
        let mut engine = MemoryEngine::open_ephemeral().unwrap();
        let id1 = engine.remember("The service is stateless", &["arch"], 0.8).unwrap();
        let id2 = engine.remember("The service stores session data in memory", &["arch"], 0.8).unwrap();
        engine.link(id1, id2, MemoryEdge::Contradicts).unwrap();

        let contradictions = engine.think("stateless").unwrap();
        assert!(!contradictions.is_empty());
    }

    #[test]
    fn test_engine_snapshot() {
        let mut engine = MemoryEngine::open_ephemeral().unwrap();
        for i in 0..5 {
            engine.remember(&format!("Memory fact {}", i), &[], 0.5).unwrap();
        }
        let snap = engine.snapshot(500).unwrap();
        assert!(snap.total_memories == 5);
        assert!(!snap.content.is_empty());
    }
}
