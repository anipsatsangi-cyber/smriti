//! In-memory `Store` — pure Rust, zero system dependencies.
//!
//! Used by:
//!
//! 1. The WASM build, where `rusqlite` cannot compile.
//! 2. Unit tests that want a fast, isolated store.
//! 3. Anyone who explicitly wants ephemeral memory (e.g. a CLI that
//!    discards state between runs).
//!
//! Internally just a `RwLock<HashMap>` of memories and a `Vec` of edges.
//! The data lives in process memory only — there is no persistence here.
//! Power users who need round-tripping should use [`Smriti::serialize`]
//! / [`Smriti::restore`] (added in a later step) to dump and restore
//! the entire state as JSON.

use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::Result;
use uuid::Uuid;

use crate::node::{MemoryEdge, MemoryNode};
use crate::scope::Scope;
use crate::store::{Store, StoreStats};

/// Pure-Rust ephemeral store. Send + Sync, safe to wrap in `Arc<dyn Store>`.
pub struct InMemoryStore {
    nodes: RwLock<HashMap<Uuid, MemoryNode>>,
    edges: RwLock<Vec<(Uuid, Uuid, MemoryEdge)>>,
}

impl InMemoryStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
            edges: RwLock::new(Vec::new()),
        }
    }

    /// Number of memories in the store. Convenience for tests.
    pub fn len(&self) -> usize {
        self.nodes.read().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Store for InMemoryStore {
    fn upsert(&self, node: &MemoryNode) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        nodes.insert(node.id, node.clone());
        Ok(())
    }

    fn link(&self, from: Uuid, to: Uuid, edge: MemoryEdge) -> Result<()> {
        let mut edges = self
            .edges
            .write()
            .map_err(|e| anyhow::anyhow!("edges lock poisoned: {}", e))?;
        // Dedupe (from, to, kind) tuples — same as the SQLite PRIMARY KEY.
        if !edges
            .iter()
            .any(|(f, t, k)| *f == from && *t == to && *k == edge)
        {
            edges.push((from, to, edge));
        }
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<MemoryNode>> {
        let nodes = self
            .nodes
            .read()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        Ok(nodes.values().cloned().collect())
    }

    fn load_edges(&self) -> Result<Vec<(Uuid, Uuid, MemoryEdge)>> {
        let edges = self
            .edges
            .read()
            .map_err(|e| anyhow::anyhow!("edges lock poisoned: {}", e))?;
        Ok(edges.clone())
    }

    fn supersede(&self, old_id: Uuid, new_id: Uuid) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        if let Some(old) = nodes.get_mut(&old_id) {
            old.superseded_by = Some(new_id);
        }
        if let Some(new_node) = nodes.get_mut(&new_id) {
            new_node.supersedes = Some(old_id);
        }
        drop(nodes);
        // Also record the edge so downstream graph queries see it.
        self.link(new_id, old_id, MemoryEdge::Supersedes)?;
        Ok(())
    }

    fn delete(&self, id: Uuid) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        nodes.remove(&id);
        drop(nodes);

        let mut edges = self
            .edges
            .write()
            .map_err(|e| anyhow::anyhow!("edges lock poisoned: {}", e))?;
        edges.retain(|(f, t, _)| *f != id && *t != id);
        Ok(())
    }

    fn touch(&self, id: Uuid) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        if let Some(node) = nodes.get_mut(&id) {
            node.access_count = node.access_count.saturating_add(1);
            node.last_accessed_at = chrono::Utc::now();
        }
        Ok(())
    }

    fn stats(&self) -> Result<StoreStats> {
        let nodes = self
            .nodes
            .read()
            .map_err(|e| anyhow::anyhow!("nodes lock poisoned: {}", e))?;
        let edges = self
            .edges
            .read()
            .map_err(|e| anyhow::anyhow!("edges lock poisoned: {}", e))?;

        let total_memories = nodes.len();
        let superseded_memories = nodes.values().filter(|n| n.superseded_by.is_some()).count();
        let active_memories = total_memories - superseded_memories;
        let total_tokens: usize = nodes.values().map(|n| n.token_count).sum();

        Ok(StoreStats {
            total_memories,
            active_memories,
            superseded_memories,
            total_edges: edges.len(),
            total_tokens,
        })
    }

    fn load_for_scope(&self, _scope: &Scope) -> Result<Vec<MemoryNode>> {
        // Same shortcut SqliteStore takes — scope filtering happens
        // in-memory after load. The dataset is small enough.
        self.load_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;

    fn mk(text: &str) -> MemoryNode {
        MemoryNode::new(text, MemoryKind::Fact, Scope::default())
    }

    #[test]
    fn upsert_and_load() {
        let s = InMemoryStore::new();
        let n = mk("hello world");
        let id = n.id;
        s.upsert(&n).unwrap();
        let all = s.load_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);
    }

    #[test]
    fn link_dedupes() {
        let s = InMemoryStore::new();
        let a = mk("a");
        let b = mk("b");
        s.upsert(&a).unwrap();
        s.upsert(&b).unwrap();
        s.link(a.id, b.id, MemoryEdge::RelatesTo).unwrap();
        s.link(a.id, b.id, MemoryEdge::RelatesTo).unwrap();
        assert_eq!(s.load_edges().unwrap().len(), 1);

        // But different edge kinds are kept
        s.link(a.id, b.id, MemoryEdge::Supports).unwrap();
        assert_eq!(s.load_edges().unwrap().len(), 2);
    }

    #[test]
    fn supersede_marks_both_sides() {
        let s = InMemoryStore::new();
        let old = mk("old");
        let new = mk("new");
        s.upsert(&old).unwrap();
        s.upsert(&new).unwrap();
        s.supersede(old.id, new.id).unwrap();

        let nodes = s.load_all().unwrap();
        let old_loaded = nodes.iter().find(|n| n.id == old.id).unwrap();
        let new_loaded = nodes.iter().find(|n| n.id == new.id).unwrap();
        assert_eq!(old_loaded.superseded_by, Some(new.id));
        assert_eq!(new_loaded.supersedes, Some(old.id));

        // Supersedes edge auto-recorded
        let edges = s.load_edges().unwrap();
        assert!(edges
            .iter()
            .any(|(f, t, k)| *f == new.id && *t == old.id && *k == MemoryEdge::Supersedes));
    }

    #[test]
    fn delete_removes_node_and_orphans_edges() {
        let s = InMemoryStore::new();
        let a = mk("a");
        let b = mk("b");
        s.upsert(&a).unwrap();
        s.upsert(&b).unwrap();
        s.link(a.id, b.id, MemoryEdge::RelatesTo).unwrap();

        s.delete(a.id).unwrap();
        assert_eq!(s.load_all().unwrap().len(), 1);
        assert_eq!(s.load_edges().unwrap().len(), 0);
    }

    #[test]
    fn touch_increments_access() {
        let s = InMemoryStore::new();
        let n = mk("x");
        let id = n.id;
        s.upsert(&n).unwrap();
        s.touch(id).unwrap();
        s.touch(id).unwrap();
        let loaded = s.load_all().unwrap();
        assert_eq!(loaded[0].access_count, 2);
    }

    #[test]
    fn stats_count_active_vs_superseded() {
        let s = InMemoryStore::new();
        let a = mk("a");
        let b = mk("b");
        s.upsert(&a).unwrap();
        s.upsert(&b).unwrap();
        s.supersede(a.id, b.id).unwrap();
        let stats = s.stats().unwrap();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.active_memories, 1);
        assert_eq!(stats.superseded_memories, 1);
    }
}
