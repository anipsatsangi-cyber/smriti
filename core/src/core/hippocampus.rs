//! Hippocampus — fast episodic buffer for recent memories.
//!
//! Implements the "fast learner" half of the Complementary Learning Systems
//! theory (McClelland, McNaughton & O'Reilly 1995). Key properties:
//!
//! - **Sparse pattern separation** — each memory stored with its HDC
//!   fingerprint. Two distinct episodes have orthogonal fingerprints
//!   (hamming distance > HV_DIM/2 ± noise) so they cannot interfere.
//! - **Bounded capacity** — fixed-size circular buffer. When full, the
//!   oldest entries are evicted (or consolidated to the neocortex).
//! - **Recency-biased recall** — a query against the hippocampus returns
//!   the most recent k matches, regardless of importance.
//!
//! This is the "what happened in the last hour/day" store. Things that
//! prove durable migrate to the neocortex via `consolidation`. Things
//! that don't simply fall out of the buffer.

use std::collections::VecDeque;
use uuid::Uuid;

use crate::core::hdc::{fingerprint, Hypervector};
use crate::node::MemoryNode;

/// Default hippocampal capacity. Exceeding this triggers consolidation
/// or eviction of the oldest entry.
pub const DEFAULT_CAPACITY: usize = 1024;

/// A single episodic entry in the hippocampus. Stores the full memory
/// node plus its HDC fingerprint (cached for fast similarity queries).
#[derive(Debug, Clone)]
pub struct EpisodicEntry {
    pub node: MemoryNode,
    pub fp: Hypervector,
}

impl EpisodicEntry {
    pub fn new(node: MemoryNode) -> Self {
        let fp = fingerprint(&node.text, &node.tags);
        Self { node, fp }
    }
}

/// The hippocampal buffer. Bounded queue with fingerprint-indexed search.
#[derive(Debug)]
pub struct Hippocampus {
    capacity: usize,
    entries: VecDeque<EpisodicEntry>,
}

impl Hippocampus {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::with_capacity(capacity.max(1)),
        }
    }

    /// Insert a new episodic memory. Returns the evicted entry if the
    /// buffer was full.
    pub fn insert(&mut self, node: MemoryNode) -> Option<EpisodicEntry> {
        let evicted = if self.entries.len() >= self.capacity {
            self.entries.pop_front()
        } else {
            None
        };
        self.entries.push_back(EpisodicEntry::new(node));
        evicted
    }

    /// All entries, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &EpisodicEntry> {
        self.entries.iter()
    }

    /// Number of entries currently in the buffer.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up an entry by id.
    pub fn get(&self, id: Uuid) -> Option<&EpisodicEntry> {
        self.entries.iter().find(|e| e.node.id == id)
    }

    /// Look up an entry by id (mutable).
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut EpisodicEntry> {
        self.entries.iter_mut().find(|e| e.node.id == id)
    }
    
    /// Remove an entry by id. Returns the entry if found.
    pub fn remove(&mut self, id: Uuid) -> Option<EpisodicEntry> {
        let pos = self.entries.iter().position(|e| e.node.id == id)?;
        self.entries.remove(pos)
    }

    /// Drain entries that match a predicate. Used by consolidation to
    /// move entries to the neocortex.
    pub fn drain_where<F>(&mut self, mut pred: F) -> Vec<EpisodicEntry>
    where
        F: FnMut(&EpisodicEntry) -> bool,
    {
        let mut out = Vec::new();
        let mut keep = VecDeque::with_capacity(self.entries.len());
        while let Some(entry) = self.entries.pop_front() {
            if pred(&entry) {
                out.push(entry);
            } else {
                keep.push_back(entry);
            }
        }
        self.entries = keep;
        out
    }

    /// Find the top-`k` entries most similar to the query fingerprint.
    /// Returns `(entry, similarity)` pairs sorted by similarity descending.
    pub fn nearest(&self, query: &Hypervector, k: usize) -> Vec<(&EpisodicEntry, f32)> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(&EpisodicEntry, f32)> = self
            .entries
            .iter()
            .map(|e| (e, e.fp.similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for Hippocampus {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;
    use crate::scope::Scope;

    fn mk(text: &str) -> MemoryNode {
        MemoryNode::new(text, MemoryKind::Event, Scope::default())
    }

    #[test]
    fn insertion_under_capacity_does_not_evict() {
        let mut h = Hippocampus::new(4);
        for i in 0..3 {
            let evicted = h.insert(mk(&format!("memory {}", i)));
            assert!(evicted.is_none());
        }
        assert_eq!(h.len(), 3);
    }

    #[test]
    fn insertion_at_capacity_evicts_oldest() {
        let mut h = Hippocampus::new(2);
        let n1 = mk("first");
        let id1 = n1.id;
        h.insert(n1);
        h.insert(mk("second"));
        let evicted = h.insert(mk("third")).unwrap();
        assert_eq!(evicted.node.id, id1);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn nearest_returns_most_similar_first() {
        let mut h = Hippocampus::default();
        h.insert(mk("the auth module uses JWT tokens"));
        h.insert(mk("user prefers dark mode"));
        h.insert(mk("authentication via JSON Web Tokens"));

        let q = fingerprint("how does auth work", &["auth".to_string()]);
        let top = h.nearest(&q, 2);
        assert_eq!(top.len(), 2);
        // First result must contain auth-related content
        assert!(
            top[0].0.node.text.contains("auth") || top[0].0.node.text.contains("authentication")
        );
    }

    #[test]
    fn drain_where_removes_matching() {
        let mut h = Hippocampus::default();
        h.insert(mk("keep me"));
        h.insert(mk("drain this"));
        h.insert(mk("keep me too"));

        let drained = h.drain_where(|e| e.node.text.starts_with("drain"));
        assert_eq!(drained.len(), 1);
        assert_eq!(h.len(), 2);
    }
}
