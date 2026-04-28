//! Persistence layer.
//!
//! The `Store` trait abstracts platform-specific storage so the same
//! Smriti core can run on SQLite (native), in-memory (WASM, tests), or
//! any other backend.
//!
//! Two implementations ship today:
//! - [`InMemoryStore`] — pure Rust, ephemeral. Available on every target
//!   including `wasm32-unknown-unknown`.
//! - [`SqliteStore`] — durable, single-file. Available only on native
//!   targets (gated behind the `native` feature).

pub mod in_memory;

#[cfg(feature = "native")]
pub mod sqlite;

pub use in_memory::InMemoryStore;

#[cfg(feature = "native")]
pub use sqlite::SqliteStore;

use anyhow::Result;
use uuid::Uuid;

use crate::node::{MemoryEdge, MemoryNode};
use crate::scope::Scope;

/// Storage backend trait. Must be `Send + Sync` so it can live behind an
/// `Arc<Mutex<_>>` in async contexts.
pub trait Store: Send + Sync {
    /// Insert or update a memory node.
    fn upsert(&self, node: &MemoryNode) -> Result<()>;

    /// Insert an edge between two memories.
    fn link(&self, from: Uuid, to: Uuid, edge: MemoryEdge) -> Result<()>;

    /// Load every memory matching a scope predicate.
    fn load_all(&self) -> Result<Vec<MemoryNode>>;

    /// Load all edges as `(from, to, kind)` triples.
    fn load_edges(&self) -> Result<Vec<(Uuid, Uuid, MemoryEdge)>>;

    /// Mark a memory as superseded by another.
    fn supersede(&self, old_id: Uuid, new_id: Uuid) -> Result<()>;

    /// Hard-delete a memory. Use with care; prefer `supersede` for audit
    /// preservation.
    fn delete(&self, id: Uuid) -> Result<()>;

    /// Bump access stats: increment counter and update `last_accessed_at`.
    fn touch(&self, id: Uuid) -> Result<()>;

    /// Stats: total memory count, scope distribution.
    fn stats(&self) -> Result<StoreStats>;

    /// Filter memories by scope (used by recall).
    fn load_for_scope(&self, scope: &Scope) -> Result<Vec<MemoryNode>>;
}

/// Aggregate statistics from the store.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StoreStats {
    pub total_memories: usize,
    pub active_memories: usize,
    pub superseded_memories: usize,
    pub total_edges: usize,
    pub total_tokens: usize,
}
