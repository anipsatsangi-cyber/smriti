//! Top-level Smriti facade — the public entry point.
//!
//! `Smriti` owns the hippocampus, the neocortex, and the persistence
//! store. All public memory operations (`remember`, `recall`, etc) flow
//! through this struct.
//!
//! # Lifecycle
//!
//! ```ignore
//! let mut smriti = Smriti::open("memories.db")?;
//!
//! // Write
//! smriti.remember("Alice prefers concise answers")
//!     .kind(MemoryKind::Preference)
//!     .tag("user").tag("style")
//!     .commit()?;
//!
//! // Read
//! let result = smriti.recall("how should I respond?")
//!     .budget(2000)
//!     .execute()?;
//!
//! println!("{}", result.render_text());
//! ```
//!
//! # Threading
//!
//! `Smriti` is `Send` but not `Sync`. Wrap in `Arc<Mutex<_>>` (or
//! `Arc<RwLock<_>>` for read-heavy workloads) for concurrent use.

#[cfg(feature = "native")]
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use uuid::Uuid;

use crate::core::consolidation::{consolidate, ConsolidationReport};
use crate::core::hippocampus::{Hippocampus, DEFAULT_CAPACITY};
use crate::core::neocortex::Neocortex;
#[cfg(feature = "embeddings")]
use crate::core::recall::DenseBridge;
#[cfg(not(feature = "embeddings"))]
use crate::core::recall::NoDense;
use crate::core::recall::{recall_with_dense, RecallConfig};
use crate::node::{MemoryEdge, MemoryKind, MemoryNode};
use crate::scope::Scope;
#[cfg(feature = "native")]
use crate::store::SqliteStore;
use crate::store::{InMemoryStore, Store, StoreStats};

pub use crate::core::recall::RecallResult;

/// The main Smriti facade.
pub struct Smriti {
    hippo: Hippocampus,
    neo: Neocortex,
    store: Arc<dyn Store>,
    /// Threshold for auto-consolidation: if hippo.len() >= this, we run a pass.
    auto_consolidate_threshold: usize,
    /// Optional dense-embedding layer (only present with `embeddings` feature).
    #[cfg(feature = "embeddings")]
    embedder: Option<Arc<crate::core::embeddings::Embedder>>,
    /// Cache of dense embeddings by memory id, populated lazily.
    #[cfg(feature = "embeddings")]
    dense_cache: std::sync::Mutex<std::collections::HashMap<Uuid, Vec<f32>>>,
}

impl Smriti {
    /// Open a Smriti store backed by SQLite at the given path.
    /// Use `:memory:` for an ephemeral SQLite instance, or call
    /// [`Smriti::new_ephemeral`] for a pure-Rust in-memory store
    /// (the only option on WASM targets).
    ///
    /// Available only on native targets (gated behind the `native`
    /// feature, which is on by default).
    #[cfg(feature = "native")]
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let store = Arc::new(SqliteStore::open(path)?);
        Self::with_store(store)
    }

    /// Build a Smriti instance with a pure-Rust ephemeral store.
    ///
    /// Works on every target — including `wasm32-unknown-unknown`. The
    /// data lives only in process memory and disappears when the
    /// process / page exits. Use [`Smriti::serialize_state`] /
    /// [`Smriti::restore_state`] (TODO) to persist across page loads.
    pub fn new_ephemeral() -> Result<Self> {
        let store = Arc::new(InMemoryStore::new());
        Self::with_store(store)
    }

    /// Build a Smriti instance backed by an arbitrary `Store`.
    pub fn with_store(store: Arc<dyn Store>) -> Result<Self> {
        let mut neo = Neocortex::new();
        // Hydrate the neocortex from the store. Hippocampal entries are
        // ephemeral; we don't persist them.
        for node in store.load_all()? {
            if node.is_active() {
                neo.insert(node);
            }
        }
        for (from, to, edge) in store.load_edges()? {
            neo.link(from, to, edge);
        }

        Ok(Self {
            hippo: Hippocampus::new(DEFAULT_CAPACITY),
            neo,
            store,
            auto_consolidate_threshold: DEFAULT_CAPACITY / 2,
            #[cfg(feature = "embeddings")]
            embedder: None,
            #[cfg(feature = "embeddings")]
            dense_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Enable optional dense-embedding recall.
    ///
    /// On first call, downloads the MiniLM model (~50MB) into the
    /// fastembed cache directory. Subsequent calls reuse the cached
    /// model. Once enabled, recall combines keyword + HDC + graph PPR
    /// + cosine similarity over dense vectors for a noticeably higher
    /// hit rate on synonym-heavy queries.
    #[cfg(feature = "embeddings")]
    pub fn enable_embeddings(&mut self) -> Result<()> {
        let embedder = crate::core::embeddings::Embedder::new()?;
        self.embedder = Some(embedder);
        Ok(())
    }

    /// Whether dense embeddings are currently enabled.
    pub fn has_embeddings(&self) -> bool {
        #[cfg(feature = "embeddings")]
        {
            self.embedder.is_some()
        }
        #[cfg(not(feature = "embeddings"))]
        {
            false
        }
    }

    /// Compute (and cache) a dense embedding for a memory's text.
    /// Returns `None` if embeddings are disabled or inference fails.
    #[cfg(feature = "embeddings")]
    pub(crate) fn dense_for(&self, id: Uuid, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        {
            let cache = self.dense_cache.lock().ok()?;
            if let Some(v) = cache.get(&id) {
                return Some(v.clone());
            }
        }
        let v = embedder.embed_one(text).ok()?;
        if let Ok(mut cache) = self.dense_cache.lock() {
            cache.insert(id, v.clone());
        }
        Some(v)
    }

    /// Embed a query text directly (no caching).
    #[cfg(feature = "embeddings")]
    pub(crate) fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        let embedder = self.embedder.as_ref()?;
        embedder.embed_one(text).ok()
    }

    /// Build a `RememberBuilder`.
    pub fn remember(&mut self, text: impl Into<String>) -> RememberBuilder<'_> {
        RememberBuilder::new(self, text.into())
    }

    /// Build a `RecallBuilder`.
    pub fn recall(&self, query: impl Into<String>) -> RecallBuilder<'_> {
        RecallBuilder::new(self, query.into())
    }

    /// Mark `old_id` as superseded by `new_id`. Both must already exist
    /// (in either the hippocampus or the neocortex).
    pub fn supersede(&mut self, old_id: Uuid, new_id: Uuid) -> Result<()> {
        // Update the neocortex view (if the memory has been consolidated).
        if let Some(node) = self.neo.get_mut(old_id) {
            node.superseded_by = Some(new_id);
            self.store.upsert(node)?;
        }
        if let Some(node) = self.neo.get_mut(new_id) {
            node.supersedes = Some(old_id);
            self.store.upsert(node)?;
        }
        // Also update the hippocampal view if either memory is still there.
        // We rebuild the hippocampus by draining + reinserting with the
        // updated state.
        let mut updated_entries: Vec<crate::core::hippocampus::EpisodicEntry> = self
            .hippo
            .drain_where(|_| true)
            .into_iter()
            .map(|mut e| {
                if e.node.id == old_id {
                    e.node.superseded_by = Some(new_id);
                } else if e.node.id == new_id {
                    e.node.supersedes = Some(old_id);
                }
                e
            })
            .collect();
        // Reinsert in original order.
        for e in updated_entries.drain(..) {
            // Bypass insert's auto-eviction since we know order is unchanged.
            self.hippo.insert(e.node);
        }
        // Persist the supersede chain.
        self.store.supersede(old_id, new_id)?;
        self.neo.link(new_id, old_id, MemoryEdge::Supersedes);
        Ok(())
    }

    /// Soft-delete by superseding with a tombstone. For hard delete, use
    /// `forget_hard`.
    pub fn forget(&mut self, id: Uuid) -> Result<()> {
        if let Some(node) = self.neo.get_mut(id) {
            node.superseded_by = Some(id); // self-supersede = soft tombstone
            self.store.upsert(node)?;
        }
        Ok(())
    }

    /// Hard delete — removes from store and neocortex entirely.
    pub fn forget_hard(&mut self, id: Uuid) -> Result<()> {
        self.store.delete(id)?;
        // Note: petgraph doesn't expose easy node removal that updates
        // node indices safely; for v1 we tolerate the in-memory
        // tombstone after a hard delete. A future version can rebuild
        // the neocortex from store on a schedule.
        if let Some(node) = self.neo.get_mut(id) {
            node.superseded_by = Some(id);
        }
        Ok(())
    }

    /// Manually link two memories.
    pub fn link(&mut self, from: Uuid, to: Uuid, edge: MemoryEdge) -> Result<()> {
        self.neo.link(from, to, edge);
        self.store.link(from, to, edge)?;
        Ok(())
    }

    /// Force a consolidation pass: drain the hippocampus into the neocortex.
    pub fn consolidate(&mut self) -> Result<ConsolidationReport> {
        let report = consolidate(&mut self.hippo, &mut self.neo, usize::MAX);
        // Persist any new nodes / reinforcements.
        for node in self.neo.iter_all() {
            self.store.upsert(node)?;
        }
        // Pre-warm dense cache for any newly-consolidated memories so the
        // first `recall()` after consolidation doesn't pay the embedding
        // cost on the hot path. This is a no-op when embeddings are off.
        #[cfg(feature = "embeddings")]
        self.prewarm_dense_cache();
        Ok(report)
    }

    /// Embed every active neocortex memory not yet in the dense cache.
    /// Runs lazily under the consolidation pass — this trades a one-time
    /// O(N) cost at consolidation for amortized 0 µs at recall time.
    #[cfg(feature = "embeddings")]
    fn prewarm_dense_cache(&self) {
        if self.embedder.is_none() {
            return;
        }
        // Snapshot the IDs we still need to embed under the lock, then
        // drop the lock before running inference (which is the slow part).
        let pending: Vec<(Uuid, String)> = {
            let cache = match self.dense_cache.lock() {
                Ok(c) => c,
                Err(_) => return,
            };
            self.neo
                .iter_active()
                .filter_map(|(node, _)| {
                    if cache.contains_key(&node.id) {
                        None
                    } else {
                        Some((node.id, node.text.clone()))
                    }
                })
                .collect()
        };
        for (id, text) in pending {
            // dense_for caches as a side effect; ignore the result.
            let _ = self.dense_for(id, &text);
        }
    }

    /// Stats from the underlying store + in-memory views.
    pub fn stats(&self) -> Result<SmritiStats> {
        let store_stats = self.store.stats()?;
        Ok(SmritiStats {
            store: store_stats,
            hippocampus_size: self.hippo.len(),
            hippocampus_capacity: self.hippo.capacity(),
            neocortex_size: self.neo.len(),
            neocortex_edges: self.neo.edge_count(),
        })
    }

    /// Direct access to a memory by id (read-only).
    pub fn get(&self, id: Uuid) -> Option<&MemoryNode> {
        self.neo.get(id)
    }

    fn maybe_auto_consolidate(&mut self) -> Result<()> {
        if self.hippo.len() >= self.auto_consolidate_threshold {
            self.consolidate()?;
        }
        Ok(())
    }
}

/// Builder for `remember()`.
pub struct RememberBuilder<'a> {
    smriti: &'a mut Smriti,
    text: String,
    tags: Vec<String>,
    kind: MemoryKind,
    scope: Scope,
    importance: f32,
    supersedes: Option<Uuid>,
    auto_tag: bool,
}

impl<'a> RememberBuilder<'a> {
    fn new(smriti: &'a mut Smriti, text: String) -> Self {
        Self {
            smriti,
            text,
            tags: Vec::new(),
            kind: MemoryKind::Fact,
            scope: Scope::default(),
            importance: 0.5,
            supersedes: None,
            auto_tag: true,
        }
    }

    /// Disable automatic NER-based tag extraction. By default, Smriti
    /// extracts proper nouns, domain keywords, and verb stems from the
    /// memory text and merges them with user-supplied tags. This builds
    /// a richer auto-link graph during consolidation. Disable this if
    /// you want full control over tagging.
    pub fn no_auto_tags(mut self) -> Self {
        self.auto_tag = false;
        self
    }

    pub fn kind(mut self, kind: MemoryKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn tags<I: IntoIterator<Item = impl Into<String>>>(mut self, tags: I) -> Self {
        for t in tags {
            self.tags.push(t.into());
        }
        self
    }

    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    pub fn importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn supersedes(mut self, old_id: Uuid) -> Self {
        self.supersedes = Some(old_id);
        self
    }

    /// Commit the memory. Returns the new memory's id.
    pub fn commit(self) -> Result<Uuid> {
        use crate::node::TagSource;
        let (final_tags, final_sources) = if self.auto_tag {
            let extracted = crate::core::ner::extract_tags(&self.text);
            crate::core::ner::merge_tags_with_sources(&self.tags, &extracted)
        } else {
            // No auto-tagging: every tag came directly from the caller.
            let n = self.tags.len();
            (self.tags, vec![TagSource::User; n])
        };

        let mut node = MemoryNode::new(self.text, self.kind, self.scope.clone());
        node.set_tags_with_sources(final_tags, final_sources);
        node.importance = self.importance;
        node.supersedes = self.supersedes;
        let id = node.id;

        // Persist immediately so we never lose data.
        self.smriti.store.upsert(&node)?;

        // Drop into the hippocampus for fast access; consolidation will
        // promote it later.
        self.smriti.hippo.insert(node);

        // If superseding, finalize the link.
        if let Some(old) = self.supersedes {
            self.smriti.supersede(old, id)?;
        }

        // Auto-consolidate if buffer is filling up.
        self.smriti.maybe_auto_consolidate()?;

        Ok(id)
    }
}

/// Builder for `recall()`.
pub struct RecallBuilder<'a> {
    smriti: &'a Smriti,
    query: String,
    tags: Vec<String>,
    scope: Scope,
    cfg: RecallConfig,
}

impl<'a> RecallBuilder<'a> {
    fn new(smriti: &'a Smriti, query: String) -> Self {
        Self {
            smriti,
            query,
            tags: Vec::new(),
            scope: Scope::default(),
            cfg: RecallConfig::default(),
        }
    }

    pub fn budget(mut self, tokens: usize) -> Self {
        self.cfg.budget = tokens;
        self
    }

    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn tags<I: IntoIterator<Item = impl Into<String>>>(mut self, tags: I) -> Self {
        for t in tags {
            self.tags.push(t.into());
        }
        self
    }

    pub fn scope(mut self, scope: Scope) -> Self {
        self.scope = scope;
        self
    }

    pub fn kinds(mut self, kinds: Vec<MemoryKind>) -> Self {
        self.cfg.kinds = kinds;
        self
    }

    pub fn lambda(mut self, lambda: f32) -> Self {
        self.cfg.mmr_lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Enable confidence-conditional truncation. When the engine's
    /// verdict on the top hit is `Confident`, the result is truncated
    /// to `confident_keep` hits (default suggested: 1). When the
    /// verdict is `LowConfidence`, it is truncated to `low_keep`
    /// (default suggested: 0 — return nothing). When `AmbiguousLeader`,
    /// it is truncated to `ambiguous_keep`.
    ///
    /// This is the knob that turns a 489-token average pack into a
    /// ~25-token confident answer plus a fuller fallback only when
    /// the engine is unsure. Off by default — opt-in.
    pub fn confident_truncation(
        mut self,
        confident_keep: usize,
        ambiguous_keep: usize,
        low_keep: usize,
    ) -> Self {
        self.cfg.truncate_when_confident = confident_keep;
        self.cfg.truncate_when_ambiguous = ambiguous_keep;
        self.cfg.truncate_when_low = low_keep;
        self
    }

    /// Tiered Confident truncation: when the verdict is `Confident`
    /// AND the top hit's `final_score` is ≥ `score_floor` AND the
    /// top1/top2 margin is ≥ `margin_floor`, the result is truncated
    /// to a single hit (overriding `confident_keep`). This catches
    /// the "extremely sure" subset and ships ~25-50 tokens on those
    /// while keeping the hedged-pair pack for ordinary-sure cases.
    pub fn confident_solo(mut self, score_floor: f32, margin_floor: f32) -> Self {
        self.cfg.confident_solo_score_floor = score_floor;
        self.cfg.confident_solo_margin_floor = margin_floor;
        self
    }

    pub fn execute(self) -> Result<RecallResult> {
        // Auto-augment query tags with NER-extracted entities from the
        // query text. This helps when the user asks an open-ended
        // question that didn't include explicit tags.
        let extracted = crate::core::ner::extract_tags(&self.query);
        let merged_tags = crate::core::ner::merge_tags(&self.tags, &extracted);

        // If embeddings are enabled, build a bridge that the recall
        // pipeline can call to compute cosine similarity per candidate.
        #[cfg(feature = "embeddings")]
        let result = {
            let bridge = SmritiDenseBridge {
                smriti: self.smriti,
            };
            recall_with_dense(
                &self.query,
                &merged_tags,
                &self.scope,
                &self.smriti.hippo,
                &self.smriti.neo,
                &self.cfg,
                &bridge,
            )
        };
        #[cfg(not(feature = "embeddings"))]
        let result = recall_with_dense(
            &self.query,
            &merged_tags,
            &self.scope,
            &self.smriti.hippo,
            &self.smriti.neo,
            &self.cfg,
            &NoDense,
        );
        // Touch every hit's access stats.
        for hit in &result.hits {
            let _ = self.smriti.store.touch(hit.node.id);
        }
        Ok(result)
    }
}

/// Dense-embedding bridge — wraps a `Smriti` reference so the recall
/// pipeline can ask for query/memory cosine similarity without owning
/// the embedder directly.
#[cfg(feature = "embeddings")]
struct SmritiDenseBridge<'a> {
    smriti: &'a Smriti,
}

#[cfg(feature = "embeddings")]
impl<'a> DenseBridge for SmritiDenseBridge<'a> {
    fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        self.smriti.embed_query(text)
    }

    fn similarity(&self, query: &[f32], id: Uuid, text: &str) -> f32 {
        let v = self.smriti.dense_for(id, text);
        match v {
            Some(v) => crate::core::embeddings::cosine(query, &v),
            None => 0.0,
        }
    }
}

/// Aggregate stats for monitoring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SmritiStats {
    pub store: StoreStats,
    pub hippocampus_size: usize,
    pub hippocampus_capacity: usize,
    pub neocortex_size: usize,
    pub neocortex_edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_and_recall_roundtrip() {
        let mut s = Smriti::open(":memory:").unwrap();
        s.remember("the auth module uses JWT RS256")
            .kind(MemoryKind::Fact)
            .tag("auth")
            .tag("security")
            .commit()
            .unwrap();
        s.remember("user prefers concise responses")
            .kind(MemoryKind::Preference)
            .tag("user")
            .tag("style")
            .commit()
            .unwrap();

        // Force consolidation so neocortex sees them
        s.consolidate().unwrap();

        let r = s
            .recall("how does authentication work")
            .budget(500)
            .execute()
            .unwrap();
        assert!(!r.hits.is_empty());
        assert!(r.hits[0].node.text.contains("auth"));
    }

    #[test]
    fn supersede_hides_old_from_recall() {
        let mut s = Smriti::open(":memory:").unwrap();
        let old = s
            .remember("user's name is Bob")
            .kind(MemoryKind::Fact)
            .tag("user")
            .commit()
            .unwrap();
        s.consolidate().unwrap();

        let new_id = s
            .remember("user's name is Robert")
            .kind(MemoryKind::Fact)
            .tag("user")
            .supersedes(old)
            .commit()
            .unwrap();
        s.consolidate().unwrap();

        let r = s.recall("user's name").budget(500).execute().unwrap();
        // The "Robert" memory wins; "Bob" is hidden.
        let any_robert = r.hits.iter().any(|h| h.node.text.contains("Robert"));
        let any_bob = r.hits.iter().any(|h| h.node.text.contains("Bob"));
        assert!(any_robert, "expected Robert in recall");
        assert!(!any_bob, "Bob should be hidden after supersede");
        let _ = new_id;
    }

    #[test]
    fn stats_reflect_state() {
        let mut s = Smriti::open(":memory:").unwrap();
        s.remember("a").commit().unwrap();
        s.remember("b").commit().unwrap();
        s.consolidate().unwrap();
        let stats = s.stats().unwrap();
        assert_eq!(stats.store.total_memories, 2);
        assert_eq!(stats.neocortex_size, 2);
    }
}
