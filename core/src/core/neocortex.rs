//! Neocortex — slow, dense, semantic memory graph.
//!
//! This is the consolidated long-term store. Entries here have survived
//! the hippocampal buffer and been deemed worth keeping. The graph
//! structure (petgraph DiGraph) lets us do **multi-hop relational
//! retrieval** via Personalized PageRank — the core technique that
//! makes graph-based memory beat pure vector lookup on relational
//! queries.
//!
//! # Algorithm: Personalized PageRank (PPR)
//!
//! Given a set of "seed" nodes (initial keyword matches), PPR computes
//! a stationary distribution over the graph that is biased toward the
//! seeds and their structural neighbors. Nodes connected to seeds via
//! many paths score higher than isolated nodes, even if they don't
//! match the keywords.
//!
//! Math: iterate `r ← (1-α) · seeds + α · M · r` until convergence,
//! where `α` is the damping factor (0.85), `M` is the column-stochastic
//! transition matrix, and `seeds` is a one-hot vector over the seed set.
//!
//! 20 iterations are typically sufficient for convergence at α=0.85.

use std::collections::HashMap;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use uuid::Uuid;

use crate::core::hdc::{fingerprint, Hypervector};
use crate::node::{MemoryEdge, MemoryNode};
use crate::scope::Scope;

/// Damping factor for PPR. 0.85 is the standard value (Brin & Page 1998).
const PPR_DAMPING: f32 = 0.85;

/// Number of PPR iterations. 20 is comfortably past convergence at α=0.85.
const PPR_ITERATIONS: usize = 20;

/// The semantic memory graph.
pub struct Neocortex {
    graph: DiGraph<MemoryNode, MemoryEdge>,
    /// Cached HDC fingerprint per node — recomputed on insert.
    fingerprints: HashMap<NodeIndex, Hypervector>,
    /// Reverse index: memory id → node index.
    by_id: HashMap<Uuid, NodeIndex>,
}

impl Neocortex {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            fingerprints: HashMap::new(),
            by_id: HashMap::new(),
        }
    }

    /// Insert a memory. Returns the new node's index.
    pub fn insert(&mut self, node: MemoryNode) -> NodeIndex {
        let id = node.id;
        let fp = fingerprint(&node.text, &node.tags);
        let idx = self.graph.add_node(node);
        self.fingerprints.insert(idx, fp);
        self.by_id.insert(id, idx);
        idx
    }

    /// Add a typed edge between two memories. No-op if either endpoint
    /// is missing.
    pub fn link(&mut self, from: Uuid, to: Uuid, edge: MemoryEdge) {
        if let (Some(&a), Some(&b)) = (self.by_id.get(&from), self.by_id.get(&to)) {
            // Avoid duplicate edges of the same kind
            let already = self
                .graph
                .edges_connecting(a, b)
                .any(|e| *e.weight() == edge);
            if !already {
                self.graph.add_edge(a, b, edge);
            }
        }
    }

    /// Look up a memory by id.
    pub fn get(&self, id: Uuid) -> Option<&MemoryNode> {
        let idx = self.by_id.get(&id)?;
        self.graph.node_weight(*idx)
    }

    /// Mutable lookup.
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut MemoryNode> {
        let idx = self.by_id.get(&id)?;
        self.graph.node_weight_mut(*idx)
    }

    /// Look up a fingerprint by id.
    pub fn fingerprint_of(&self, id: Uuid) -> Option<&Hypervector> {
        let idx = self.by_id.get(&id)?;
        self.fingerprints.get(idx)
    }

    /// Number of memories in the neocortex.
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }

    /// Iterate all active memories with their fingerprints.
    pub fn iter_active(&self) -> impl Iterator<Item = (&MemoryNode, &Hypervector)> {
        self.graph.node_indices().filter_map(move |i| {
            let n = self.graph.node_weight(i)?;
            if !n.is_active() {
                return None;
            }
            let fp = self.fingerprints.get(&i)?;
            Some((n, fp))
        })
    }

    /// Iterate all memories, regardless of active state. Used by sync,
    /// audit, and snapshot.
    pub fn iter_all(&self) -> impl Iterator<Item = &MemoryNode> {
        self.graph
            .node_indices()
            .filter_map(|i| self.graph.node_weight(i))
    }

    /// Find ids of memories whose fingerprint is similar to `query` above
    /// `min_sim`. Returns up to `k` results sorted by similarity descending.
    pub fn nearest_by_fingerprint(
        &self,
        query: &Hypervector,
        k: usize,
        min_sim: f32,
        reader_scope: Option<&Scope>,
    ) -> Vec<(Uuid, f32)> {
        let mut scored: Vec<(Uuid, f32)> = self
            .iter_active()
            .filter(|(n, _)| match reader_scope {
                Some(scope) => scope.can_read(&n.scope),
                None => true,
            })
            .map(|(n, fp)| (n.id, fp.similarity(query)))
            .filter(|(_, s)| *s >= min_sim)
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Run Personalized PageRank from the given seed set.
    ///
    /// Returns a map of `node_id → ppr_score`. Higher scores mean more
    /// structurally relevant to the seeds.
    pub fn personalized_pagerank(&self, seeds: &[Uuid]) -> HashMap<Uuid, f32> {
        let n = self.graph.node_count();
        if n == 0 || seeds.is_empty() {
            return HashMap::new();
        }

        // Build seed mask: each seed gets equal weight.
        let seed_weight = 1.0 / seeds.len() as f32;
        let mut seed_vec = vec![0.0f32; n];
        let idx_of: HashMap<NodeIndex, usize> = self
            .graph
            .node_indices()
            .enumerate()
            .map(|(i, ni)| (ni, i))
            .collect();
        let id_of: Vec<Uuid> = self
            .graph
            .node_indices()
            .filter_map(|ni| self.graph.node_weight(ni).map(|n| n.id))
            .collect();

        for s in seeds {
            if let Some(ni) = self.by_id.get(s) {
                if let Some(&i) = idx_of.get(ni) {
                    seed_vec[i] += seed_weight;
                }
            }
        }

        let mut rank = seed_vec.clone();
        let mut next = vec![0.0f32; n];

        for _ in 0..PPR_ITERATIONS {
            for v in next.iter_mut() {
                *v = 0.0;
            }
            // Distribute current rank across outgoing and incoming edges.
            // Edge weights determine how much rank flows through an edge.
            // Incoming edges receive a 0.5 penalty to heavily favor forward causation/time.
            for ni in self.graph.node_indices() {
                let i = idx_of[&ni];
                let r = rank[i];
                if r == 0.0 {
                    continue;
                }

                let mut total_weight = 0.0_f32;
                let mut targets = Vec::new(); // (target_idx, weight_to_add)

                // Forward edges
                let mut out_edges = self.graph.edges_directed(ni, petgraph::Direction::Outgoing);
                while let Some(edge) = out_edges.next() {
                    let weight = edge.weight().weight();
                    total_weight += weight;
                    let target_idx = idx_of[&edge.target()];
                    targets.push((target_idx, weight));
                }

                // Backward edges (0.5 penalty)
                let mut in_edges = self.graph.edges_directed(ni, petgraph::Direction::Incoming);
                while let Some(edge) = in_edges.next() {
                    let weight = edge.weight().weight() * 0.5;
                    total_weight += weight;
                    let source_idx = idx_of[&edge.source()];
                    targets.push((source_idx, weight));
                }

                if total_weight < 1e-9 {
                    // Dangling node: redistribute back to seeds
                    for s in seeds {
                        if let Some(ni2) = self.by_id.get(s) {
                            if let Some(&i2) = idx_of.get(ni2) {
                                next[i2] += r * seed_weight;
                            }
                        }
                    }
                    continue;
                }

                for (target, weight) in targets {
                    next[target] += r * (weight / total_weight);
                }
            }
            // r ← (1-α) · seed + α · next
            for i in 0..n {
                rank[i] = (1.0 - PPR_DAMPING) * seed_vec[i] + PPR_DAMPING * next[i];
            }
        }

        // ── Build the result map ──
        //
        // We considered inverse-sqrt-degree hub penalization (Chakrabarti
        // 2007). It correctly suppresses pure hubs but in our domain
        // *legitimate* answers are often well-connected — a "Backend is
        // Rust + Axum" fact is the hub of all backend memories AND the
        // right answer to "what backend framework". The penalty hurt
        // direct/multihop more than it helped factual. Keeping raw PPR.
        let mut out = HashMap::with_capacity(n);
        for (i, score) in rank.iter().enumerate() {
            if *score > 1e-6 {
                out.insert(id_of[i], *score);
            }
        }
        out
    }

    /// Number of edges (for stats).
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for Neocortex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;

    fn mk(text: &str, tags: &[&str]) -> MemoryNode {
        let mut n = MemoryNode::new(text, MemoryKind::Fact, Scope::default());
        n.tags = tags.iter().map(|s| s.to_string()).collect();
        n
    }

    #[test]
    fn insert_and_fetch() {
        let mut nx = Neocortex::new();
        let a = mk("hello", &["greeting"]);
        let id = a.id;
        nx.insert(a);
        assert!(nx.get(id).is_some());
        assert_eq!(nx.len(), 1);
    }

    #[test]
    fn ppr_seeds_score_highest() {
        let mut nx = Neocortex::new();
        let a = mk("seed memory", &["seed"]);
        let b = mk("connected memory", &["connected"]);
        let c = mk("isolated memory", &["isolated"]);
        let id_a = a.id;
        let id_b = b.id;
        let id_c = c.id;
        nx.insert(a);
        nx.insert(b);
        nx.insert(c);
        nx.link(id_a, id_b, MemoryEdge::RelatesTo);

        let scores = nx.personalized_pagerank(&[id_a]);
        let sa = scores.get(&id_a).copied().unwrap_or(0.0);
        let sb = scores.get(&id_b).copied().unwrap_or(0.0);
        let sc = scores.get(&id_c).copied().unwrap_or(0.0);
        assert!(
            sa >= sb,
            "seed should score >= neighbor: sa={}, sb={}",
            sa,
            sb
        );
        assert!(
            sb > sc,
            "connected should beat isolated: sb={}, sc={}",
            sb,
            sc
        );
    }

    #[test]
    fn duplicate_edges_are_deduped() {
        let mut nx = Neocortex::new();
        let a = mk("a", &[]);
        let b = mk("b", &[]);
        let id_a = a.id;
        let id_b = b.id;
        nx.insert(a);
        nx.insert(b);
        nx.link(id_a, id_b, MemoryEdge::RelatesTo);
        nx.link(id_a, id_b, MemoryEdge::RelatesTo);
        assert_eq!(nx.edge_count(), 1);

        // But different edge kinds DO add
        nx.link(id_a, id_b, MemoryEdge::Supports);
        assert_eq!(nx.edge_count(), 2);
    }

    #[test]
    fn scope_filters_fingerprint_search() {
        let mut nx = Neocortex::new();
        let mut alice_mem = mk("alice secret", &["secret"]);
        alice_mem.scope = Scope::agent("a").with_user("alice");
        let mut bob_mem = mk("bob secret", &["secret"]);
        bob_mem.scope = Scope::agent("a").with_user("bob");
        nx.insert(alice_mem.clone());
        nx.insert(bob_mem);

        let q = fingerprint("secret", &["secret".to_string()]);
        let alice_view =
            nx.nearest_by_fingerprint(&q, 10, -1.0, Some(&Scope::agent("a").with_user("alice")));
        // Alice should only see her own memory
        for (id, _) in &alice_view {
            assert_eq!(*id, alice_mem.id);
        }
    }
}
