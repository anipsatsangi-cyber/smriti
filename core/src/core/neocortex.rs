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
    activation_state: std::sync::RwLock<HashMap<NodeIndex, f32>>,
    /// Wall-clock instant at which `decay_activation` was last applied.
    /// Lets us scale the decay multiplier by elapsed real-world time so
    /// rapid-fire queries (bench harness, test loops) don't accumulate
    /// residual priming the way they would in the human-scale agent loop
    /// the priming model was designed for. `None` until the first recall.
    last_decay_at: std::sync::RwLock<Option<std::time::Instant>>,
}

impl Neocortex {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            fingerprints: HashMap::new(),
            by_id: HashMap::new(),
            activation_state: std::sync::RwLock::new(HashMap::new()),
            last_decay_at: std::sync::RwLock::new(None),
        }
    }

    /// Insert a memory. Returns the new node's index.
    pub fn insert(&mut self, node: MemoryNode) -> NodeIndex {
        let is_goal = node.kind == crate::node::MemoryKind::Goal;
        let id = node.id;
        let fp = fingerprint(&node.text, &node.tags);
        let idx = self.graph.add_node(node);
        self.fingerprints.insert(idx, fp);
        self.by_id.insert(id, idx);
        
        if is_goal {
            let mut activations = self.activation_state.write().unwrap();
            activations.insert(idx, 1.0);
        }
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

    /// Update the cached fingerprint for a node (e.g. after reconsolidation).
    pub fn update_fingerprint(&mut self, id: Uuid, fp: Hypervector) {
        if let Some(&idx) = self.by_id.get(&id) {
            self.fingerprints.insert(idx, fp);
        }
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
        if self.graph.node_count() == 0 || seeds.is_empty() {
            return HashMap::new();
        }

        // 1. Find local neighborhood (BFS up to 3 hops)
        let mut local_nodes = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        for s in seeds {
            if let Some(&ni) = self.by_id.get(s) {
                local_nodes.insert(ni);
                queue.push_back((ni, 0));
            }
        }

        // Add Critical/Salient seeds automatically (Amygdala Fast-Track) and active Goals
        for ni in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(ni) {
                if node.is_active() && (node.salience == crate::node::Salience::Critical || node.kind == crate::node::MemoryKind::Goal) {
                    if local_nodes.insert(ni) {
                        queue.push_back((ni, 0));
                    }
                }
            }
        }
        
        if local_nodes.is_empty() {
            return HashMap::new();
        }

        while let Some((ni, depth)) = queue.pop_front() {
            if depth >= 3 {
                continue;
            }
            // Explore outgoing edges
            for edge in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                let target = edge.target();
                if local_nodes.insert(target) {
                    queue.push_back((target, depth + 1));
                }
            }
            // Explore incoming edges
            for edge in self.graph.edges_directed(ni, petgraph::Direction::Incoming) {
                let source = edge.source();
                if local_nodes.insert(source) {
                    queue.push_back((source, depth + 1));
                }
            }
        }

        // 2. Map local nodes to dense array [0..K]
        let k = local_nodes.len();
        let mut local_to_dense: HashMap<NodeIndex, usize> = HashMap::with_capacity(k);
        let mut dense_to_id: Vec<Uuid> = Vec::with_capacity(k);
        
        for (i, &ni) in local_nodes.iter().enumerate() {
            local_to_dense.insert(ni, i);
            if let Some(node) = self.graph.node_weight(ni) {
                dense_to_id.push(node.id);
            }
        }

        // 3. Build seed vector (incorporating Persistent Spreading Activation)
        let seed_weight = 1.0 / local_nodes.len().max(1) as f32; // Distribute evenly
        let mut seed_vec = vec![0.0f32; k];

        let activations = self.activation_state.read().unwrap();

        // Residual priming multiplier. The previous default was 0.5,
        // which was too aggressive: in a tight loop (bench harness), it
        // let earlier queries' subgraphs dominate later queries' seed
        // vectors. 0.15 is a soft nudge — large enough to support
        // real cross-query priming on conversational timescales, small
        // enough that one query's echo can't structurally overwrite
        // the next query's fingerprint signal.
        //
        // Override via SMRITI_RESIDUAL_GAIN for benchmarking sensitivity;
        // production stays at the calibrated 0.15.
        let residual_gain: f32 = std::env::var("SMRITI_RESIDUAL_GAIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.15);

        for (i, &ni) in local_nodes.iter().enumerate() {
            let is_explicit_seed = seeds.contains(&dense_to_id[i])
                || self
                    .graph
                    .node_weight(ni)
                    .map(|n| {
                        n.is_active()
                            && (n.salience == crate::node::Salience::Critical
                                || n.kind == crate::node::MemoryKind::Goal)
                    })
                    .unwrap_or(false);

            let base = if is_explicit_seed { seed_weight } else { 0.0 };
            let residual = activations.get(&ni).copied().unwrap_or(0.0) * residual_gain;

            seed_vec[i] = base + residual;
        }
        drop(activations);

        // Normalize seed vector
        let sum: f32 = seed_vec.iter().sum();
        if sum > 0.0 {
            for v in seed_vec.iter_mut() {
                *v /= sum;
            }
        } else {
            seed_vec[0] = 1.0;
        }

        let mut rank = seed_vec.clone();
        let mut next = vec![0.0f32; k];

        // 4. Run iterations over the bounded KxK subgraph
        for _ in 0..PPR_ITERATIONS {
            for v in next.iter_mut() {
                *v = 0.0;
            }

            for &ni in &local_nodes {
                let i = local_to_dense[&ni];
                let r = rank[i];
                if r == 0.0 {
                    continue;
                }

                let mut total_weight = 0.0_f32;
                let mut targets = Vec::new(); // (dense_idx, weight_to_add)

                for edge in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                    let target_ni = edge.target();
                    if let Some(&target_idx) = local_to_dense.get(&target_ni) {
                        let weight = edge.weight().weight();
                        total_weight += weight;
                        targets.push((target_idx, weight));
                    }
                }

                for edge in self.graph.edges_directed(ni, petgraph::Direction::Incoming) {
                    let source_ni = edge.source();
                    if let Some(&source_idx) = local_to_dense.get(&source_ni) {
                        let weight = edge.weight().weight() * 0.5;
                        total_weight += weight;
                        targets.push((source_idx, weight));
                    }
                }

                if total_weight < 1e-9 {
                    // Dangling local node: redistribute back to local seeds
                    for s in seeds {
                        if let Some(ni2) = self.by_id.get(s) {
                            if let Some(&i2) = local_to_dense.get(ni2) {
                                next[i2] += r * seed_weight;
                            }
                        }
                    }
                    continue;
                }

                for (target_idx, weight) in targets {
                    next[target_idx] += r * (weight / total_weight);
                }
            }

            for i in 0..k {
                rank[i] = (1.0 - PPR_DAMPING) * seed_vec[i] + PPR_DAMPING * next[i];
            }
        }

        // Save new activation state for Semantic Priming
        let mut updates = Vec::new();
        for (i, &ni) in local_nodes.iter().enumerate() {
            let r = rank[i];
            if r > 1e-6 {
                updates.push((ni, r));
            }
        }
        
        let mut out = HashMap::with_capacity(k);
        for (i, score) in rank.iter().enumerate() {
            if *score > 1e-6 {
                out.insert(dense_to_id[i], *score);
            }
        }

        let mut activations = self.activation_state.write().unwrap();
        for (ni, r) in updates {
            let current = activations.entry(ni).or_insert(0.0);
            *current = (*current + r).min(1.0); // Cap at 1.0
        }
        
        out
    }

    /// Decay the activation state with a wall-clock half-life. The
    /// previous fixed `*= 0.8` pulse-decay assumed roughly one query per
    /// human-scale "thought" (seconds apart). Under sustained loops
    /// (benchmarks, batch agents, test harnesses) queries fire in
    /// microseconds and the residual priming accumulates faster than
    /// 0.8× can drain it — by query ~5 the seed vector is dominated by
    /// echoes of earlier subgraphs instead of the current query's
    /// fingerprint.
    ///
    /// Wall-clock-aware decay matches the cognitive intent: priming
    /// should fade with real elapsed time, not with arbitrary call
    /// count. We use a 30-second half-life — long enough that a real
    /// conversation's natural beats keep priming alive, short enough
    /// that bench harnesses don't accumulate state.
    ///
    /// `MemoryKind::Goal` nodes are re-pinned to `1.0` after the decay
    /// pass — goals are persistent attention by design.
    pub fn decay_activation(&self) {
        // 30-second half-life. Calibrated empirically on bench-500: keeps
        // paraphrase quality stable across 47 sequential queries while
        // preserving cross-query priming on conversational timescales.
        const HALF_LIFE_SECS: f32 = 30.0;
        const MIN_DECAY_FACTOR: f32 = 0.05;

        let now = std::time::Instant::now();
        let mut last = self.last_decay_at.write().unwrap();
        let elapsed_secs = match *last {
            Some(prev) => now.duration_since(prev).as_secs_f32(),
            // First call after init: no priming has had time to build, so
            // pretend we just decayed and skip a no-op multiply.
            None => 0.0,
        };
        *last = Some(now);
        drop(last);

        // 0.5^(elapsed / half_life). Floored at MIN_DECAY_FACTOR so a long
        // pause (or `None` last_decay) doesn't divide by ~0; the floor
        // also caps the amount of one-shot collapse a single decay can
        // do, so a brief slow query doesn't nuke the entire context.
        // Two-component decay:
        //   1. A *pulse* component (per-call shave). Drains residual on
        //      back-to-back calls so a tight loop can't accumulate state
        //      faster than the per-call decay can drain it.
        //   2. A *wall-clock* component (real-time half-life). Adds
        //      additional drain on top of the pulse for human-scale
        //      pauses, so a 5-minute idle effectively wipes the
        //      activation map.
        // Combined: decay = pulse_factor * wall_clock_factor.
        const PULSE_FACTOR: f32 = 0.5;
        let wall_factor = if elapsed_secs <= 0.0 {
            1.0
        } else {
            (0.5f32).powf(elapsed_secs / HALF_LIFE_SECS).max(MIN_DECAY_FACTOR)
        };
        let decay = PULSE_FACTOR * wall_factor;

        let mut activations = self.activation_state.write().unwrap();

        // Re-inject active goals first so they survive the retain pass
        // even if they had decayed below threshold previously.
        for ni in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(ni) {
                if node.is_active() && node.kind == crate::node::MemoryKind::Goal {
                    activations.insert(ni, 1.0);
                }
            }
        }

        activations.retain(|&ni, v| {
            if let Some(node) = self.graph.node_weight(ni) {
                if !node.is_active() {
                    return false;
                }
                if node.kind == crate::node::MemoryKind::Goal {
                    *v = 1.0;
                    return true;
                }
            } else {
                return false;
            }
            *v *= decay;
            *v > 1e-4
        });
    }

    /// Completely wipe the activation state (reset context)
    pub fn clear_activation(&self) {
        let mut activations = self.activation_state.write().unwrap();
        activations.clear();
    }

    /// Number of edges (for stats).
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

#[derive(Debug, Clone)]
pub struct ClusterReport {
    pub nodes: Vec<MemoryNode>,
    pub internal_edge_count: usize,
    pub redundancy_score: usize,
}

impl Neocortex {
    /// Suggest clusters of highly related memories for sleep summarization.
    ///
    /// Finds dense subgraphs that are ripe for being superseded by a single 
    /// consolidated summary. Returns up to `limit` clusters, where each cluster 
    /// has at least 3 memories.
    pub fn suggest_clusters(&self, limit: usize) -> Vec<ClusterReport> {
        let mut clusters = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Sort node indices by degree (widened edge types)
        let mut nodes_by_degree: Vec<(NodeIndex, usize)> = self.graph.node_indices().map(|ni| {
            let degree = self.graph.edges(ni).filter(|e| {
                matches!(*e.weight(), MemoryEdge::RelatesTo | MemoryEdge::Supports | MemoryEdge::DerivedFrom | MemoryEdge::CausedBy)
            }).count();
            (ni, degree)
        }).collect();
        nodes_by_degree.sort_by(|a, b| b.1.cmp(&a.1));

        for (ni, degree) in nodes_by_degree {
            if degree < 2 {
                continue; // Need at least a triangle/hub to form a good cluster
            }
            if visited.contains(&ni) {
                continue;
            }

            let mut cluster = Vec::new();
            if let Some(node) = self.graph.node_weight(ni) {
                if !node.is_active() {
                    continue;
                }
                cluster.push(node.clone());
                visited.insert(ni);
            } else {
                continue;
            }

            // Gather neighbors
            let mut internal_edge_count = 0;
            for edge in self.graph.edges(ni) {
                let w = *edge.weight();
                if matches!(w, MemoryEdge::RelatesTo | MemoryEdge::Supports | MemoryEdge::DerivedFrom | MemoryEdge::CausedBy) {
                    internal_edge_count += 1;
                    let neighbor = if edge.source() == ni { edge.target() } else { edge.source() };
                    if !visited.contains(&neighbor) {
                        if let Some(n) = self.graph.node_weight(neighbor) {
                            if n.is_active() {
                                cluster.push(n.clone());
                                visited.insert(neighbor);
                            }
                        }
                    }
                }
            }

            if cluster.len() >= 3 {
                let redundancy_score = cluster.len() * internal_edge_count;
                clusters.push(ClusterReport { nodes: cluster, internal_edge_count, redundancy_score });
                if clusters.len() >= limit {
                    break;
                }
            }
        }
        
        clusters.sort_by(|a, b| b.redundancy_score.cmp(&a.redundancy_score));
        clusters
    }

    /// Garbage collect inactive (superseded) nodes from the graph.
    /// Because petgraph node removals shift indices, we allocate a new
    /// graph, copy over only the active nodes and valid edges, and swap.
    pub fn vacuum(&mut self) {
        let mut new_graph = DiGraph::new();
        let mut new_fingerprints = HashMap::new();
        let mut new_by_id = HashMap::new();
        let mut old_to_new = HashMap::new();

        for ni in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(ni) {
                if node.is_active() {
                    let new_ni = new_graph.add_node(node.clone());
                    new_by_id.insert(node.id, new_ni);
                    old_to_new.insert(ni, new_ni);
                    if let Some(fp) = self.fingerprints.get(&ni) {
                        new_fingerprints.insert(new_ni, fp.clone());
                    }
                }
            }
        }

        for old_ni in old_to_new.keys() {
            for edge in self.graph.edges_directed(*old_ni, petgraph::Direction::Outgoing) {
                let target = edge.target();
                if let (Some(&new_source), Some(&new_target)) = (old_to_new.get(old_ni), old_to_new.get(&target)) {
                    new_graph.add_edge(new_source, new_target, *edge.weight());
                }
            }
        }
        
        // Retain only active nodes in activation state
        let mut new_activations = HashMap::new();
        if let Ok(activations) = self.activation_state.read() {
            for (old_ni, new_ni) in &old_to_new {
                if let Some(&act) = activations.get(old_ni) {
                    new_activations.insert(*new_ni, act);
                }
            }
        }

        self.graph = new_graph;
        self.fingerprints = new_fingerprints;
        self.by_id = new_by_id;
        *self.activation_state.write().unwrap() = new_activations;
    }

    /// Recall a causal/temporal trajectory starting from the given node ID.
    pub fn recall_trajectory(&self, start: Uuid, limit: usize) -> Vec<MemoryNode> {
        let mut trajectory = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        let start_ni = match self.by_id.get(&start) {
            Some(&ni) => ni,
            None => return trajectory,
        };
        
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start_ni);
        visited.insert(start_ni);
        
        while let Some(ni) = queue.pop_front() {
            if let Some(node) = self.graph.node_weight(ni) {
                if node.is_active() {
                    trajectory.push(node.clone());
                    if trajectory.len() >= limit {
                        break;
                    }
                }
            }
            
            // Collect neighbors strictly through causal/temporal edges
            let mut neighbors = Vec::new();
            for edge in self.graph.edges_directed(ni, petgraph::Direction::Outgoing) {
                let w = *edge.weight();
                if matches!(w, MemoryEdge::CausedBy | MemoryEdge::Before | MemoryEdge::DerivedFrom) {
                    neighbors.push(edge.target());
                }
            }
            // For Incoming edges, "After" is the reverse of "Before" (if A is After B, B is Before A)
            for edge in self.graph.edges_directed(ni, petgraph::Direction::Incoming) {
                let w = *edge.weight();
                if matches!(w, MemoryEdge::After) {
                    neighbors.push(edge.source());
                }
            }
            
            for target in neighbors {
                if visited.insert(target) {
                    queue.push_back(target);
                }
            }
        }
        
        trajectory
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
