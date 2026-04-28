//! In-memory graph of MemoryNodes connected by typed MemoryEdges.
//!
//! Uses the same PPR algorithm as codegraph-core's KnowledgeGraph but
//! applied to agent memories (facts, entities, experiences) instead of code symbols.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A single unit of memory stored in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Stable unique identifier (also used as the SQLite primary key)
    pub id: Uuid,
    /// The raw text of the memory / fact
    pub text: String,
    /// Free-form tags for filtering
    pub tags: Vec<String>,
    /// User-provided or graph-derived importance (0.0 – 1.0)
    pub importance: f32,
    /// ISO-8601 timestamp of when this memory was stored
    pub created_at: String,
    /// Pre-computed token count (tiktoken cl100k_base)
    pub token_count: usize,
}

/// Typed relationship between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryEdge {
    /// General semantic relationship
    RelatesTo,
    /// One memory contradicts another
    Contradicts,
    /// One memory provides supporting evidence for another
    Supports,
    /// One memory was derived / inferred from another
    DerivedFrom,
}

impl std::fmt::Display for MemoryEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryEdge::RelatesTo => write!(f, "relates_to"),
            MemoryEdge::Contradicts => write!(f, "contradicts"),
            MemoryEdge::Supports => write!(f, "supports"),
            MemoryEdge::DerivedFrom => write!(f, "derived_from"),
        }
    }
}

/// The in-memory graph holding all memories and their relationships.
pub struct MemoryGraph {
    graph: DiGraph<MemoryNode, MemoryEdge>,
    /// Map from Uuid → NodeIndex for O(1) lookup
    id_to_index: HashMap<Uuid, NodeIndex>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_index: HashMap::new(),
        }
    }

    /// Insert a new memory node and return its NodeIndex.
    pub fn add_node(&mut self, node: MemoryNode) -> NodeIndex {
        let id = node.id;
        let idx = self.graph.add_node(node);
        self.id_to_index.insert(id, idx);
        idx
    }

    /// Add a directed edge between two memories (by Uuid).
    /// Silently ignores if either node is not found.
    pub fn add_edge(&mut self, from: Uuid, to: Uuid, kind: MemoryEdge) {
        if let (Some(&a), Some(&b)) = (self.id_to_index.get(&from), self.id_to_index.get(&to)) {
            self.graph.add_edge(a, b, kind);
        }
    }

    /// Remove a memory by Uuid. Returns true if the node existed.
    pub fn remove_node(&mut self, id: &Uuid) -> bool {
        if let Some(&idx) = self.id_to_index.get(id) {
            self.graph.remove_node(idx);
            self.id_to_index.remove(id);
            true
        } else {
            false
        }
    }

    /// Lookup a node by Uuid.
    pub fn get_node(&self, id: &Uuid) -> Option<&MemoryNode> {
        self.id_to_index
            .get(id)
            .and_then(|&idx| self.graph.node_weight(idx))
    }

    /// All memory nodes in insertion order.
    pub fn all_nodes(&self) -> Vec<&MemoryNode> {
        self.graph.node_weights().collect()
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    // ─── Personalized PageRank ────────────────────────────────────────────────

    /// Compute Personalized PageRank scores starting from `seed_ids`.
    ///
    /// Same algorithm as codegraph-core:
    ///   - 20 power-iteration steps, damping 0.85
    ///   - Bidirectional traversal: forward edges at full weight, reverse at 0.5×
    ///   - Dangling node mass redistributed to seeds
    ///
    /// Returns a map from NodeIndex → PPR score (higher = more related to seeds).
    pub fn personalized_pagerank(&self, seed_ids: &[Uuid]) -> HashMap<NodeIndex, f64> {
        let n = self.graph.node_count();
        if n == 0 {
            return HashMap::new();
        }

        // Build seed set
        let seeds: Vec<NodeIndex> = seed_ids
            .iter()
            .filter_map(|id| self.id_to_index.get(id).copied())
            .collect();

        if seeds.is_empty() {
            return HashMap::new();
        }

        let seed_weight = 1.0 / seeds.len() as f64;
        let damping = 0.85_f64;
        let all_indices: Vec<NodeIndex> = self.graph.node_indices().collect();

        // Initial distribution: uniform over seeds
        let mut scores: HashMap<NodeIndex, f64> = all_indices
            .iter()
            .map(|&idx| {
                let v = if seeds.contains(&idx) { seed_weight } else { 0.0 };
                (idx, v)
            })
            .collect();

        for _ in 0..20 {
            let mut new_scores: HashMap<NodeIndex, f64> =
                all_indices.iter().map(|&idx| (idx, 0.0)).collect();

            let mut dangling_mass = 0.0_f64;

            for &idx in &all_indices {
                let out_edges: Vec<_> = self.graph.edges(idx).collect();
                let rev_edges: Vec<_> = self
                    .graph
                    .edges_directed(idx, petgraph::Direction::Incoming)
                    .collect();

                let total_out = out_edges.len() as f64 + rev_edges.len() as f64 * 0.5;

                if total_out == 0.0 {
                    dangling_mass += scores[&idx];
                } else {
                    let s = scores[&idx];
                    // Forward edges (full weight)
                    for edge in &out_edges {
                        *new_scores.entry(edge.target()).or_default() +=
                            s * (1.0 / total_out);
                    }
                    // Reverse edges (0.5× weight — bidirectional awareness)
                    for edge in &rev_edges {
                        *new_scores.entry(edge.source()).or_default() +=
                            s * (0.5 / total_out);
                    }
                }
            }

            // Redistribute dangling mass to seeds
            let dangling_per_seed = dangling_mass / seeds.len() as f64;
            for &seed in &seeds {
                *new_scores.entry(seed).or_default() += dangling_per_seed;
            }

            // Apply damping: (1 - d) * teleport + d * propagated
            let teleport_weight = (1.0 - damping) / seeds.len() as f64;
            for &idx in &all_indices {
                let is_seed = seeds.contains(&idx);
                let teleport = if is_seed { teleport_weight } else { 0.0 };
                *new_scores.entry(idx).or_default() =
                    teleport + damping * new_scores[&idx];
            }

            scores = new_scores;
        }

        scores
    }

    /// Find memories that have a `Contradicts` edge to or from any of `node_ids`.
    pub fn find_contradictions(&self, node_ids: &[NodeIndex]) -> Vec<(&MemoryNode, &MemoryNode)> {
        let mut pairs = Vec::new();
        let node_set: std::collections::HashSet<NodeIndex> = node_ids.iter().copied().collect();

        for edge in self.graph.edge_references() {
            if *edge.weight() == MemoryEdge::Contradicts {
                let (src, tgt) = (edge.source(), edge.target());
                if node_set.contains(&src) || node_set.contains(&tgt) {
                    if let (Some(a), Some(b)) = (
                        self.graph.node_weight(src),
                        self.graph.node_weight(tgt),
                    ) {
                        pairs.push((a, b));
                    }
                }
            }
        }
        pairs
    }

    /// Get the NodeIndex for a Uuid (for PPR input).
    pub fn index_of(&self, id: &Uuid) -> Option<NodeIndex> {
        self.id_to_index.get(id).copied()
    }

    /// All NodeIndices (for PPR seed building).
    pub fn all_indices(&self) -> Vec<NodeIndex> {
        self.graph.node_indices().collect()
    }

    /// Top-N nodes by a score map (PPR output).
    pub fn top_by_score<'a>(
        &'a self,
        scores: &HashMap<NodeIndex, f64>,
        limit: usize,
    ) -> Vec<(&'a MemoryNode, f64)> {
        let mut ranked: Vec<(NodeIndex, f64)> = scores
            .iter()
            .map(|(&idx, &s)| (idx, s))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(limit)
            .filter_map(|(idx, score)| {
                self.graph.node_weight(idx).map(|n| (n, score))
            })
            .collect()
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(text: &str) -> MemoryNode {
        MemoryNode {
            id: Uuid::new_v4(),
            text: text.to_string(),
            tags: vec![],
            importance: 0.5,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            token_count: text.split_whitespace().count(),
        }
    }

    #[test]
    fn test_add_and_remove() {
        let mut g = MemoryGraph::new();
        let n = make_node("The auth module uses JWT");
        let id = n.id;
        g.add_node(n);
        assert_eq!(g.node_count(), 1);
        assert!(g.remove_node(&id));
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_ppr_seeds_score_highest() {
        let mut g = MemoryGraph::new();
        let n1 = make_node("JWT authentication is used");
        let n2 = make_node("Database uses PostgreSQL");
        let n3 = make_node("Auth module calls the DB");
        let id1 = n1.id;
        let id3 = n3.id;
        g.add_node(n1);
        g.add_node(n2);
        g.add_node(n3);
        g.add_edge(id3, id1, MemoryEdge::RelatesTo);

        let scores = g.personalized_pagerank(&[id1]);
        // Seed should have a high score
        let idx1 = g.index_of(&id1).unwrap();
        let idx3 = g.index_of(&id3).unwrap();
        assert!(scores[&idx1] > scores[&g.index_of(&n2.id).unwrap_or(idx3)]);
    }

    #[test]
    fn test_contradictions() {
        let mut g = MemoryGraph::new();
        let n1 = make_node("The service is stateless");
        let n2 = make_node("The service stores sessions in memory");
        let id1 = n1.id;
        let id2 = n2.id;
        let idx1 = g.add_node(n1);
        g.add_node(n2);
        g.add_edge(id1, id2, MemoryEdge::Contradicts);

        let contradictions = g.find_contradictions(&[idx1]);
        assert_eq!(contradictions.len(), 1);
    }
}
