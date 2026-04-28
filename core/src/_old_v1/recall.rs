//! Token-budget-aware PPR recall.
//!
//! Pipeline:
//!   1. Tantivy keyword search → candidate Uuid list
//!   2. PPR from candidate seeds over the entity graph
//!   3. Knapsack pack: include highest-scoring nodes until token budget exhausted
//!   4. Return formatted context string

use crate::graph::{MemoryGraph, MemoryNode};
use crate::index::MemoryIndex;
use anyhow::Result;
use uuid::Uuid;

/// Result of a recall query.
pub struct RecallResult {
    /// Selected memories in relevance order (highest first)
    pub memories: Vec<MemoryNode>,
    /// Total tokens used
    pub tokens_used: usize,
    /// Tokens available (budget)
    pub token_budget: usize,
    /// Number of candidates found before packing
    pub candidates_found: usize,
}

impl RecallResult {
    /// Format as a compact context string for injection into a prompt.
    pub fn format_context(&self) -> String {
        let mut out = String::from("## Relevant Memories\n\n");
        for (i, mem) in self.memories.iter().enumerate() {
            out.push_str(&format!("{}. {}", i + 1, mem.text));
            if !mem.tags.is_empty() {
                out.push_str(&format!(" [{}]", mem.tags.join(", ")));
            }
            out.push('\n');
        }
        out.push_str(&format!(
            "\n*{} memories, {} / {} tokens*\n",
            self.memories.len(),
            self.tokens_used,
            self.token_budget
        ));
        out
    }
}

/// Perform token-budget-aware PPR recall.
///
/// # Arguments
/// * `graph` — in-memory MemoryGraph
/// * `index` — Tantivy index for keyword search
/// * `query` — natural-language query string
/// * `token_budget` — maximum tokens to return
/// * `search_limit` — how many candidates to retrieve from tantivy (default 20)
pub fn recall(
    graph: &MemoryGraph,
    index: &MemoryIndex,
    query: &str,
    token_budget: usize,
    search_limit: usize,
) -> Result<RecallResult> {
    // ── Step 1: keyword search for candidate seeds ─────────────────────────
    let candidate_ids: Vec<Uuid> = index.search(query, search_limit)?;
    let candidates_found = candidate_ids.len();

    if candidate_ids.is_empty() {
        return Ok(RecallResult {
            memories: vec![],
            tokens_used: 0,
            token_budget,
            candidates_found: 0,
        });
    }

    // ── Step 2: PPR from search results as seeds ───────────────────────────
    let ppr_scores = graph.personalized_pagerank(&candidate_ids);

    // ── Step 3: score each node = text_relevance * (1 + ppr * 10) ─────────
    // Nodes found by search get a base relevance of 1.0; others get 0.3
    let candidate_set: std::collections::HashSet<&Uuid> = candidate_ids.iter().collect();

    let mut scored: Vec<(&MemoryNode, f64)> = graph
        .all_nodes()
        .into_iter()
        .filter_map(|node| {
            let idx = graph.index_of(&node.id)?;
            let ppr = ppr_scores.get(&idx).copied().unwrap_or(0.0);
            let base = if candidate_set.contains(&node.id) { 1.0 } else { 0.3 };
            // Boost by user-set importance
            let score = base * (1.0 + ppr * 10.0) * (0.5 + node.importance as f64 * 0.5);
            if score > 0.0 {
                Some((node, score))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 4: knapsack pack within token budget ──────────────────────────
    let mut selected = Vec::new();
    let mut tokens_used = 0usize;

    for (node, _score) in &scored {
        let cost = node.token_count + 8; // +8 overhead for numbering + tags
        if tokens_used + cost > token_budget {
            continue; // skip if doesn't fit; keep trying smaller items
        }
        tokens_used += cost;
        selected.push((*node).clone());
    }

    Ok(RecallResult {
        memories: selected,
        tokens_used,
        token_budget,
        candidates_found,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MemoryEdge, MemoryNode};
    use crate::index::MemoryIndex;

    fn make_node(text: &str, importance: f32) -> MemoryNode {
        MemoryNode {
            id: Uuid::new_v4(),
            text: text.to_string(),
            tags: vec![],
            importance,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            token_count: text.split_whitespace().count(),
        }
    }

    #[test]
    fn test_recall_respects_budget() {
        let mut graph = MemoryGraph::new();
        let mut index = MemoryIndex::open_in_ram().unwrap();

        for i in 0..10 {
            let n = make_node(&format!("JWT authentication fact number {}", i), 0.5);
            index.add(&n.id, &n.text, &n.tags).unwrap();
            graph.add_node(n);
        }
        index.commit().unwrap();

        let result = recall(&graph, &index, "JWT authentication", 50, 20).unwrap();
        assert!(result.tokens_used <= 50);
        assert!(result.candidates_found > 0);
    }

    #[test]
    fn test_recall_empty_query() {
        let graph = MemoryGraph::new();
        let index = MemoryIndex::open_in_ram().unwrap();
        let result = recall(&graph, &index, "something not indexed", 1000, 10).unwrap();
        assert!(result.memories.is_empty());
    }

    #[test]
    fn test_ppr_expansion_finds_related() {
        let mut graph = MemoryGraph::new();
        let mut index = MemoryIndex::open_in_ram().unwrap();

        // n1 is directly searchable; n2 is related via graph but not in search results
        let n1 = make_node("JWT authentication is used for API access", 0.9);
        let n2 = make_node("The API gateway validates tokens before routing", 0.8);
        let id1 = n1.id;
        let id2 = n2.id;

        index.add(&n1.id, &n1.text, &n1.tags).unwrap();
        // n2 is NOT in the search index — only reachable via graph
        index.commit().unwrap();

        graph.add_node(n1);
        graph.add_node(n2);
        graph.add_edge(id2, id1, MemoryEdge::RelatesTo);

        let result = recall(&graph, &index, "JWT", 2000, 10).unwrap();
        // n2 should appear because PPR expands from n1 through the graph edge
        let found_n2 = result.memories.iter().any(|m| m.id == id2);
        assert!(found_n2, "PPR should surface n2 via graph expansion from n1");
    }
}
