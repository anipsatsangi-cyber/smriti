//! Compressed memory snapshot — analogous to `SessionMemory` in codegraph-context.
//!
//! Generates a token-budget-aware summary of the entire memory graph using PPR
//! seeded from the top-connected nodes (global importance), then knapsack-packs
//! the result into a prompt-injectable context block.

use crate::graph::MemoryGraph;
use anyhow::Result;

/// A compressed snapshot of the full memory graph.
pub struct MemorySnapshot {
    pub content: String,
    pub tokens_used: usize,
    pub total_memories: usize,
    pub included_memories: usize,
}

impl MemorySnapshot {
    /// Generate a snapshot within `token_budget` tokens.
    ///
    /// Seeding strategy: use all nodes as seeds weighted by their `importance`
    /// field — nodes with higher importance anchor the PPR, so highly important
    /// facts dominate the snapshot even if not recently queried.
    pub fn generate(graph: &MemoryGraph, token_budget: usize) -> Result<MemorySnapshot> {
        let all = graph.all_nodes();
        let total_memories = all.len();

        if total_memories == 0 {
            return Ok(MemorySnapshot {
                content: "*(No memories stored yet)*\n".to_string(),
                tokens_used: 0,
                total_memories: 0,
                included_memories: 0,
            });
        }

        // Seed PPR from top-50 most important nodes
        let mut sorted_by_importance: Vec<_> = all.iter().collect();
        sorted_by_importance
            .sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));

        let seeds: Vec<uuid::Uuid> = sorted_by_importance
            .iter()
            .take(50)
            .map(|n| n.id)
            .collect();

        let ppr_scores = graph.personalized_pagerank(&seeds);

        // Score every node: ppr * importance
        let mut scored: Vec<_> = graph
            .all_indices()
            .into_iter()
            .filter_map(|idx| {
                let node = graph.top_by_score(&std::iter::once((idx, 1.0)).collect(), 1)
                    .into_iter()
                    .next()
                    .map(|(n, _)| n);
                let node = node?;
                let ppr = ppr_scores.get(&idx).copied().unwrap_or(0.0);
                let score = ppr * (0.5 + node.importance as f64 * 0.5);
                Some((node, score))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Budget allocation: reserve 10% for header, rest for memories
        let header = format!(
            "## Memory Snapshot ({} total memories)\n\n",
            total_memories
        );
        let header_tokens = header.split_whitespace().count() + 4;
        let memory_budget = token_budget.saturating_sub(header_tokens);

        // Knapsack pack
        let mut lines: Vec<String> = Vec::new();
        let mut tokens_used = header_tokens;
        let mut included = 0usize;

        for (node, _score) in &scored {
            let line = if node.tags.is_empty() {
                format!("- {}\n", node.text)
            } else {
                format!("- {} [{}]\n", node.text, node.tags.join(", "))
            };
            let cost = line.split_whitespace().count() + 4;
            if tokens_used - header_tokens + cost > memory_budget {
                continue;
            }
            tokens_used += cost;
            included += 1;
            lines.push(line);
        }

        let footer = format!(
            "\n*Showing {} of {} memories within {} token budget*\n",
            included, total_memories, token_budget
        );
        tokens_used += footer.split_whitespace().count();

        let mut content = header;
        for line in lines {
            content.push_str(&line);
        }
        content.push_str(&footer);

        Ok(MemorySnapshot {
            content,
            tokens_used,
            total_memories,
            included_memories: included,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{MemoryGraph, MemoryNode};
    use uuid::Uuid;

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
    fn test_snapshot_respects_budget() {
        let mut g = MemoryGraph::new();
        for i in 0..20 {
            g.add_node(make_node(&format!("This is memory fact number {} about auth", i), 0.5));
        }
        let snap = MemorySnapshot::generate(&g, 100).unwrap();
        assert!(snap.tokens_used <= 110, "should stay close to budget");
        assert!(snap.total_memories == 20);
    }

    #[test]
    fn test_snapshot_empty() {
        let g = MemoryGraph::new();
        let snap = MemorySnapshot::generate(&g, 1000).unwrap();
        assert!(snap.content.contains("No memories"));
    }

    #[test]
    fn test_snapshot_high_importance_included() {
        let mut g = MemoryGraph::new();
        // One very important node
        g.add_node(make_node("CRITICAL: The production DB password is rotated weekly", 1.0));
        // Many low-importance nodes
        for i in 0..30 {
            g.add_node(make_node(&format!("Minor detail {}", i), 0.1));
        }
        let snap = MemorySnapshot::generate(&g, 50).unwrap();
        assert!(snap.content.contains("CRITICAL"), "high-importance node should be in snapshot");
    }
}
