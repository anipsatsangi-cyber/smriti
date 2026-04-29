//! Consolidation — moving memories from Hippocampus to Neocortex.
//!
//! Inspired by the hippocampal replay phenomenon (Buzsáki 1989, Wilson &
//! McNaughton 1994): during sleep / quiet wakefulness, the hippocampus
//! "replays" recent experiences to the neocortex, integrating them into
//! the long-term semantic graph.
//!
//! In Smriti, consolidation runs:
//! - Automatically when the hippocampus crosses a high-water mark.
//! - On explicit `consolidate()` calls.
//! - Periodically (configurable).
//!
//! For each candidate hippocampal entry, we:
//! 1. Run the predictive-coding redundancy check.
//! 2. If novel → promote to the neocortex; auto-link to similar nodes
//!    (tag overlap → `RelatesTo`).
//! 3. If reinforces → bump the existing node's access count and merge
//!    new tags; drop the hippocampal entry.
//! 4. If redundant → drop the hippocampal entry.

use std::collections::HashSet;

use crate::core::hippocampus::{EpisodicEntry, Hippocampus};
use crate::core::neocortex::Neocortex;
use crate::core::prediction::{predict, PredictionVerdict};
use crate::node::{MemoryEdge, MemoryNode};

/// Minimum number of shared tags to auto-link two memories with `RelatesTo`.
const TAG_LINK_THRESHOLD: usize = 2;

/// Stats from a consolidation pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidationReport {
    pub processed: usize,
    pub promoted: usize,
    pub reinforced: usize,
    pub dropped: usize,
    pub edges_created: usize,
}

/// Consolidate up to `max` entries from the hippocampus into the neocortex.
///
/// Pass `max = usize::MAX` to consolidate everything currently in the
/// buffer.
pub fn consolidate(
    hippo: &mut Hippocampus,
    neo: &mut Neocortex,
    max: usize,
) -> ConsolidationReport {
    let mut report = ConsolidationReport::default();

    // Collect entries oldest-first into a vec we can drain.
    let candidates: Vec<EpisodicEntry> = hippo
        .drain_where(|_| true) // drain all
        .into_iter()
        .take(max)
        .collect();

    for mut entry in candidates {
        report.processed += 1;
        let scope = entry.node.scope.clone();

        // Memories that explicitly supersede an older one always promote
        // as Novel — they ARE the replacement, even if their text looks
        // similar to the original. Skipping the redundancy filter here
        // is critical: otherwise the predict() check sees the old
        // memory as a near-duplicate and merges/drops the new one,
        // hiding the corrected fact from recall.
        let verdict = if entry.node.supersedes.is_some() {
            PredictionVerdict::Novel { max_similarity: 1.0 } // Don't trigger surprise for supersedes
        } else {
            predict(&entry.node, neo, &scope)
        };

        match verdict {
            PredictionVerdict::Novel { max_similarity } => {
                // Predictive Coding / Surprise — Friston-style "this memory
                // is so unlike anything I've seen, I should learn from it."
                //
                // Tuning rules:
                //
                //   1. Threshold: previously 0.2, now 0.05. HDC fingerprint
                //      similarity in a diverse 500-memory corpus typically
                //      bottoms out around 0.10-0.15 — the old 0.2 threshold
                //      caused most newly-consolidated memories in such a
                //      corpus to be auto-promoted to Critical, polluting
                //      the PPR seed pool (Critical nodes are fast-tracked
                //      as seeds) and flattening the score landscape on
                //      paraphrase queries. 0.05 only fires on memories
                //      that are essentially orthogonal to the existing
                //      graph — the genuine "I've never seen this" signal.
                //
                //   2. Minimum corpus size: don't fire on near-empty
                //      neocortex. With 3 memories, everything looks
                //      surprising; surprise only conveys information once
                //      we have enough context to predict from.
                const SURPRISE_THRESHOLD: f32 = 0.05;
                const SURPRISE_MIN_CORPUS: usize = 30;
                if max_similarity < SURPRISE_THRESHOLD && neo.len() >= SURPRISE_MIN_CORPUS {
                    entry.node.salience = crate::node::Salience::Critical;
                    entry.node.importance = 1.0;
                }
                
                let new_id = entry.node.id;
                let new_tags: HashSet<String> = entry.node.tags.iter().cloned().collect();
                neo.insert(entry.node);
                report.promoted += 1;

                // Auto-link to existing memories with ≥ TAG_LINK_THRESHOLD shared tags.
                let mut to_link: Vec<uuid::Uuid> = Vec::new();
                for (existing, _) in neo.iter_active() {
                    if existing.id == new_id {
                        continue;
                    }
                    if !scope.can_read(&existing.scope) {
                        continue;
                    }
                    let shared = existing
                        .tags
                        .iter()
                        .filter(|t| new_tags.contains(*t))
                        .count();
                    if shared >= TAG_LINK_THRESHOLD {
                        to_link.push(existing.id);
                    }
                }
                // Collect temporal edge candidates: Event memories sharing ≥1 tag
                // get Before/After edges based on created_at ordering.
                let mut temporal_targets: Vec<(uuid::Uuid, chrono::DateTime<chrono::Utc>)> =
                    Vec::new();
                let new_node = neo.get(new_id);
                let new_is_event = new_node
                    .map(|n| matches!(n.kind, crate::node::MemoryKind::Event))
                    .unwrap_or(false);
                let new_created = new_node
                    .map(|n| n.created_at)
                    .unwrap_or_else(chrono::Utc::now);

                for other in &to_link {
                    if let Some(existing) = neo.get(*other) {
                        if new_is_event && matches!(existing.kind, crate::node::MemoryKind::Event) {
                            temporal_targets.push((existing.id, existing.created_at));
                        }
                    }
                }

                for other in to_link {
                    neo.link(new_id, other, MemoryEdge::RelatesTo);
                    neo.link(other, new_id, MemoryEdge::RelatesTo);
                    report.edges_created += 2;
                }

                // Auto-create temporal edges for events
                for (other_id, other_created) in temporal_targets {
                    if new_created < other_created {
                        neo.link(new_id, other_id, MemoryEdge::Before);
                        neo.link(other_id, new_id, MemoryEdge::After);
                    } else if new_created > other_created {
                        neo.link(new_id, other_id, MemoryEdge::After);
                        neo.link(other_id, new_id, MemoryEdge::Before);
                    }
                    report.edges_created += 2;
                }
            }
            PredictionVerdict::Reinforces { existing, .. } => {
                if let Some(node) = neo.get_mut(existing) {
                    reinforce(node, &entry.node);
                }
                report.reinforced += 1;
            }
            PredictionVerdict::Redundant { existing, .. } => {
                if let Some(node) = neo.get_mut(existing) {
                    node.access_count = node.access_count.saturating_add(1);
                    node.last_accessed_at = chrono::Utc::now();
                }
                report.dropped += 1;
            }
        }
    }

    report
}

/// Merge information from a hippocampal entry into an existing neocortex node.
fn reinforce(target: &mut MemoryNode, source: &MemoryNode) {
    // Bump access count and refresh the last-accessed timestamp.
    target.access_count = target.access_count.saturating_add(1);
    target.last_accessed_at = chrono::Utc::now();

    // Merge tags (preserve order, dedupe).
    for tag in &source.tags {
        if !target.tags.contains(tag) {
            target.tags.push(tag.clone());
        }
    }

    // Bump version for sync compatibility.
    target.version = target.version.saturating_add(1);

    // Importance: take the max so reinforcement can't lower it.
    if source.importance > target.importance {
        target.importance = source.importance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{MemoryKind, MemoryNode};
    use crate::scope::Scope;

    fn mk(text: &str, tags: &[&str], kind: MemoryKind) -> MemoryNode {
        let mut n = MemoryNode::new(text, kind, Scope::default());
        n.tags = tags.iter().map(|s| s.to_string()).collect();
        n
    }

    #[test]
    fn novel_memory_promotes_to_neocortex() {
        let mut hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        hippo.insert(mk("brand new fact", &["unique"], MemoryKind::Fact));

        let report = consolidate(&mut hippo, &mut neo, usize::MAX);
        assert_eq!(report.promoted, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(neo.len(), 1);
    }

    #[test]
    fn auto_link_creates_edges_for_shared_tags() {
        let mut hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        // Insert first memory directly into neo
        neo.insert(mk(
            "Service uses Postgres",
            &["db", "infra", "backend"],
            MemoryKind::Fact,
        ));
        // Push second memory (with overlapping tags) through hippocampus
        hippo.insert(mk(
            "Postgres tuning best practices",
            &["db", "infra", "performance"],
            MemoryKind::Fact,
        ));

        let report = consolidate(&mut hippo, &mut neo, usize::MAX);
        assert_eq!(report.promoted, 1);
        // ≥ 2 shared tags ("db", "infra") → bidirectional edges
        assert!(report.edges_created >= 2);
        assert!(neo.edge_count() >= 2);
    }

    #[test]
    fn duplicate_memory_is_dropped() {
        let mut hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        neo.insert(mk(
            "the auth module uses JWT RS256",
            &["auth"],
            MemoryKind::Fact,
        ));
        hippo.insert(mk(
            "the auth module uses JWT RS256",
            &["auth"],
            MemoryKind::Fact,
        ));

        let report = consolidate(&mut hippo, &mut neo, usize::MAX);
        // Should be dropped (or at worst reinforced) — but NOT promoted.
        assert_eq!(report.promoted, 0);
        assert_eq!(neo.len(), 1);
    }
}
