//! Predictive coding — MDL-based redundancy filtering.
//!
//! This module decides whether a new memory adds genuine information to
//! the store, or whether it's already implied by what we have. We follow
//! the Minimum Description Length principle (Rissanen 1978, Grünwald
//! 2007): the optimal store is the one that minimizes
//!
//! ```text
//! L(store) + L(input | store)
//! ```
//!
//! where `L(.)` is description length in bits.
//!
//! In practice we approximate this with a similarity-based test:
//! - Compute the HDC fingerprint of the new memory.
//! - Find the most similar existing memory in the neocortex.
//! - If similarity > REDUNDANT_THRESHOLD → reject as redundant.
//! - If similarity > REINFORCE_THRESHOLD → reinforce existing memory
//!   (bump access count, optionally append to tags).
//! - Otherwise → store as new.
//!
//! This is an approximation of Friston's predictive coding: we only
//! store the **prediction error** — the part of the new input that
//! the existing store didn't already encode.

use crate::core::hdc::{fingerprint, Hypervector};
use crate::core::neocortex::Neocortex;
use crate::node::MemoryNode;
use crate::scope::Scope;
use uuid::Uuid;

/// Above this similarity, a new memory is considered redundant and rejected.
/// Tuned empirically: 0.85 is roughly "near-duplicate text with rephrasing."
pub const REDUNDANT_THRESHOLD: f32 = 0.85;

/// Above this (but below redundant), the new memory is similar enough to
/// reinforce an existing one rather than create a new node.
pub const REINFORCE_THRESHOLD: f32 = 0.65;

/// Outcome of the redundancy check.
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionVerdict {
    /// The memory is genuinely new. Store it. Includes the highest similarity found for Surprise calculation.
    Novel { max_similarity: f32 },
    /// The memory reinforces an existing one. Bump that one's access
    /// counter and merge tags.
    Reinforces { existing: Uuid, similarity: f32 },
    /// The memory is fully redundant. Drop it.
    Redundant { existing: Uuid, similarity: f32 },
}

impl PredictionVerdict {
    pub fn is_novel(&self) -> bool {
        matches!(self, PredictionVerdict::Novel { .. })
    }
}

/// Check whether a candidate memory adds genuine information given the
/// current neocortex.
///
/// `reader_scope` constrains which existing memories are considered when
/// looking for redundancy: a memory in user A's scope is never
/// considered redundant by a memory in user B's scope.
pub fn predict(candidate: &MemoryNode, nx: &Neocortex, reader_scope: &Scope) -> PredictionVerdict {
    if nx.is_empty() {
        return PredictionVerdict::Novel { max_similarity: 0.0 };
    }

    let q = fingerprint(&candidate.text, &candidate.tags);
    let near = nx.nearest_by_fingerprint(&q, 1, -1.0, Some(reader_scope));

    match near.first() {
        Some(&(id, sim)) if sim >= REDUNDANT_THRESHOLD => PredictionVerdict::Redundant {
            existing: id,
            similarity: sim,
        },
        Some(&(id, sim)) if sim >= REINFORCE_THRESHOLD => PredictionVerdict::Reinforces {
            existing: id,
            similarity: sim,
        },
        Some(&(_, sim)) => PredictionVerdict::Novel { max_similarity: sim },
        None => PredictionVerdict::Novel { max_similarity: 0.0 },
    }
}

/// Estimate the information-theoretic cost of storing a memory, in bits.
///
/// We use the text length as a proxy for entropy: a longer memory carries
/// more information and is worth more to keep. This gives us a way to
/// compare "is the existing memory more or less informative than the new
/// one?" without actually computing entropy.
pub fn description_length(node: &MemoryNode) -> f32 {
    // Bits = bytes * 8, but we discount because text is highly compressible.
    // Use log of length to emulate entropy.
    let len = node.text.len() as f32 + node.tags.iter().map(|t| t.len() as f32).sum::<f32>();
    (len + 1.0).log2() * 8.0
}

/// Helper used by HDC fingerprint comparison without re-importing.
pub fn fingerprint_of(node: &MemoryNode) -> Hypervector {
    fingerprint(&node.text, &node.tags)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;

    fn mk(text: &str) -> MemoryNode {
        MemoryNode::new(text, MemoryKind::Fact, Scope::default())
    }

    #[test]
    fn empty_neocortex_accepts_everything() {
        let nx = Neocortex::new();
        let cand = mk("anything");
        assert!(predict(&cand, &nx, &Scope::default()).is_novel());
    }

    #[test]
    fn near_duplicate_is_redundant() {
        let mut nx = Neocortex::new();
        let original = mk("the auth module uses JWT RS256 tokens");
        nx.insert(original);

        let dup = mk("the auth module uses JWT RS256 tokens"); // identical
        let v = predict(&dup, &nx, &Scope::default());
        match v {
            PredictionVerdict::Redundant { similarity, .. } => {
                assert!(similarity > REDUNDANT_THRESHOLD);
            }
            other => panic!("expected Redundant, got {:?}", other),
        }
    }

    #[test]
    fn unrelated_memory_is_novel() {
        let mut nx = Neocortex::new();
        nx.insert(mk("the auth module uses JWT RS256 tokens"));

        let novel = mk("user prefers spicy food");
        assert!(predict(&novel, &nx, &Scope::default()).is_novel());
    }
}
