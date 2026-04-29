//! Forgetting curves — Ebbinghaus-style decay applied at recall time.
//!
//! Memory traces are not deleted as they age; instead, their effective
//! importance decays exponentially. We use a per-kind half-life and a
//! Hebbian boost: memories that are accessed often have their effective
//! age reset.
//!
//! # Decay function
//!
//! ```text
//! effective_importance =
//!     base_importance
//!   * 2^(-effective_age / half_life)
//!   * (1 + log(1 + access_count) * HEBB_GAIN)
//! ```
//!
//! Where:
//! - `effective_age` = `idle_days` for hebbian kinds, `age_days` otherwise.
//! - `HEBB_GAIN` = 0.1 (10% bump per log-access).
//!
//! # References
//!
//! - Ebbinghaus (1885), *Über das Gedächtnis*
//! - Murre & Dros (2015), *Replication and Analysis of Ebbinghaus's
//!   Forgetting Curve*

use crate::node::{MemoryKind, MemoryNode};

/// Per-log-access importance multiplier (Hebbian reinforcement).
const HEBB_GAIN: f32 = 0.1;

/// Compute the effective importance of a memory at the current moment.
///
/// Combines the base importance, the kind-specific decay curve, and the
/// access-count boost. Always returns a value in `[0, ~importance * 2]`.
pub fn effective_importance(node: &MemoryNode) -> f32 {
    if node.salience == crate::node::Salience::Critical {
        // Critical memories bypass decay and get a massive additive boost
        return node.importance + 1.0;
    }

    let half_life = match node.salience {
        crate::node::Salience::Important => node.kind.half_life_days() * 3.0,
        _ => node.kind.half_life_days(),
    };

    // For Hebbian kinds (Decision, Preference), age starts from last
    // access — frequent access keeps the memory "fresh."
    let age = if node.kind.hebbian() {
        node.idle_days()
    } else {
        node.age_days()
    };

    let decay = 2.0_f64.powf(-age / half_life) as f32;

    let access_boost = 1.0 + (1.0 + node.access_count as f32).ln() * HEBB_GAIN;

    node.importance * decay * access_boost
}

/// Whether a memory has decayed below the eviction threshold.
///
/// The hippocampus uses this to evict stale episodic traces.
pub fn should_evict(node: &MemoryNode, threshold: f32) -> bool {
    effective_importance(node) < threshold
}

/// Decay rate at the present moment — useful for sorting recall candidates
/// by "freshness."
pub fn decay_factor(kind: MemoryKind, age_days: f64) -> f32 {
    let half_life = kind.half_life_days();
    2.0_f64.powf(-age_days / half_life) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::Scope;
    use chrono::{Duration, Utc};

    fn aged(kind: MemoryKind, days: i64) -> MemoryNode {
        let mut n = MemoryNode::new("test", kind, Scope::default());
        n.created_at = Utc::now() - Duration::days(days);
        n.last_accessed_at = n.created_at;
        n.importance = 1.0;
        n
    }

    #[test]
    fn decay_at_half_life() {
        let n = aged(MemoryKind::Fact, 90);
        let imp = effective_importance(&n);
        // After 1 half-life: ~0.5 (modulo small access boost)
        assert!(
            (imp - 0.5).abs() < 0.05,
            "importance after half-life = {}",
            imp
        );
    }

    #[test]
    fn fresh_memory_full_strength() {
        let n = aged(MemoryKind::Fact, 0);
        let imp = effective_importance(&n);
        assert!((imp - 1.0).abs() < 0.05);
    }

    #[test]
    fn decisions_decay_slowest() {
        let dec = aged(MemoryKind::Decision, 30);
        let fct = aged(MemoryKind::Fact, 30);
        let evt = aged(MemoryKind::Event, 30);
        let dec_i = effective_importance(&dec);
        let fct_i = effective_importance(&fct);
        let evt_i = effective_importance(&evt);
        assert!(dec_i > fct_i, "{} > {}", dec_i, fct_i);
        assert!(fct_i > evt_i, "{} > {}", fct_i, evt_i);
    }

    #[test]
    fn access_count_boosts_importance() {
        let mut a = aged(MemoryKind::Fact, 30);
        let mut b = aged(MemoryKind::Fact, 30);
        a.access_count = 0;
        b.access_count = 100;
        assert!(effective_importance(&b) > effective_importance(&a));
    }

    #[test]
    fn hebbian_uses_idle_time_not_age() {
        // Two preferences, both 100 days old, but one was just accessed.
        let mut a = aged(MemoryKind::Preference, 100);
        let mut b = aged(MemoryKind::Preference, 100);
        a.last_accessed_at = Utc::now(); // just accessed
        b.last_accessed_at = a.created_at; // never accessed since creation
        let ia = effective_importance(&a);
        let ib = effective_importance(&b);
        assert!(ia > ib * 5.0, "ia={}, ib={}", ia, ib);
    }

    #[test]
    fn critical_bypasses_decay() {
        let mut n = aged(MemoryKind::Event, 1000); // Super old event
        n.salience = crate::node::Salience::Critical;
        assert_eq!(effective_importance(&n), 2.0);
    }
}
