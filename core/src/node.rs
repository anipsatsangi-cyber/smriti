//! Memory node and edge types.
//!
//! A `MemoryNode` is the atomic unit of memory in Smriti. Every memory has a
//! `MemoryKind` that determines its decay curve, retention policy, and
//! recall priority — modeling the distinction Tulving (1972) drew between
//! episodic, semantic, and procedural memory in cognitive psychology.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::scope::Scope;

/// The cognitive category of a memory.
///
/// Each kind has a different forgetting curve and retention strategy:
///
/// | Kind | Half-life | Notes |
/// |------|-----------|-------|
/// | `Decision` | 365 days | Architectural choices, design rationale. Decay-resistant. |
/// | `Fact` | 90 days | Stable knowledge: "service uses JWT". Moderate decay. |
/// | `Event` | 14 days | Time-stamped happenings: "yesterday I deployed". Fast decay. |
/// | `Preference` | 30 days* | User/agent preferences. Decay extends on access (Hebbian). |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryKind {
    /// A long-term decision or architectural choice. Slowest decay.
    Decision,
    /// A stable factual claim. Moderate decay.
    Fact,
    /// A time-stamped event or experience. Fast decay.
    Event,
    /// A user/agent preference. Decay extends on access.
    Preference,
}

impl MemoryKind {
    /// Half-life in days. After this many days, importance drops to 50% of
    /// original (assuming no access boosts).
    pub fn half_life_days(&self) -> f64 {
        match self {
            MemoryKind::Decision => 365.0,
            MemoryKind::Fact => 90.0,
            MemoryKind::Event => 14.0,
            MemoryKind::Preference => 30.0,
        }
    }

    /// Whether this kind extends its half-life on every access (Hebbian
    /// reinforcement: "neurons that fire together, wire together").
    pub fn hebbian(&self) -> bool {
        matches!(self, MemoryKind::Preference | MemoryKind::Decision)
    }

    /// Parse from a string. Falls back to `Fact`.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "decision" => MemoryKind::Decision,
            "event" => MemoryKind::Event,
            "preference" | "pref" => MemoryKind::Preference,
            _ => MemoryKind::Fact,
        }
    }

    /// All kinds in canonical order.
    pub fn all() -> &'static [MemoryKind] {
        &[
            MemoryKind::Decision,
            MemoryKind::Fact,
            MemoryKind::Event,
            MemoryKind::Preference,
        ]
    }
}

impl std::fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MemoryKind::Decision => "decision",
            MemoryKind::Fact => "fact",
            MemoryKind::Event => "event",
            MemoryKind::Preference => "preference",
        };
        f.write_str(s)
    }
}

/// Where a tag came from. Used downstream by field-aware lexical
/// scoring: tags supplied by the caller are treated as gold signal,
/// while tags pulled from raw text by the NER pass are weighted lower
/// because they can be noisy. Without this distinction, over-weighting
/// auto-tags would amplify NER errors on every recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TagSource {
    /// User or agent supplied this tag explicitly via `RememberBuilder::tag()`.
    User,
    /// Tag was extracted from the memory text by the NER pass.
    Auto,
}

/// A single memory.
///
/// Storage layout: text + tags are the primary content. The HDC fingerprint
/// (when computed) is a 2048-bit binary hypervector hash that lets us do
/// O(1) compositional queries via XOR/popcount. The `supersedes` chain
/// preserves audit trail when memories are corrected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Globally unique id (UUIDv4).
    pub id: Uuid,
    /// The raw text of the memory.
    pub text: String,
    /// User/agent-supplied + auto-extracted tags. Drive auto-linking.
    /// Provenance for each tag is tracked in the parallel `tag_sources`
    /// vector (same length, indices aligned).
    pub tags: Vec<String>,
    /// Provenance for each tag in `tags`. Defaults to `User` when
    /// deserializing legacy records that didn't have this field.
    #[serde(default)]
    pub tag_sources: Vec<TagSource>,
    /// Cognitive kind (determines decay curve).
    pub kind: MemoryKind,
    /// Multi-tenant scope (agent / user / session).
    pub scope: Scope,
    /// Static importance, 0.0 – 1.0. Combined with decay at recall time.
    pub importance: f32,
    /// When the memory was created.
    pub created_at: DateTime<Utc>,
    /// When the memory was last successfully retrieved by `recall()`.
    /// Used for Hebbian reinforcement.
    pub last_accessed_at: DateTime<Utc>,
    /// Number of times this memory has been retrieved.
    pub access_count: u32,
    /// If this memory replaces an older one, the old one's id.
    pub supersedes: Option<Uuid>,
    /// If this memory has been replaced, the replacement's id.
    /// Memories with `superseded_by.is_some()` are hidden from recall.
    pub superseded_by: Option<Uuid>,
    /// Pre-computed token count for budget packing.
    pub token_count: usize,
    /// Schema/version field for sync compatibility.
    pub version: u64,
}

impl MemoryNode {
    /// Create a new memory with sensible defaults.
    pub fn new(text: impl Into<String>, kind: MemoryKind, scope: Scope) -> Self {
        let now = Utc::now();
        let text = text.into();
        let token_count = estimate_tokens(&text);
        Self {
            id: Uuid::new_v4(),
            text,
            tags: Vec::new(),
            tag_sources: Vec::new(),
            kind,
            scope,
            importance: 0.5,
            created_at: now,
            last_accessed_at: now,
            access_count: 0,
            supersedes: None,
            superseded_by: None,
            token_count,
            version: 1,
        }
    }

    /// Iterate tag/source pairs. If `tag_sources` is shorter than
    /// `tags` (legacy data, hand-set tags) missing entries default to
    /// `TagSource::User` — matches the historical assumption.
    pub fn iter_tags_with_source(&self) -> impl Iterator<Item = (&str, TagSource)> {
        self.tags.iter().enumerate().map(move |(i, t)| {
            let src = self.tag_sources.get(i).copied().unwrap_or(TagSource::User);
            (t.as_str(), src)
        })
    }

    /// Set tags + sources atomically; lengths must match. If they don't,
    /// the function pads `tag_sources` with `User` to match `tags.len()`.
    /// This is the recommended setter.
    pub fn set_tags_with_sources(&mut self, tags: Vec<String>, sources: Vec<TagSource>) {
        let mut sources = sources;
        sources.resize(tags.len(), TagSource::User);
        self.tags = tags;
        self.tag_sources = sources;
    }

    /// Whether this memory is currently visible (not superseded).
    pub fn is_active(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Age in days (used for decay calculations).
    pub fn age_days(&self) -> f64 {
        let now = Utc::now();
        (now - self.created_at).num_seconds() as f64 / 86400.0
    }

    /// Time since last access in days.
    pub fn idle_days(&self) -> f64 {
        let now = Utc::now();
        (now - self.last_accessed_at).num_seconds() as f64 / 86400.0
    }
}

/// The kind of relationship between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEdge {
    /// Loose association (auto-created by tag overlap or co-occurrence).
    RelatesTo,
    /// One memory contradicts another (manual or detected).
    Contradicts,
    /// One memory provides evidence for another.
    Supports,
    /// One memory is derived from another (causal chain, summarization).
    DerivedFrom,
    /// One memory replaces another (the supersedes chain in graph form).
    Supersedes,
    /// Temporal: this memory's event happened before the target's.
    Before,
    /// Temporal: this memory's event happened after the target's.
    After,
    /// Causal: this memory caused or led to the target.
    CausedBy,
}

impl std::fmt::Display for MemoryEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            MemoryEdge::RelatesTo => "relates_to",
            MemoryEdge::Contradicts => "contradicts",
            MemoryEdge::Supports => "supports",
            MemoryEdge::DerivedFrom => "derived_from",
            MemoryEdge::Supersedes => "supersedes",
            MemoryEdge::Before => "before",
            MemoryEdge::After => "after",
            MemoryEdge::CausedBy => "caused_by",
        };
        f.write_str(s)
    }
}

impl MemoryEdge {
    /// Parse an edge kind from a string. Returns `None` for unknown kinds.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "relates_to" => Some(MemoryEdge::RelatesTo),
            "contradicts" => Some(MemoryEdge::Contradicts),
            "supports" => Some(MemoryEdge::Supports),
            "derived_from" => Some(MemoryEdge::DerivedFrom),
            "supersedes" => Some(MemoryEdge::Supersedes),
            "before" => Some(MemoryEdge::Before),
            "after" => Some(MemoryEdge::After),
            "caused_by" => Some(MemoryEdge::CausedBy),
            _ => None,
        }
    }

    /// Weight of this edge type in graph traversals (e.g., Personalized PageRank).
    /// Causal and temporal edges have higher weight than fuzzy semantic relationships.
    pub fn weight(&self) -> f32 {
        match self {
            MemoryEdge::CausedBy => 5.0,
            MemoryEdge::DerivedFrom => 4.0,
            MemoryEdge::Supports => 3.0,
            MemoryEdge::Supersedes => 3.0,
            MemoryEdge::Before => 2.0,
            MemoryEdge::After => 2.0,
            MemoryEdge::RelatesTo => 1.0,
            MemoryEdge::Contradicts => 1.0,
        }
    }
}

/// Cheap token estimate. We use a 4-chars-per-token approximation for the
/// hot path (estimate_tokens runs on every `remember`). The accurate
/// `tiktoken-rs` count is reserved for budget packing in recall.
pub(crate) fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_have_distinct_half_lives() {
        assert!(MemoryKind::Decision.half_life_days() > MemoryKind::Fact.half_life_days());
        assert!(MemoryKind::Fact.half_life_days() > MemoryKind::Preference.half_life_days());
        assert!(MemoryKind::Preference.half_life_days() > MemoryKind::Event.half_life_days());
    }

    #[test]
    fn kind_parse_falls_back_to_fact() {
        assert_eq!(MemoryKind::parse("decision"), MemoryKind::Decision);
        assert_eq!(MemoryKind::parse("EVENT"), MemoryKind::Event);
        assert_eq!(MemoryKind::parse("garbage"), MemoryKind::Fact);
    }

    #[test]
    fn new_node_has_active_state() {
        let n = MemoryNode::new("hello world", MemoryKind::Fact, Scope::default());
        assert!(n.is_active());
        assert_eq!(n.access_count, 0);
        assert_eq!(n.version, 1);
        assert!(n.token_count > 0);
    }
}
