//! Memory node and edge types.
//!
//! A `MemoryNode` is the atomic unit of memory in Smriti. Every memory has a
//! `MemoryKind` that determines its decay curve, retention policy, and
//! recall priority — modeling the distinction Tulving (1972) drew between
//! episodic, semantic, and procedural memory in cognitive psychology.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::scope::Scope;

/// Generic typed attribute value attached to a memory. Used by the
/// optional structured-filter layer: agents (or callers) can attach
/// arbitrary metadata at write time and filter on it at recall time
/// without the engine baking in any specific dimension (time, space,
/// price, sentiment, etc).
///
/// Equality is *epsilon-aware* on `Number` (tolerance `1e-9`) so that
/// computed numerics like `0.1 + 0.2` compare equal to `0.3`. The
/// derived `PartialEq` would have produced strict-bitwise comparison
/// and a steady stream of "I'm sure I tagged it" bug reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    Boolean(bool),
    Number(f64),
    Text(String),
    List(Vec<AttributeValue>),
}

/// Tolerance used by `AttributeValue::PartialEq` on `Number`. Picked
/// to be tighter than f32 round-off (~1e-7) but loose enough to
/// absorb f64 arithmetic noise.
pub const ATTR_NUMBER_EPSILON: f64 = 1e-9;

impl PartialEq for AttributeValue {
    fn eq(&self, other: &Self) -> bool {
        use AttributeValue::*;
        match (self, other) {
            (Boolean(a), Boolean(b)) => a == b,
            (Number(a), Number(b)) => (a - b).abs() <= ATTR_NUMBER_EPSILON,
            (Text(a), Text(b)) => a == b,
            (List(a), List(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
            }
            _ => false,
        }
    }
}

/// Outcome of `AttrFilter::matches`. The tri-state lets us distinguish
/// "this memory genuinely doesn't match" from "the caller's filter
/// was the wrong type for this attribute" — which is almost certainly
/// a bug in the caller, not a deliberate exclusion. Surfacing the
/// distinction lets recall log a one-off warning instead of silently
/// dropping memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchResult {
    /// Filter applies to this attribute and the value satisfies it.
    Match,
    /// Filter applies to this attribute and the value does NOT satisfy it.
    NoMatch,
    /// Filter and value are of incompatible types (e.g. `Gt(Number)` vs `Text`).
    /// The caller almost certainly meant something else; treat as "exclude
    /// this candidate" but flag at the log level so it's debuggable.
    TypeMismatch,
}

impl MatchResult {
    /// Convenience: was this a positive match? Both `NoMatch` and
    /// `TypeMismatch` return `false`. Recall uses this for the include
    /// decision while still being able to inspect the variant for logging.
    pub fn is_match(self) -> bool {
        matches!(self, MatchResult::Match)
    }
}

/// Filter primitive on a single named attribute.
///
/// Composition primitives (`All`, `Any`) compose multiple filters
/// against the *same* attribute key. They flatten to AND / OR over
/// child results. A `TypeMismatch` from any child propagates up — we
/// don't quietly absorb it, because the agent's wrong-shape filter
/// shouldn't silently turn into "no match."
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrFilter {
    /// Exact equality (epsilon-aware for numbers — see `AttributeValue::eq`).
    Eq(AttributeValue),
    /// Strictly greater than. Numbers only — anything else is `TypeMismatch`.
    Gt(AttributeValue),
    /// Strictly less than. Numbers only.
    Lt(AttributeValue),
    /// **List membership only.** `Contains(v)` against a `List` value
    /// returns `Match` iff `v` is one of the list's elements (epsilon-
    /// aware for `Number` elements). Against a non-list value this is
    /// `TypeMismatch` — use `Substring` for text containment.
    Contains(AttributeValue),
    /// Substring search inside a `Text` value. Case-sensitive by design
    /// (callers who want case-insensitive can lowercase at write time).
    Substring(String),
    /// Inclusive numeric range `[min, max]`. Numbers only.
    Range(AttributeValue, AttributeValue),
    /// All sub-filters must `Match`. Empty `All` matches trivially.
    All(Vec<AttrFilter>),
    /// At least one sub-filter must `Match`. Empty `Any` is `NoMatch`
    /// (no positive evidence to support inclusion).
    Any(Vec<AttrFilter>),
}

impl AttrFilter {
    /// Apply this filter to a single attribute value. See `MatchResult`
    /// for the meaning of each variant.
    pub fn matches(&self, val: &AttributeValue) -> MatchResult {
        use AttrFilter::*;
        use AttributeValue as V;
        use MatchResult::*;

        match self {
            Eq(target) => {
                // Cross-type equality is `TypeMismatch`, not `NoMatch`.
                // Same-type equality uses our epsilon-aware `PartialEq`.
                if std::mem::discriminant(target) != std::mem::discriminant(val) {
                    TypeMismatch
                } else if target == val {
                    Match
                } else {
                    NoMatch
                }
            }
            Gt(target) => match (target, val) {
                (V::Number(t), V::Number(v)) => {
                    if v > t {
                        Match
                    } else {
                        NoMatch
                    }
                }
                _ => TypeMismatch,
            },
            Lt(target) => match (target, val) {
                (V::Number(t), V::Number(v)) => {
                    if v < t {
                        Match
                    } else {
                        NoMatch
                    }
                }
                _ => TypeMismatch,
            },
            Contains(needle) => match val {
                V::List(items) => {
                    if items.iter().any(|item| item == needle) {
                        Match
                    } else {
                        NoMatch
                    }
                }
                _ => TypeMismatch,
            },
            Substring(needle) => match val {
                V::Text(haystack) => {
                    if haystack.contains(needle) {
                        Match
                    } else {
                        NoMatch
                    }
                }
                _ => TypeMismatch,
            },
            Range(lo, hi) => match (lo, hi, val) {
                (V::Number(l), V::Number(h), V::Number(v)) => {
                    if v >= l && v <= h {
                        Match
                    } else {
                        NoMatch
                    }
                }
                _ => TypeMismatch,
            },
            All(children) => {
                if children.is_empty() {
                    return Match;
                }
                for c in children {
                    match c.matches(val) {
                        Match => continue,
                        NoMatch => return NoMatch,
                        TypeMismatch => return TypeMismatch,
                    }
                }
                Match
            }
            Any(children) => {
                if children.is_empty() {
                    return NoMatch;
                }
                let mut saw_type_mismatch = false;
                for c in children {
                    match c.matches(val) {
                        Match => return Match,
                        NoMatch => continue,
                        TypeMismatch => saw_type_mismatch = true,
                    }
                }
                // If at least one child genuinely didn't match, prefer
                // that over the type-mismatch story (the caller had at
                // least one well-typed branch). If ALL branches were
                // type-mismatched, propagate the diagnostic upward.
                if saw_type_mismatch && children.len() == children.iter().filter(|c| matches!(c.matches(val), TypeMismatch)).count() {
                    TypeMismatch
                } else {
                    NoMatch
                }
            }
        }
    }
}

/// The emotional or operational salience of a memory.
/// Highly salient memories bypass standard decay curves and are retained longer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Salience {
    /// Normal decay based on MemoryKind.
    #[default]
    Routine,
    /// Moderately elevated importance; slower decay.
    Important,
    /// Life-safety or system-critical. Bypasses standard decay entirely.
    Critical,
}

impl Salience {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "important" => Salience::Important,
            "critical" => Salience::Critical,
            _ => Salience::Routine,
        }
    }
}

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
    /// An active agent objective. Emits persistent semantic priming. Never decays naturally.
    Goal,
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
            MemoryKind::Goal => 365.0, // Goals are superseded when complete, not decayed.
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
            "goal" => MemoryKind::Goal,
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
            MemoryKind::Goal,
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
            MemoryKind::Goal => "goal",
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
    /// Salience (can bypass decay).
    #[serde(default)]
    pub salience: Salience,
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
    /// Generic structured attributes.
    #[serde(default)]
    pub attributes: HashMap<String, AttributeValue>,
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
            salience: Salience::Routine,
            scope,
            importance: 0.5,
            created_at: now,
            last_accessed_at: now,
            access_count: 0,
            supersedes: None,
            superseded_by: None,
            token_count,
            version: 1,
            attributes: HashMap::new(),
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

    // ── AttributeValue + AttrFilter primitives ──────────────────────

    #[test]
    fn attribute_eq_uses_epsilon_for_numbers() {
        // 0.1 + 0.2 != 0.3 under strict-bitwise equality. Our PartialEq
        // is the user-facing definition of "same number."
        let a = AttributeValue::Number(0.1 + 0.2);
        let b = AttributeValue::Number(0.3);
        assert_eq!(a, b);

        // Genuine inequality stays inequal.
        let c = AttributeValue::Number(0.3);
        let d = AttributeValue::Number(0.4);
        assert_ne!(c, d);

        // Cross-type still not equal.
        assert_ne!(
            AttributeValue::Number(1.0),
            AttributeValue::Text("1.0".to_string())
        );
    }

    #[test]
    fn attr_eq_filter_matches_only_same_type() {
        let f = AttrFilter::Eq(AttributeValue::Text("Seattle".to_string()));
        assert_eq!(
            f.matches(&AttributeValue::Text("Seattle".to_string())),
            MatchResult::Match
        );
        assert_eq!(
            f.matches(&AttributeValue::Text("Boston".to_string())),
            MatchResult::NoMatch
        );
        // Cross-type is TypeMismatch, NOT NoMatch.
        assert_eq!(
            f.matches(&AttributeValue::Number(50.0)),
            MatchResult::TypeMismatch
        );
    }

    #[test]
    fn attr_gt_lt_numbers_only() {
        let f = AttrFilter::Gt(AttributeValue::Number(50.0));
        assert_eq!(f.matches(&AttributeValue::Number(75.0)), MatchResult::Match);
        assert_eq!(f.matches(&AttributeValue::Number(50.0)), MatchResult::NoMatch); // strict >
        assert_eq!(f.matches(&AttributeValue::Number(25.0)), MatchResult::NoMatch);
        assert_eq!(
            f.matches(&AttributeValue::Text("75".to_string())),
            MatchResult::TypeMismatch
        );

        let f = AttrFilter::Lt(AttributeValue::Number(50.0));
        assert_eq!(f.matches(&AttributeValue::Number(25.0)), MatchResult::Match);
        assert_eq!(f.matches(&AttributeValue::Number(50.0)), MatchResult::NoMatch);
    }

    #[test]
    fn attr_range_inclusive_numeric_only() {
        let f = AttrFilter::Range(
            AttributeValue::Number(50.0),
            AttributeValue::Number(100.0),
        );
        assert_eq!(f.matches(&AttributeValue::Number(50.0)), MatchResult::Match); // inclusive lo
        assert_eq!(f.matches(&AttributeValue::Number(100.0)), MatchResult::Match); // inclusive hi
        assert_eq!(f.matches(&AttributeValue::Number(75.0)), MatchResult::Match);
        assert_eq!(f.matches(&AttributeValue::Number(49.9)), MatchResult::NoMatch);
        assert_eq!(f.matches(&AttributeValue::Number(100.1)), MatchResult::NoMatch);
        assert_eq!(
            f.matches(&AttributeValue::Text("75".to_string())),
            MatchResult::TypeMismatch
        );
    }

    #[test]
    fn attr_contains_is_list_membership_only() {
        let needle = AttributeValue::Text("Seattle".to_string());
        let f = AttrFilter::Contains(needle.clone());
        let list = AttributeValue::List(vec![
            AttributeValue::Text("Seattle".to_string()),
            AttributeValue::Text("Portland".to_string()),
        ]);
        assert_eq!(f.matches(&list), MatchResult::Match);

        let other_list =
            AttributeValue::List(vec![AttributeValue::Text("Boston".to_string())]);
        assert_eq!(f.matches(&other_list), MatchResult::NoMatch);

        // CRITICAL: Contains against Text is TypeMismatch, not substring search.
        // Use Substring for that.
        assert_eq!(
            f.matches(&AttributeValue::Text("Seattle, WA".to_string())),
            MatchResult::TypeMismatch
        );
    }

    #[test]
    fn attr_substring_text_only() {
        let f = AttrFilter::Substring("Seattle".to_string());
        assert_eq!(
            f.matches(&AttributeValue::Text("Seattle, WA".to_string())),
            MatchResult::Match
        );
        assert_eq!(
            f.matches(&AttributeValue::Text("Boston".to_string())),
            MatchResult::NoMatch
        );
        // Substring against List is TypeMismatch — use Contains for that.
        assert_eq!(
            f.matches(&AttributeValue::List(vec![AttributeValue::Text(
                "Seattle".to_string()
            )])),
            MatchResult::TypeMismatch
        );
    }

    #[test]
    fn attr_all_short_circuits_on_first_failure() {
        let f = AttrFilter::All(vec![
            AttrFilter::Gt(AttributeValue::Number(50.0)),
            AttrFilter::Lt(AttributeValue::Number(100.0)),
        ]);
        assert_eq!(f.matches(&AttributeValue::Number(75.0)), MatchResult::Match);
        assert_eq!(f.matches(&AttributeValue::Number(150.0)), MatchResult::NoMatch);

        // Empty All is trivially Match (vacuous truth).
        let empty = AttrFilter::All(vec![]);
        assert_eq!(
            empty.matches(&AttributeValue::Number(75.0)),
            MatchResult::Match
        );

        // Type mismatch in any child propagates upward.
        let mixed = AttrFilter::All(vec![
            AttrFilter::Gt(AttributeValue::Number(50.0)),
            AttrFilter::Substring("foo".to_string()), // type-mismatched against Number
        ]);
        assert_eq!(
            mixed.matches(&AttributeValue::Number(75.0)),
            MatchResult::TypeMismatch
        );
    }

    #[test]
    fn attr_any_short_circuits_on_first_match() {
        let f = AttrFilter::Any(vec![
            AttrFilter::Eq(AttributeValue::Text("Seattle".to_string())),
            AttrFilter::Eq(AttributeValue::Text("Portland".to_string())),
        ]);
        assert_eq!(
            f.matches(&AttributeValue::Text("Seattle".to_string())),
            MatchResult::Match
        );
        assert_eq!(
            f.matches(&AttributeValue::Text("Portland".to_string())),
            MatchResult::Match
        );
        assert_eq!(
            f.matches(&AttributeValue::Text("Boston".to_string())),
            MatchResult::NoMatch
        );

        // Empty Any is NoMatch — no positive evidence.
        let empty = AttrFilter::Any(vec![]);
        assert_eq!(
            empty.matches(&AttributeValue::Text("Seattle".to_string())),
            MatchResult::NoMatch
        );
    }

    #[test]
    fn match_result_is_match_only_for_match() {
        assert!(MatchResult::Match.is_match());
        assert!(!MatchResult::NoMatch.is_match());
        assert!(!MatchResult::TypeMismatch.is_match());
    }

    #[test]
    fn attribute_value_round_trips_through_json() {
        // Untagged enum serialization: the JSON shape is the value itself,
        // not {"Text": "..."}. This is what keeps the MCP API ergonomic.
        let v = AttributeValue::List(vec![
            AttributeValue::Text("a".to_string()),
            AttributeValue::Number(1.5),
            AttributeValue::Boolean(true),
        ]);
        let s = serde_json::to_string(&v).unwrap();
        let v2: AttributeValue = serde_json::from_str(&s).unwrap();
        assert_eq!(v, v2);
    }
}
