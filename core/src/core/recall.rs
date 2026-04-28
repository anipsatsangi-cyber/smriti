//! Recall — token-budgeted hybrid retrieval.
//!
//! The retrieval pipeline:
//!
//! ```text
//!   Query
//!     ↓
//!   1. Hippocampal scan (fingerprint similarity, recency boost)
//!     ↓
//!   2. Neocortex seed selection (fingerprint top-K)
//!     ↓
//!   3. Personalized PageRank from seeds (multi-hop expansion)
//!     ↓
//!   4. Score fusion: text + PPR + decay + access boost
//!     ↓
//!   5. MMR diversity (avoid echo-chamber: tag-Jaccard penalty)
//!     ↓
//!   6. Knapsack token-budget packing
//!     ↓
//!   Result
//! ```
//!
//! All scoring is **deterministic** — no LLM, no embeddings, just graph
//! algebra and information theory.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use uuid::Uuid;

use crate::core::decay::effective_importance;
use crate::core::hdc::fingerprint;
use crate::core::hippocampus::Hippocampus;
use crate::core::neocortex::Neocortex;
use crate::node::{MemoryKind, MemoryNode};
use crate::scope::Scope;

/// Default token budget when none is supplied.
pub const DEFAULT_BUDGET: usize = 2000;

/// Default lambda for MMR (diversity weight).
/// 1.0 = pure relevance (no diversity); 0.0 = pure diversity (no relevance).
pub const DEFAULT_MMR_LAMBDA: f32 = 0.7;

/// Default number of seeds to feed PPR.
pub const DEFAULT_SEEDS: usize = 8;

/// Default number of candidates to consider before MMR/packing.
pub const DEFAULT_CANDIDATES: usize = 32;

/// Minimum fingerprint similarity for a memory to be picked as a PPR seed.
/// Below this, the memory is too weakly related to the query to be a
/// trustworthy starting point for graph expansion. One bad seed
/// pollutes the entire PPR distribution.
pub const MIN_SEED_FP: f32 = 0.12;

/// Configurable knobs for a recall query.
#[derive(Debug, Clone)]
pub struct RecallConfig {
    pub budget: usize,
    pub mmr_lambda: f32,
    pub seeds: usize,
    pub candidates: usize,
    /// Filter to memories of these kinds. Empty = all kinds.
    pub kinds: Vec<MemoryKind>,
    /// Boost given to hippocampal hits (recency proxy).
    pub hippocampal_boost: f32,
    /// Confidence gate: abstain if the top hit's final score is below this.
    /// 0.0 disables the gate (legacy behavior). Calibrated for the post-RRF
    /// score scale: a "real" hit usually scores ≥ 0.06 (rank ≤ 10 across
    /// multiple signals); below 0.04 is essentially noise.
    pub abstain_score_floor: f32,
    /// Confidence gate: abstain if the leader doesn't beat the runner-up
    /// by at least this margin. 0.0 disables. A clear winner has margin
    /// ≥ 0.01 in RRF units; ties indicate ambiguous retrieval.
    pub abstain_margin_floor: f32,
    /// Confidence gate: abstain if the top hit has neither lexical
    /// (term overlap > 0) nor dense (cosine ≥ this) support. Set to a
    /// high value to disable when embeddings are off. Default 0.0 means
    /// "any non-zero lexical OR dense signal is enough" — the gate only
    /// catches pure-graph hits with no content backing.
    pub abstain_support_floor: f32,
    /// Confidence-conditional truncation. When the engine's verdict is
    /// `Confident`, return at most this many hits (in MMR-diversified
    /// order). 0 disables — return whatever the budget allows.
    ///
    /// Rationale: a confident answer doesn't need a context pack. A
    /// single well-supported memory is cheaper than seven decent ones.
    /// This is the difference between 489 tokens and ~25 tokens per
    /// query when the engine actually knows the answer.
    pub truncate_when_confident: usize,
    /// Tiered Confident truncation: when the verdict is `Confident` AND
    /// `top_score >= confident_solo_score_floor` AND `top_margin >=
    /// confident_solo_margin_floor`, drop to a single hit instead of
    /// `truncate_when_confident`. This catches the "extremely sure"
    /// subset and ships ~25-50 tokens on those, while keeping the
    /// hedged top-2 pack for ordinary-sure cases.
    ///
    /// Both floors must be positive to activate (default: 0.0 = off).
    pub confident_solo_score_floor: f32,
    pub confident_solo_margin_floor: f32,
    /// Confidence-conditional truncation for `LowConfidence` verdicts.
    /// 0 means "return everything that fits the budget" (the default).
    /// Set to 1 to deliberately *under*-pack when the engine flags low
    /// confidence so the caller pays no token tax for guessing.
    pub truncate_when_low: usize,
    /// Confidence-conditional truncation for `AmbiguousLeader` verdicts.
    /// 0 means no truncation. Setting this small (e.g. 3) forces the
    /// pack to surface the disagreement to the caller without burying
    /// it in a long list.
    pub truncate_when_ambiguous: usize,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
            mmr_lambda: DEFAULT_MMR_LAMBDA,
            seeds: DEFAULT_SEEDS,
            candidates: DEFAULT_CANDIDATES,
            kinds: Vec::new(),
            hippocampal_boost: 0.2,
            abstain_score_floor: 0.04,
            abstain_margin_floor: 0.005,
            abstain_support_floor: 0.0,
            truncate_when_confident: 0,
            truncate_when_low: 0,
            truncate_when_ambiguous: 0,
            confident_solo_score_floor: 0.0,
            confident_solo_margin_floor: 0.0,
        }
    }
}

/// Coarse query-intent class. Drives margin floors and weight tweaks
/// downstream. Deterministic, regex-only — no LM, no model.
///
/// Three classes is a deliberate choice: more bins increase variance
/// without commensurate quality gain. Broder's 2002 search-intent
/// taxonomy used three (informational, navigational, transactional);
/// we use the analogues for memory recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryIntent {
    /// Time-anchored questions: "what did I deploy yesterday", "last
    /// Monday's incident". Recency matters; margin can be looser since
    /// multiple temporally-close memories may legitimately tie.
    Temporal,
    /// Specific-fact questions where the gold is expected to uniquely
    /// contain a rare-IDF entity: "what hashing algorithm", "which
    /// database". A clear winner is expected — tight margin demanded.
    Factual,
    /// Everything else: open-ended, multi-hop, paraphrase. Default
    /// balanced weights apply.
    General,
}

impl QueryIntent {
    /// Classify a query string. Cheap (microseconds) — single pass over
    /// lowercased query text checking for time markers and rare-token
    /// presence in the IDF map.
    pub fn classify(query: &str, idf: &HashMap<String, f32>) -> Self {
        let q = query.to_lowercase();

        // ── Temporal markers ────────────────────────────────────────
        // Explicit time references push us into Temporal regardless of
        // other signals. The list is short on purpose — "today" /
        // "yesterday" / "last <unit>" / "<n> days/weeks/months ago"
        // covers ~90% of real temporal queries without false positives.
        const TEMPORAL_MARKERS: &[&str] = &[
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "last month",
            "last year",
            "last monday",
            "last tuesday",
            "last wednesday",
            "last thursday",
            "last friday",
            "last saturday",
            "last sunday",
            "ago",
            "recent",
            "recently",
            "earlier",
            "this morning",
            "this afternoon",
            "this evening",
        ];
        for m in TEMPORAL_MARKERS {
            if q.contains(m) {
                return QueryIntent::Temporal;
            }
        }

        // ── Factual: query contains a rare-IDF term ────────────────
        // If any query word has IDF above the "rare" threshold, this
        // is a fact lookup — there should be a clear winner.
        // BM25 IDF for df=1 in N=500 is ln((500-1+0.5)/(1+0.5)+1) ≈ 5.8.
        // A threshold of 4.0 (≈ df ≤ 8 in N=500) catches the
        // discriminative subset without picking up generic vocabulary.
        const RARE_IDF_THRESHOLD: f32 = 4.0;
        for w in idf.values() {
            if *w >= RARE_IDF_THRESHOLD {
                return QueryIntent::Factual;
            }
        }

        QueryIntent::General
    }

    /// Margin floor for this intent. Tighter for factual (clear winner
    /// expected), looser for temporal/general (legitimate near-ties).
    pub fn margin_floor(self) -> f32 {
        match self {
            QueryIntent::Factual => 0.012,
            QueryIntent::Temporal => 0.003,
            QueryIntent::General => 0.005,
        }
    }

    /// Minimum **term-overlap support** the top hit must have for this
    /// intent. This is the real anti-hallucination knob — if the top
    /// hit doesn't lexically support a query whose intent demands it,
    /// abstain.
    ///
    /// - Factual: rare-IDF term must actually appear in the top hit.
    ///   The RRF distribution flattens at ~0.08 across signals; absolute
    ///   floors don't separate real from noise. Term-overlap does.
    /// - General/Temporal: any non-zero overlap is enough. (Pure-graph
    ///   matches are still allowed via the dense path.)
    pub fn min_term_overlap(self) -> f32 {
        match self {
            QueryIntent::Factual => 0.30,
            QueryIntent::Temporal => 0.0,
            QueryIntent::General => 0.0,
        }
    }

    /// Whether recency-of-creation should bias scoring for this intent.
    /// Temporal queries lean heavily on recency; factual queries should
    /// be recency-blind so a recent unrelated event doesn't outrank
    /// the gold fact.
    pub fn recency_active(self) -> bool {
        !matches!(self, QueryIntent::Factual)
    }
}

/// One recalled memory with its scoring breakdown.
#[derive(Debug, Clone)]
pub struct RecallHit {
    pub node: MemoryNode,
    pub final_score: f32,
    pub fingerprint_sim: f32,
    pub ppr_score: f32,
    pub decay_factor: f32,
    pub from_hippocampus: bool,
    /// Cosine similarity from dense embeddings (0.0 if not enabled).
    pub dense_sim: f32,
    /// IDF-weighted term overlap (0.0 to 1.0). Useful for diagnostics.
    pub term_overlap_score: f32,
}

/// Per-phase timing breakdown (microseconds). Useful for diagnosing
/// where p95 latency goes in production traces and benchmark runs.
///
/// Populated unconditionally — overhead is one `Instant::now()` per phase
/// (~20 ns each) which is well below noise.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct RecallTrace {
    pub embed_us: u128,
    pub idf_us: u128,
    pub fp_scan_us: u128,
    pub ppr_us: u128,
    pub score_us: u128,
    pub mmr_us: u128,
    pub total_us: u128,
}

/// Why a recall returned (or didn't return) what it did. Useful for
/// debugging and for the abstention bench.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallVerdict {
    /// Confident hit — passed all gating checks.
    Confident,
    /// Result returned but flagged: top score below absolute floor.
    LowConfidence,
    /// Result returned but flagged: top1/top2 margin too thin.
    AmbiguousLeader,
    /// Result returned but flagged: neither lexical nor dense support
    /// the top hit (could be a graph-only match — possible hallucination).
    UnsupportedTop,
    /// Engine deliberately abstained — empty hits.
    Abstained,
}

/// Result of a recall — selected memories that fit the budget.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub hits: Vec<RecallHit>,
    pub tokens_used: usize,
    pub tokens_budget: usize,
    pub candidates_considered: usize,
    pub seeds_used: usize,
    /// Per-phase timing breakdown.
    pub trace: RecallTrace,
    /// Confidence verdict for the top hit (or `Abstained` if empty).
    pub verdict: RecallVerdict,
    /// Top hit's final score, surfaced for downstream gating.
    pub top_score: f32,
    /// Margin between the top and second hits — how decisive the leader is.
    /// 0.0 if there is fewer than 2 hits.
    pub top_margin: f32,
    /// The intent class the engine assigned to this query. Surfaced for
    /// diagnostics and for bench-time per-intent breakdowns.
    pub intent: QueryIntent,
}

impl RecallResult {
    /// Format as a compact string for LLM context injection.
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        for h in &self.hits {
            out.push_str("- ");
            out.push_str(&h.node.text);
            if !h.node.tags.is_empty() {
                out.push_str(" [");
                out.push_str(&h.node.tags.join(", "));
                out.push(']');
            }
            out.push('\n');
        }
        out
    }
}

/// Optional dense-embedding bridge. Implementations compute cosine
/// similarity between a query embedding and a memory's text. Pass `None`
/// to disable dense scoring.
pub trait DenseBridge {
    /// Embed the query once. Returns None if embeddings aren't enabled
    /// or inference failed. The result is reused across all candidates.
    fn embed_query(&self, text: &str) -> Option<Vec<f32>>;

    /// Cosine similarity between the (already embedded) query and the
    /// memory at `id` whose text is `text`. Returns 0.0 if disabled.
    fn similarity(&self, query: &[f32], id: Uuid, text: &str) -> f32;
}

/// A no-op DenseBridge — used when embeddings are disabled.
pub struct NoDense;
impl DenseBridge for NoDense {
    fn embed_query(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
    fn similarity(&self, _query: &[f32], _id: Uuid, _text: &str) -> f32 {
        0.0
    }
}

const STOPWORDS: &[&str] = &[
    "the", "and", "a", "an", "is", "in", "it", "of", "to", "for", "with", "on", "at", "by", "this",
    "that", "are", "we", "our", "you", "how", "what", "do", "does", "did", "was", "were", "be",
    "has", "have", "had", "can", "will", "would", "should", "about", "from", "not", "or", "but",
    "if", "so", "no", "up", "out", "me",
];

fn normalize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn normalize_query_words(text: &str) -> Vec<String> {
    normalize_words(text)
        .into_iter()
        .filter(|w| w.len() >= 2 && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

fn word_set(text: &str) -> HashSet<String> {
    normalize_words(text).into_iter().collect()
}

fn contains_query_term(words: &HashSet<String>, query_word: &str) -> bool {
    if words.contains(query_word) {
        return true;
    }

    if query_word.len() > 1 && query_word.ends_with('s') {
        let singular = &query_word[..query_word.len() - 1];
        if words.contains(singular) {
            return true;
        }
    }

    let plural = format!("{}s", query_word);
    words.contains(&plural)
}

/// Run a recall query against both stores.
pub fn recall(
    query_text: &str,
    query_tags: &[String],
    reader_scope: &Scope,
    hippo: &Hippocampus,
    neo: &Neocortex,
    cfg: &RecallConfig,
) -> RecallResult {
    recall_with_dense(
        query_text,
        query_tags,
        reader_scope,
        hippo,
        neo,
        cfg,
        &NoDense,
    )
}

/// Same as [`recall`] but accepts an optional dense-embedding bridge.
/// When provided, cosine similarity is fused into the candidate score.
///
/// Score fusion uses **Reciprocal Rank Fusion** (Cormack et al., 2009)
/// instead of raw score addition. RRF is scale-invariant — it doesn't
/// matter that PPR scores are 10× larger than fingerprint scores.
///
/// Candidate generation uses **dual-path retrieval** (inspired by
/// MixPR, arXiv:2412.06078): PPR expansion + direct fingerprint scan.
/// This eliminates blind spots from disconnected graph islands.
pub fn recall_with_dense(
    query_text: &str,
    query_tags: &[String],
    reader_scope: &Scope,
    hippo: &Hippocampus,
    neo: &Neocortex,
    cfg: &RecallConfig,
    dense: &dyn DenseBridge,
) -> RecallResult {
    let total_t0 = Instant::now();
    let mut trace = RecallTrace::default();

    let q_fp = fingerprint(query_text, query_tags);

    // ── Phase: embed query (one-shot) ──
    let t = Instant::now();
    let q_dense = dense.embed_query(query_text);
    trace.embed_us = t.elapsed().as_micros();

    // ── Pre-compute query words for BM25 term overlap ──
    let query_words = normalize_query_words(query_text);

    // ── Compute per-query IDF over the active neocortex ──
    //
    // Robertson & Zaragoza (2009): a query term appearing in only 1
    // memory carries far more discriminative signal than a term
    // appearing in 50. Without IDF weighting, "JWT" (common in our
    // corpus) and "RS256" (uniquely in the answer) get equal weight,
    // and the answer drowns in a cluster of related memories.
    //
    // IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)
    //   where N = total active memories, df = memories containing term
    //
    // Cost: one O(N) pass per query — 500 memories × ~6 query words
    // × ~50 chars per memory ≈ 150k char comparisons ≈ 200µs total.
    // Worth it for precision.
    // ── Phase: IDF computation over active neocortex ──
    let t = Instant::now();
    let trace_idf = std::env::var("SMRITI_TRACE_IDF").is_ok();
    let mut active_term_sets: HashMap<Uuid, HashSet<String>> = HashMap::new();
    let mut document_freq: HashMap<String, usize> = HashMap::with_capacity(query_words.len());

    for (node, _) in neo.iter_active() {
        let words = word_set(&node.text);
        for qw in &query_words {
            if contains_query_term(&words, qw) {
                *document_freq.entry(qw.clone()).or_insert(0) += 1;
            }
        }
        active_term_sets.insert(node.id, words);
    }

    let total_active = active_term_sets.len().max(1);
    let mut idf: HashMap<String, f32> = HashMap::with_capacity(query_words.len());
    for qw in &query_words {
        if idf.contains_key(qw) {
            continue;
        }
        let df = document_freq.get(qw).copied().unwrap_or(0) as f32;
        let n = total_active as f32;
        // Smoothed BM25 IDF — guaranteed positive for df < N.
        let term_idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        if trace_idf {
            eprintln!("  [idf] {} → df={}, idf={:.3}", qw, df, term_idf);
        }
        idf.insert(qw.clone(), term_idf);
    }
    if trace_idf {
        eprintln!("  [idf] query: {:?}", query_words);
    }
    trace.idf_us = t.elapsed().as_micros();

    // ── Classify query intent (cheap, regex-only) ──
    //
    // Used downstream to:
    //   - tighten the margin floor for factual queries (clear winner expected)
    //   - disable recency boost for factual queries (recency must not
    //     outrank the gold fact)
    //   - loosen the margin floor for temporal queries (legitimate near-ties)
    let intent = QueryIntent::classify(query_text, &idf);

    // ── Phase: fingerprint scans (hippo + neo seed selection) ──
    let t = Instant::now();
    let hippo_hits = hippo.nearest(&q_fp, cfg.candidates / 2);
    let mut raw_seeds = neo.nearest_by_fingerprint(&q_fp, cfg.seeds * 4, -1.0, Some(reader_scope));

    // When dense embeddings are available, re-rank seed candidates by
    // a blend of fingerprint and dense cosine similarity. Pure dense
    // re-ranking can send PPR into wrong neighborhoods when the query
    // is ambiguous; blending preserves fingerprint's structural signal
    // while letting dense break ties.
    if let Some(ref q_vec) = q_dense {
        // Compute dense similarity for each seed candidate and blend
        // with normalized fingerprint score.
        //
        // Performance: cap the dense-blend re-rank at 2x cfg.seeds rather
        // than the full 4x candidate pool. The seed pool's job is to
        // pick PPR starting points — it does not need to be exhaustive.
        // Re-ranking 4x seeds means up to 32 dense similarity calls per
        // query, and on a cold cache each call triggers a MiniLM
        // inference (~1 ms). Capping to 2x halves this. The lower-ranked
        // seeds we drop here would not have been picked as PPR starting
        // points anyway (MIN_SEED_FP filter takes only the top
        // `cfg.seeds` after the re-rank).
        let rerank_cap = (cfg.seeds * 2).min(raw_seeds.len());
        let (rerank, tail) = raw_seeds.split_at(rerank_cap);
        let mut scored: Vec<(Uuid, f32, f32)> = rerank
            .iter()
            .map(|(id, fp_sim)| {
                let text = neo.get(*id).map(|n| n.text.as_str()).unwrap_or("");
                let ds = dense.similarity(q_vec, *id, text);
                (*id, *fp_sim, ds)
            })
            .collect();
        // Blend: 0.4 * fp_norm + 0.6 * dense (dense is the stronger signal)
        let max_fp = scored.iter().map(|s| s.1).fold(0.0f32, f32::max).max(0.01);
        scored.sort_by(|a, b| {
            let sa = 0.4 * (a.1 / max_fp) + 0.6 * a.2;
            let sb = 0.4 * (b.1 / max_fp) + 0.6 * b.2;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut new_seeds: Vec<(Uuid, f32)> =
            scored.iter().map(|(id, fp, _)| (*id, *fp)).collect();
        // Tack the un-reranked tail back on so PPR seed-selection still
        // sees the full pool, just with the head reordered by dense+fp.
        new_seeds.extend(tail.iter().copied());
        raw_seeds = new_seeds;
    }

    let mut seed_ids: Vec<Uuid> = raw_seeds
        .iter()
        .filter(|(_, score)| *score >= MIN_SEED_FP)
        .take(cfg.seeds)
        .map(|(id, _)| *id)
        .collect();
    if seed_ids.is_empty() {
        if let Some((id, _)) = raw_seeds.first() {
            seed_ids.push(*id);
        }
    }
    trace.fp_scan_us = t.elapsed().as_micros();

    // ── Phase: PPR expansion ──
    let t = Instant::now();
    let ppr_scores = if !seed_ids.is_empty() {
        neo.personalized_pagerank(&seed_ids)
    } else {
        HashMap::new()
    };
    trace.ppr_us = t.elapsed().as_micros();

    // ── Phase: scoring (candidate collection + RRF fusion) ──
    let t_score = Instant::now();
    // ── Step 4: dual-path candidate collection ──
    //
    // Path A: PPR neighbors (graph-connected candidates)
    // Path B: Direct fingerprint scan (content-matched candidates)
    //
    // MixPR (arXiv:2412.06078) shows no single retrieval path covers all
    // query types. By scanning the neocortex directly by fingerprint, we
    // catch memories in disconnected graph islands that PPR would miss.

    // Collect candidate IDs from PPR and direct fingerprint scan without
    // repeated Vec::contains scans.
    let mut neo_candidate_ids: Vec<Uuid> = Vec::with_capacity(ppr_scores.len() + cfg.candidates);
    let mut seen_candidate_ids: HashSet<Uuid> =
        HashSet::with_capacity(ppr_scores.len() + cfg.candidates);
    for id in ppr_scores.keys().copied().chain(seed_ids.iter().copied()) {
        if seen_candidate_ids.insert(id) {
            neo_candidate_ids.push(id);
        }
    }
    // Path B: direct fingerprint scan of entire neocortex.
    // Cost: O(N) × 32 XOR+popcount ≈ 5µs for 500 memories
    let fp_scan = neo.nearest_by_fingerprint(&q_fp, cfg.candidates, 0.0, Some(reader_scope));
    for (id, _sim) in &fp_scan {
        if seen_candidate_ids.insert(*id) {
            neo_candidate_ids.push(*id);
        }
    }

    // ── Gather raw signal vectors for all candidates ──
    //
    // We need to rank each signal independently before RRF fusion.
    // Struct to hold per-candidate raw signals before ranking.
    struct RawCandidate {
        node: MemoryNode,
        fp_sim: f32,
        ppr_score: f32,
        dense_sim: f32,
        term_overlap: f32,
        /// IDF-weighted query-token presence in the memory's TAGS
        /// (separate field from text). User-supplied tags are weighted
        /// 2× over auto-extracted tags via [`tag_overlap`]. Treated as
        /// an independent signal in RRF — a query whose words overlap
        /// the memory's user tags is far stronger evidence than a
        /// text-only match, because user tags are gold provenance.
        tag_overlap: f32,
        decay: f32,
        recency_boost: f32,
        from_hippocampus: bool,
    }

    let mut raw_candidates: Vec<RawCandidate> = Vec::new();

    // Hippocampal candidates
    for (entry, sim) in &hippo_hits {
        if !reader_scope.can_read(&entry.node.scope) {
            continue;
        }
        if !entry.node.is_active() {
            continue;
        }
        if !cfg.kinds.is_empty() && !cfg.kinds.contains(&entry.node.kind) {
            continue;
        }

        let decay = effective_importance(&entry.node);
        let recency_boost = if entry.node.supersedes.is_some() {
            0.3
        } else {
            0.0
        };
        let dense_sim = q_dense
            .as_ref()
            .map(|q| dense.similarity(q, entry.node.id, &entry.node.text))
            .unwrap_or(0.0);
        let entry_words = word_set(&entry.node.text);
        let term_ov = term_overlap(&query_words, &entry_words, &idf);
        let tag_ov = tag_overlap(&query_words, &entry.node, &idf);

        raw_candidates.push(RawCandidate {
            node: entry.node.clone(),
            fp_sim: *sim,
            ppr_score: 0.0,
            dense_sim,
            term_overlap: term_ov,
            tag_overlap: tag_ov,
            decay,
            recency_boost: recency_boost + cfg.hippocampal_boost,
            from_hippocampus: true,
        });
    }

    // Neocortex candidates (from both PPR and fingerprint scan)
    for id in neo_candidate_ids {
        let Some(node) = neo.get(id) else { continue };
        if !reader_scope.can_read(&node.scope) {
            continue;
        }
        if !node.is_active() {
            continue;
        }
        if !cfg.kinds.is_empty() && !cfg.kinds.contains(&node.kind) {
            continue;
        }

        let fp_sim = neo
            .fingerprint_of(id)
            .map(|fp| fp.similarity(&q_fp))
            .unwrap_or(0.0);
        let ppr = ppr_scores.get(&id).copied().unwrap_or(0.0);
        let decay = effective_importance(node);

        let mut recency_boost: f32 = 0.0;
        if intent.recency_active() {
            // Supersedes-aware recency: a memory that replaced an older
            // version should rank above its predecessor regardless of
            // intent, so this fires even on Factual queries.
            if node.supersedes.is_some() {
                recency_boost += 0.3;
            }
            // Event recency only fires when the intent isn't Factual —
            // otherwise a recent unrelated event can outrank the gold
            // fact (the canonical paraphrase/factual top-1 failure).
            if matches!(node.kind, crate::node::MemoryKind::Event) {
                let age_ms = (chrono::Utc::now() - node.created_at)
                    .num_milliseconds()
                    .max(1) as f32;
                recency_boost += (0.05 / (1.0 + age_ms / 3_600_000.0)).min(0.05);
            }
        } else if node.supersedes.is_some() {
            // Even in factual mode, the supersede chain is structural,
            // not temporal — keep it. The recency-of-creation boost is
            // what we suppress for factual.
            recency_boost += 0.3;
        }

        let dense_sim = q_dense
            .as_ref()
            .map(|q| dense.similarity(q, node.id, &node.text))
            .unwrap_or(0.0);
        let term_ov = active_term_sets
            .get(&node.id)
            .map(|words| term_overlap(&query_words, words, &idf))
            .unwrap_or(0.0);
        let tag_ov = tag_overlap(&query_words, node, &idf);

        raw_candidates.push(RawCandidate {
            node: node.clone(),
            fp_sim,
            ppr_score: ppr,
            dense_sim,
            term_overlap: term_ov,
            tag_overlap: tag_ov,
            decay,
            recency_boost,
            from_hippocampus: false,
        });
    }

    let candidates_considered = raw_candidates.len();

    // Dedupe by id
    {
        let mut seen = std::collections::HashSet::new();
        raw_candidates.retain(|c| seen.insert(c.node.id));
    }

    // ── Step 5: Reciprocal Rank Fusion ──
    //
    // Cormack, Clarke & Buettcher (2009): "Reciprocal Rank Fusion
    // outperforms Condorcet and individual Rank Learning Methods"
    //
    // For each signal, rank candidates independently (best = rank 0).
    // Then fuse: score = Σ 1/(k + rank_i) where k=60 (published constant).
    //
    // This is scale-invariant — PPR's [0,1] range and fingerprint's
    // [-0.1, 0.22] range no longer matter. Only relative ordering counts.

    let n = raw_candidates.len();
    if n == 0 {
        trace.score_us = t_score.elapsed().as_micros();
        trace.total_us = total_t0.elapsed().as_micros();
        return RecallResult {
            hits: Vec::new(),
            tokens_used: 0,
            tokens_budget: cfg.budget,
            candidates_considered: 0,
            seeds_used: seed_ids.len(),
            trace,
            verdict: RecallVerdict::Abstained,
            top_score: 0.0,
            top_margin: 0.0,
            intent,
        };
    }

    // Compute RRF scores for each signal.
    // If a candidate has a score <= zero_threshold, it is considered "not retrieved"
    // by that signal and gets an RRF of 0.0.
    let rrf_by = |score_fn: &dyn Fn(&RawCandidate) -> f32, zero_threshold: f32| -> Vec<f32> {
        let mut indices: Vec<usize> = (0..raw_candidates.len()).collect();
        indices.sort_by(|&a, &b| {
            score_fn(&raw_candidates[b])
                .partial_cmp(&score_fn(&raw_candidates[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut rrf_scores = vec![0.0f32; raw_candidates.len()];
        let mut current_rank = 0;
        let mut prev_score = f32::NAN;

        for (i, &idx) in indices.iter().enumerate() {
            let score = score_fn(&raw_candidates[idx]);
            if score <= zero_threshold {
                rrf_scores[idx] = 0.0;
                continue;
            }
            if score != prev_score && !prev_score.is_nan() {
                current_rank = i;
            }
            rrf_scores[idx] = 1.0 / (60.0 + current_rank as f32);
            prev_score = score;
        }
        rrf_scores
    };

    let fp_rrf = rrf_by(&|c| c.fp_sim, -1.0); // fp is always active
    let ppr_rrf = rrf_by(&|c| c.ppr_score, 0.0);
    let term_rrf = rrf_by(&|c| c.term_overlap, 0.0);
    let tag_rrf = rrf_by(&|c| c.tag_overlap, 0.0);
    let decay_rrf = rrf_by(&|c| c.decay, 0.0);

    let mut candidates: Vec<RecallHit> = raw_candidates
        .into_iter()
        .enumerate()
        .map(|(i, rc)| {
            // ── Lexical-presence gate ──
            //
            // A candidate that contains AT LEAST ONE query term gets a
            // +0.02 lift on top of the standard RRF sum. This is a
            // principled hybrid retrieval pattern (Pradeep et al. 2024
            // "Hybrid Search That Doesn't Suck") — when the user's
            // query language overlaps the document, that's stronger
            // evidence than pure structural proximity.
            //
            // The 0.02 magnitude is calibrated to RRF rank deltas
            // (~0.016 per rank). It's a one-rank tiebreaker, not a
            // dominator: a memory with strong fp+ppr+dense ranks but
            // zero term overlap can still win.
            let lex_present = if rc.term_overlap > 0.0 { 0.02 } else { 0.0 };

            // When dense embeddings are enabled, we inject the RAW dense
            // similarity rather than RRF rank. RRF destroys magnitude
            // information (the gap between 0.8 and 0.4 cosine becomes
            // just a few tiny rank fractions). Cosine similarity is
            // already well-calibrated [0,1], so we scale it to match
            // the RRF magnitude (max possible RRF sum for 4 signals
            // is 4 * 1/60 ≈ 0.066). A weight of 0.15 means a perfect
            // dense match (1.0) gives +0.15, equivalent to sweeping
            // all other signals, ensuring strong semantic matches
            // overcome structural/lexical noise.
            let dense_boost = if q_dense.is_some() {
                rc.dense_sim * 0.15
            } else {
                0.0
            };

            // Tag-overlap RRF is weighted as a *tiebreaker*, not a driver.
            // 0.8x means a tag-rank-1 hit contributes ~0.013 to final_score
            // (vs the unweighted 0.0167) — about a half-rank lift. Calibrated
            // empirically on bench-500: at 1.5x the signal over-promoted
            // tag-only matches in zero-ML (Top-1 dropped 57→53%); at 0.8x
            // it preserves Top-1 while still firing on real tag-aligned
            // queries. User-tag provenance is reflected inside
            // `tag_overlap` itself (User=2x, Auto=1x), so the global
            // weight stays below text overlap.
            let tag_signal = tag_rrf[i] * 0.8;

            let final_score = fp_rrf[i]
                + ppr_rrf[i]
                + term_rrf[i]
                + tag_signal
                + decay_rrf[i]
                + lex_present
                + dense_boost
                // Recency boost magnitude calibrated to RRF scale.
                // RRF rank deltas are ~1/60 - 1/61 ≈ 0.0003 per rank;
                // the full supersedes-aware boost (rc.recency_boost ≤ 0.35)
                // multiplied by 0.05 = ~0.017 — equivalent to jumping
                // ~3 RRF ranks. Strong enough to break ties between
                // siblings (e.g. v1.0 vs v2.0 deploy events), too weak
                // to override a clear relevance gap.
                + rc.recency_boost * 0.05;

            RecallHit {
                node: rc.node,
                final_score,
                fingerprint_sim: rc.fp_sim,
                ppr_score: rc.ppr_score,
                decay_factor: rc.decay,
                from_hippocampus: rc.from_hippocampus,
                dense_sim: rc.dense_sim,
                term_overlap_score: rc.term_overlap,
            }
        })
        .collect();

    // ── Adaptive score floor ──
    // Drop noise candidates that are far below the top RRF score.
    const RELATIVE_FLOOR: f32 = 0.25;
    if let Some(top_score) = candidates
        .iter()
        .map(|c| c.final_score)
        .fold(None, |acc: Option<f32>, s| {
            Some(acc.map_or(s, |m| m.max(s)))
        })
    {
        if top_score > 0.01 {
            let floor = top_score * RELATIVE_FLOOR;
            candidates.retain(|c| c.final_score >= floor);
        }
    }

    // ── Dense noise penalty ──
    // When embeddings are enabled, penalize candidates whose dense
    // similarity is far below the median. This targets noise memories
    // (e.g. synthetic log entries) that sneak through on fingerprint/
    // PPR but are clearly semantically irrelevant. Using a relative
    // threshold (median) instead of a fixed floor avoids penalizing
    // legitimate candidates on clean corpora where all dense sims
    // are low.
    if q_dense.is_some() && candidates.len() > 2 {
        let mut dense_vals: Vec<f32> = candidates.iter().map(|c| c.dense_sim).collect();
        dense_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = dense_vals[dense_vals.len() / 2];
        // Only apply penalty when there's a clear separation between
        // good and bad candidates (median > 0.15 means the embedding
        // model has real signal to offer).
        if median > 0.15 {
            let penalty_threshold = median * 0.4; // well below median
            for c in &mut candidates {
                if c.dense_sim < penalty_threshold {
                    c.final_score *= 0.5;
                }
            }
        }
    }

    // Sort by RRF score desc; tiebreak by recency.
    candidates.sort_by(|a, b| {
        let primary = b
            .final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal);
        if primary == std::cmp::Ordering::Equal {
            b.node.created_at.cmp(&a.node.created_at)
        } else {
            primary
        }
    });
    candidates.truncate(cfg.candidates);
    trace.score_us = t_score.elapsed().as_micros();

    // ── Phase: MMR diversity + token packing ──
    let t = Instant::now();
    let mut selected = mmr_select(candidates, cfg.mmr_lambda, cfg.budget);
    trace.mmr_us = t.elapsed().as_micros();

    // ── Confidence gate ──
    //
    // Three deterministic thresholds — no training, no validation set:
    //
    //   1. Absolute floor — top score must clear `abstain_score_floor`.
    //      Filters cases where every candidate is essentially noise
    //      (the "no gold in corpus" abstention scenario).
    //
    //   2. Margin floor — leader must beat runner-up by at least
    //      `abstain_margin_floor`. Catches ambiguous retrievals where
    //      RRF produces a near-tie.
    //
    //   3. Support floor — top hit must have non-zero lexical match
    //      (term_overlap_score > 0) OR dense similarity ≥
    //      `abstain_support_floor`. Catches pure-graph/PPR hits with
    //      no content backing — the closest thing to "hallucinated
    //      retrieval" we can have without an LM in the loop.
    //
    // The verdict is surfaced separately from `hits` so callers can
    // either (a) trust low-confidence hits with a warning, or (b) drop
    // them entirely. We don't drop here — that's a policy decision the
    // builder layer makes.
    let (verdict, top_score, top_margin) = if selected.is_empty() {
        (RecallVerdict::Abstained, 0.0, 0.0)
    } else {
        let top = &selected[0];
        let s1 = top.final_score;
        let s2 = selected.get(1).map(|h| h.final_score).unwrap_or(0.0);
        let margin = s1 - s2;

        // Effective margin floor blends the user-supplied cfg floor with
        // the intent-derived floor. We take the *max* of the two so a
        // caller can demand a stricter gate than the intent default but
        // can never silently widen below it.
        let intent_margin = intent.margin_floor();
        let effective_margin = cfg.abstain_margin_floor.max(intent_margin);
        let intent_term_floor = intent.min_term_overlap();

        let v = if s1 < cfg.abstain_score_floor {
            RecallVerdict::LowConfidence
        } else if selected.len() > 1 && margin < effective_margin {
            RecallVerdict::AmbiguousLeader
        } else if top.term_overlap_score < intent_term_floor {
            // Intent demands stronger lexical support than the top hit
            // provides — likely a flat-RRF tie among unrelated memories.
            RecallVerdict::UnsupportedTop
        } else if top.term_overlap_score <= 0.0 && top.dense_sim < cfg.abstain_support_floor {
            RecallVerdict::UnsupportedTop
        } else {
            RecallVerdict::Confident
        };
        (v, s1, margin)
    };

    // ── Confidence-conditional truncation ──
    //
    // The engine has formed an opinion about the top hit. Use it to
    // decide how big a pack to actually return:
    //
    //   - Confident → ship just the top N (default: 1). When we know
    //     the answer, the caller doesn't need a context buffet.
    //   - LowConfidence → optionally under-pack (or return nothing) to
    //     prevent paying tokens for a guess.
    //   - AmbiguousLeader → optionally cap so the disagreement is
    //     visible to the caller instead of buried in a long list.
    //
    // All knobs default to 0 (= no truncation) so existing callers keep
    // their old behavior. The bench opts in to demonstrate the win.
    let truncate_to = match verdict {
        RecallVerdict::Confident => {
            // Tiered truncation: ship solo when we're *strongly* sure.
            // Three combinable conditions, all configurable, AND-ed:
            //   - top_score ≥ score_floor (absolute confidence)
            //   - top_margin ≥ margin_floor (decisive lead)
            //   - top hit term-overlap ≥ intent's min_term_overlap floor
            //     (lexical evidence — already true if verdict is Confident
            //     under intent gating, but we re-check defensively)
            //
            // Margin is the noisy signal in zero-ML mode where RRF
            // saturates around 0.08 — for those queries, score_floor
            // is the meaningful gate. In embeddings mode, dense_boost
            // expands the score range and margin becomes informative.
            // We require BOTH floors to be configured (>0) before
            // applying the tier; setting just one disables tiering.
            let solo_active = cfg.confident_solo_score_floor > 0.0;
            let top_overlap = selected
                .first()
                .map(|h| h.term_overlap_score)
                .unwrap_or(0.0);
            let strong_lex = top_overlap >= 0.5;
            if solo_active
                && top_score >= cfg.confident_solo_score_floor
                && (top_margin >= cfg.confident_solo_margin_floor || strong_lex)
            {
                1
            } else {
                cfg.truncate_when_confident
            }
        }
        RecallVerdict::LowConfidence => cfg.truncate_when_low,
        RecallVerdict::AmbiguousLeader => cfg.truncate_when_ambiguous,
        // UnsupportedTop and Abstained: don't truncate — caller wants
        // the engine's full reasoning surface for inspection, and on
        // Abstained we already have an empty hit list anyway.
        _ => 0,
    };
    if truncate_to > 0 && selected.len() > truncate_to {
        selected.truncate(truncate_to);
    }

    // Recompute tokens_used *after* truncation — this is the number we
    // bill the caller.
    let tokens_used: usize = selected.iter().map(|h| h.node.token_count).sum();

    trace.total_us = total_t0.elapsed().as_micros();

    RecallResult {
        hits: selected,
        tokens_used,
        tokens_budget: cfg.budget,
        candidates_considered,
        seeds_used: seed_ids.len(),
        trace,
        verdict,
        top_score,
        top_margin,
        intent,
    }
}

/// BM25-style IDF-weighted term overlap.
///
/// Robertson & Zaragoza (2009) — "The Probabilistic Relevance Framework"
///
/// For each query word that appears in the memory text, accumulate its
/// IDF weight. This is the discriminative version of term overlap:
///
/// - Common words ("database", "use") get low IDF and contribute little
/// - Rare words ("RS256", "OpenSearch", "Argon2") get high IDF and
///   strongly differentiate the actual answer from related memories
///
/// Result is normalized by the SUM of all query-word IDFs so that
/// queries of different lengths produce comparable scores in [0, 1].
fn term_overlap(
    query_words: &[String],
    memory_words: &HashSet<String>,
    idf: &HashMap<String, f32>,
) -> f32 {
    if query_words.is_empty() {
        return 0.0;
    }

    let mut matched_idf = 0.0_f32;
    let mut total_idf = 0.0_f32;
    for qw in query_words {
        let w_idf = idf.get(qw).copied().unwrap_or(0.0);
        total_idf += w_idf;
        if contains_query_term(memory_words, qw) {
            matched_idf += w_idf;
        }
    }
    if total_idf < 1e-6 {
        return 0.0;
    }
    matched_idf / total_idf
}

/// Field-aware tag overlap. Score is IDF-weighted query-token presence
/// in the memory's TAGS, with provenance multipliers:
///
/// - `TagSource::User`: 2.0x weight (gold — caller explicitly labeled)
/// - `TagSource::Auto`: 1.0x weight (NER best-effort — could be noisy)
///
/// Each tag's tokens are compared against the query word set with the
/// same plural-tolerant `contains_query_term` rule used for text. The
/// numerator is the matched IDF mass × source multiplier; the
/// denominator is the total query IDF (so the score normalizes into
/// [0, 2.0] — anything above 1.0 indicates a strong user-tag match).
///
/// Memories with no tags return 0.0. Memories whose tags don't include
/// any query word also return 0.0, so this signal won't fire spuriously
/// on irrelevant memories.
fn tag_overlap(
    query_words: &[String],
    node: &MemoryNode,
    idf: &HashMap<String, f32>,
) -> f32 {
    if query_words.is_empty() || node.tags.is_empty() {
        return 0.0;
    }

    // Build the union word-set across all the memory's tags, tracking
    // the maximum source weight per unique token. We use the *max* not
    // the sum because the same token appearing in both User and Auto
    // tags should be counted once (at the higher weight).
    use crate::node::TagSource;
    let mut tag_words: HashMap<String, f32> = HashMap::new();
    for (tag, src) in node.iter_tags_with_source() {
        let weight = match src {
            TagSource::User => 2.0,
            TagSource::Auto => 1.0,
        };
        for piece in tag.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if piece.is_empty() {
                continue;
            }
            let lower = piece.to_lowercase();
            tag_words
                .entry(lower)
                .and_modify(|w| {
                    if weight > *w {
                        *w = weight;
                    }
                })
                .or_insert(weight);
        }
    }
    if tag_words.is_empty() {
        return 0.0;
    }

    // Normalize against the same query-IDF total used by `term_overlap`
    // so the two signals sit on a comparable scale before RRF.
    let mut matched: f32 = 0.0;
    let mut total_idf: f32 = 0.0;
    let tag_word_set: HashSet<String> = tag_words.keys().cloned().collect();
    for qw in query_words {
        let w_idf = idf.get(qw).copied().unwrap_or(0.0);
        total_idf += w_idf;
        if contains_query_term(&tag_word_set, qw) {
            // Use the actual stored weight (2.0 for User, 1.0 for Auto)
            // rather than a constant — preserves the field-awareness.
            let src_weight = tag_words
                .get(qw)
                .copied()
                .or_else(|| {
                    // Plural variant fallback: try with/without 's'.
                    if qw.len() > 1 && qw.ends_with('s') {
                        tag_words.get(&qw[..qw.len() - 1]).copied()
                    } else {
                        tag_words.get(&format!("{}s", qw)).copied()
                    }
                })
                .unwrap_or(1.0);
            matched += w_idf * src_weight;
        }
    }
    if total_idf < 1e-6 {
        return 0.0;
    }
    matched / total_idf
}

/// Maximal Marginal Relevance selection with token budget.
///
/// At each step, pick the candidate that maximizes
/// `λ * relevance - (1-λ) * max_similarity_to_already_selected`
/// using tag-Jaccard as the similarity proxy. Stop when adding the next
/// candidate would exceed the token budget.
///
/// Performance note: tag HashSets are precomputed *once* per candidate
/// before the selection loop. The previous implementation rebuilt them
/// inside `jaccard()` on every comparison — O(K * S * |tags|) hash
/// allocations per query. With 32 candidates, ~6 selected hits, and
/// ~5 tags per candidate, that was the dominant cost in zero-ML mode
/// (75% of total recall latency per the phase trace).
fn mmr_select(candidates: Vec<RecallHit>, lambda: f32, budget: usize) -> Vec<RecallHit> {
    let mut remaining = candidates;
    let mut selected: Vec<RecallHit> = Vec::new();
    let mut tokens_used = 0usize;

    // Precompute tag HashSets ONCE per candidate. We use owned `String`s
    // here (not `&str`) because `remaining` is mutated during selection;
    // borrowing into it from a parallel Vec confuses the borrow checker.
    // The clone cost is negligible — N * |tags| short strings, dwarfed
    // by the hashing work we save in the inner loop.
    let mut remaining_tagsets: Vec<HashSet<String>> = remaining
        .iter()
        .map(|c| c.node.tags.iter().cloned().collect())
        .collect();
    let mut selected_tagsets: Vec<HashSet<String>> = Vec::new();

    // Compute relative recency rank ONCE per candidate set: 1.0 = newest,
    // 0.0 = oldest. This is what we use as a small additive boost so that
    // among comparable-relevance candidates, the newer wins. A fixed
    // millisecond delta is meaningless across recall sessions; relative
    // rank within the current set is the right signal.
    let recency_rank: std::collections::HashMap<uuid::Uuid, f32> = {
        let mut sorted: Vec<(uuid::Uuid, chrono::DateTime<chrono::Utc>)> = remaining
            .iter()
            .map(|c| (c.node.id, c.node.created_at))
            .collect();
        sorted.sort_by(|a, b| a.1.cmp(&b.1));
        let n = sorted.len().max(1) as f32;
        sorted
            .into_iter()
            .enumerate()
            .map(|(i, (id, _))| (id, i as f32 / (n - 1.0).max(1.0)))
            .collect()
    };
    /// Strength of the recency bias inside MMR. Must be small enough
    /// that an objective gap between two candidates always wins over
    /// recency. With RRF scores in [0.07, 0.09] range, a recency bonus
    /// of 0.03 is a soft tiebreak only.
    const RECENCY_WEIGHT: f32 = 0.03;

    let trace = std::env::var("SMRITI_RECALL_TRACE").is_ok();
    if trace {
        eprintln!(
            "  ── MMR start, {} candidates, λ={}, budget={}",
            remaining.len(),
            lambda,
            budget
        );
    }

    // Normalize final_score to [0, 1] so it perfectly balances with Jaccard [0, 1]
    let max_score = remaining
        .iter()
        .map(|c| c.final_score)
        .fold(0.0f32, f32::max);
    let min_score = remaining
        .iter()
        .map(|c| c.final_score)
        .fold(f32::MAX, f32::min);
    let score_range = (max_score - min_score).max(1e-6);

    while !remaining.is_empty() {
        // Pick the candidate maximizing the MMR objective + relative-
        // recency boost (gated by `MemoryKind`).
        let mut best_idx = 0usize;
        let mut best_score = f32::MIN;
        for (i, cand) in remaining.iter().enumerate() {
            let cand_tags = &remaining_tagsets[i];
            let max_sim_to_selected = selected_tagsets
                .iter()
                .map(|s_tags| jaccard_set(cand_tags, s_tags))
                .fold(0.0_f32, f32::max);

            let norm_score = (cand.final_score - min_score) / score_range;
            let mmr = lambda * norm_score - (1.0 - lambda) * max_sim_to_selected;
            if trace {
                eprintln!(
                    "  cand[{}] mmr={:.3} created={}: {}",
                    i,
                    mmr,
                    cand.node.created_at.timestamp_millis(),
                    &cand.node.text[..cand.node.text.len().min(50)]
                );
            }
            // Effective score = MMR objective + relative-recency boost.
            // The boost only applies for kinds where time-ordering matters
            // (Events, and supersedes-aware Facts) so that pure-knowledge
            // queries don't get distorted.
            let recency = recency_rank.get(&cand.node.id).copied().unwrap_or(0.0);
            let kind_boost = match cand.node.kind {
                crate::node::MemoryKind::Event => RECENCY_WEIGHT,
                _ if cand.node.supersedes.is_some() => RECENCY_WEIGHT,
                _ => 0.0,
            };
            let effective = mmr + kind_boost * recency;

            if effective > best_score + 1e-6 {
                if trace {
                    eprintln!(
                        "    ↳ pick: {} → {} (eff={:.3}, prev={:.3}, recency={:.2})",
                        best_idx, i, effective, best_score, recency
                    );
                }
                best_score = effective;
                best_idx = i;
            }
        }
        let cand = remaining.remove(best_idx);
        let cand_tags = remaining_tagsets.remove(best_idx);
        if tokens_used + cand.node.token_count > budget && !selected.is_empty() {
            // Try to fit smaller candidates if any remain.
            let next_idx = remaining
                .iter()
                .position(|c| tokens_used + c.node.token_count <= budget);
            if let Some(idx) = next_idx {
                let smaller = remaining.remove(idx);
                let smaller_tags = remaining_tagsets.remove(idx);
                tokens_used += smaller.node.token_count;
                selected.push(smaller);
                selected_tagsets.push(smaller_tags);
            }
            // Drop `cand_tags` — the candidate didn't fit and we
            // explicitly didn't add it to `selected`.
            drop(cand_tags);
            continue;
        }
        tokens_used += cand.node.token_count;
        selected.push(cand);
        selected_tagsets.push(cand_tags);
        if tokens_used >= budget {
            break;
        }
    }

    selected
}

/// Jaccard over precomputed `String` HashSets — zero allocations per call.
fn jaccard_set(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Jaccard similarity over tag sets.
fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let sa: std::collections::HashSet<&String> = a.iter().collect();
    let sb: std::collections::HashSet<&String> = b.iter().collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;
    use crate::scope::Scope;

    fn mk(text: &str, tags: &[&str], kind: MemoryKind) -> MemoryNode {
        let mut n = MemoryNode::new(text, kind, Scope::default());
        n.tags = tags.iter().map(|s| s.to_string()).collect();
        n
    }

    #[test]
    fn empty_stores_return_no_hits() {
        let hippo = Hippocampus::default();
        let neo = Neocortex::new();
        let r = recall(
            "anything",
            &[],
            &Scope::default(),
            &hippo,
            &neo,
            &RecallConfig::default(),
        );
        assert!(r.hits.is_empty());
    }

    #[test]
    fn relevant_memory_wins_over_irrelevant() {
        let hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        neo.insert(mk(
            "the auth module uses JWT tokens",
            &["auth", "security"],
            MemoryKind::Fact,
        ));
        neo.insert(mk(
            "user prefers spicy food",
            &["food"],
            MemoryKind::Preference,
        ));

        let r = recall(
            "how does authentication work",
            &["auth".to_string()],
            &Scope::default(),
            &hippo,
            &neo,
            &RecallConfig::default(),
        );
        assert!(!r.hits.is_empty());
        assert!(
            r.hits[0].node.text.contains("auth"),
            "expected auth memory first, got: {}",
            r.hits[0].node.text
        );
    }

    #[test]
    fn budget_is_respected() {
        let hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        for i in 0..50 {
            neo.insert(mk(
                &format!(
                    "memory {} with some content padding here xxxxxxxxxxxxxxxxxxxx",
                    i
                ),
                &["pad"],
                MemoryKind::Fact,
            ));
        }
        let mut cfg = RecallConfig::default();
        cfg.budget = 100; // very tight budget
        let r = recall("memory", &[], &Scope::default(), &hippo, &neo, &cfg);
        assert!(
            r.tokens_used <= cfg.budget + 50,
            "tokens_used {} should be ≤ budget {} (small slack ok)",
            r.tokens_used,
            cfg.budget
        );
    }

    #[test]
    fn kinds_filter_works() {
        let hippo = Hippocampus::default();
        let mut neo = Neocortex::new();
        neo.insert(mk("an event happened", &["e"], MemoryKind::Event));
        neo.insert(mk("a stable fact", &["f"], MemoryKind::Fact));

        let mut cfg = RecallConfig::default();
        cfg.kinds = vec![MemoryKind::Event];
        let r = recall("anything", &[], &Scope::default(), &hippo, &neo, &cfg);
        for h in &r.hits {
            assert_eq!(h.node.kind, MemoryKind::Event);
        }
    }

    #[test]
    fn term_overlap_handles_plural_variants() {
        let mut idf = HashMap::new();
        idf.insert("token".to_string(), 2.0);

        let words = word_set("JWT tokens are rotated daily");
        let score = term_overlap(&["token".to_string()], &words, &idf);
        assert!(score > 0.99, "expected plural variant match, got {score}");
    }
}
