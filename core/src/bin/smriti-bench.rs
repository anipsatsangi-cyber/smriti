//! Smriti benchmark — LongMemEval-style evaluation harness.
//!
//! This binary measures Smriti against the same five categories the
//! LongMemEval benchmark uses (Wu et al., 2024):
//!
//! 1. **Single-session-user** — recall a fact the user told you in one session.
//! 2. **Single-session-assistant** — recall something you told the user.
//! 3. **Multi-session** — fact spans multiple sessions; integrate.
//! 4. **Knowledge-update** — a fact was corrected; recall the latest.
//! 5. **Temporal** — "what did I say last Monday" — time-aware retrieval.
//!
//! For each category we measure:
//!
//! - **Hit rate** — was the gold answer in the recalled set?
//! - **Top-1 accuracy** — was the gold answer ranked #1?
//! - **Token efficiency** — fraction of budget actually used.
//! - **Latency** — wall-clock per query.
//!
//! Results are printed in a leaderboard-style table.
//!
//! Run with: `cargo run -p codegraph-memory --bench longmemeval --release`

use smriti::{MemoryKind, Smriti};
use std::time::Instant;

/// One evaluation case.
#[derive(Clone)]
struct EvalCase {
    category: &'static str,
    /// Memories to insert before the query.
    setup: Vec<(&'static str, MemoryKind, &'static [&'static str])>,
    /// Optional supersedes: index into `setup`, the new entry replaces it.
    supersede: Option<(usize, usize)>,
    /// The query text the agent issues at recall time.
    query: &'static str,
    /// Optional tags supplied with the query (allowed under the rules).
    query_tags: &'static [&'static str],
    /// The "gold" memory (substring that must appear in a hit).
    gold_substring: &'static str,
}

fn cases() -> Vec<EvalCase> {
    vec![
        // ── Single-session-user ──
        EvalCase {
            category: "single_user",
            setup: vec![
                (
                    "My favorite programming language is Rust",
                    MemoryKind::Preference,
                    &["lang", "preference"],
                ),
                (
                    "I work at Acme Corp as a backend engineer",
                    MemoryKind::Fact,
                    &["job", "company"],
                ),
                (
                    "Today I drank 4 cups of coffee",
                    MemoryKind::Event,
                    &["coffee"],
                ),
            ],
            supersede: None,
            query: "what is my favorite programming language",
            query_tags: &["lang"],
            gold_substring: "Rust",
        },
        EvalCase {
            category: "single_user",
            setup: vec![
                (
                    "I am allergic to peanuts",
                    MemoryKind::Fact,
                    &["health", "allergy"],
                ),
                ("My birthday is March 15th", MemoryKind::Fact, &["personal"]),
                (
                    "My phone number is 555-0123",
                    MemoryKind::Fact,
                    &["contact"],
                ),
            ],
            supersede: None,
            query: "what allergies do I have",
            query_tags: &["health"],
            gold_substring: "peanuts",
        },
        // ── Single-session-assistant ──
        EvalCase {
            category: "single_assistant",
            setup: vec![
                (
                    "I recommended the user try a Mediterranean diet",
                    MemoryKind::Decision,
                    &["recommendation", "diet"],
                ),
                (
                    "User asked about productivity tips earlier",
                    MemoryKind::Event,
                    &["productivity"],
                ),
                (
                    "I shared three book recommendations: Sapiens, Atomic Habits, Range",
                    MemoryKind::Decision,
                    &["books", "recommendation"],
                ),
            ],
            supersede: None,
            query: "what diet did I recommend",
            query_tags: &["diet"],
            gold_substring: "Mediterranean",
        },
        // ── Multi-session ──
        EvalCase {
            category: "multi_session",
            setup: vec![
                (
                    "Project Phoenix kicked off in January 2026",
                    MemoryKind::Event,
                    &["project", "phoenix"],
                ),
                (
                    "Phoenix uses Rust and Tokio for the backend",
                    MemoryKind::Fact,
                    &["project", "phoenix", "tech"],
                ),
                (
                    "Phoenix MVP shipped in March 2026",
                    MemoryKind::Event,
                    &["project", "phoenix"],
                ),
                (
                    "Total budget for Phoenix is $500K",
                    MemoryKind::Fact,
                    &["project", "phoenix", "budget"],
                ),
            ],
            supersede: None,
            query: "tell me about Project Phoenix",
            query_tags: &["phoenix"],
            gold_substring: "Phoenix",
        },
        EvalCase {
            category: "multi_session",
            setup: vec![
                (
                    "Alice joined the team as a senior engineer",
                    MemoryKind::Event,
                    &["team", "alice"],
                ),
                (
                    "Alice prefers async communication over meetings",
                    MemoryKind::Preference,
                    &["alice", "style"],
                ),
                (
                    "Alice is leading the auth refactor project",
                    MemoryKind::Fact,
                    &["alice", "project"],
                ),
            ],
            supersede: None,
            query: "what is Alice working on",
            query_tags: &["alice"],
            gold_substring: "auth",
        },
        // ── Knowledge update (supersedes) ──
        EvalCase {
            category: "knowledge_update",
            setup: vec![
                (
                    "My address is 123 Main Street, Springfield",
                    MemoryKind::Fact,
                    &["address", "personal"],
                ),
                (
                    "I just moved to 456 Oak Avenue, Portland",
                    MemoryKind::Fact,
                    &["address", "personal"],
                ),
            ],
            supersede: Some((0, 1)),
            query: "what is my current address",
            query_tags: &["address"],
            gold_substring: "Oak Avenue",
        },
        EvalCase {
            category: "knowledge_update",
            setup: vec![
                (
                    "The user's preferred language is Python",
                    MemoryKind::Preference,
                    &["lang", "preference"],
                ),
                (
                    "Actually, the user's preferred language is now Rust",
                    MemoryKind::Preference,
                    &["lang", "preference"],
                ),
            ],
            supersede: Some((0, 1)),
            query: "what programming language does the user prefer",
            query_tags: &["lang"],
            gold_substring: "Rust",
        },
        // ── Temporal ──
        EvalCase {
            category: "temporal",
            setup: vec![
                (
                    "Deployed v1.0 to production",
                    MemoryKind::Event,
                    &["deploy", "release"],
                ),
                (
                    "Hotfixed null pointer crash in auth",
                    MemoryKind::Event,
                    &["bug", "auth"],
                ),
                (
                    "Released v2.0 with new dashboard",
                    MemoryKind::Event,
                    &["deploy", "release"],
                ),
            ],
            supersede: None,
            query: "what did we recently deploy",
            query_tags: &["deploy"],
            gold_substring: "v2.0",
        },
        EvalCase {
            category: "temporal",
            setup: vec![
                (
                    "Met with VP of Engineering on Monday",
                    MemoryKind::Event,
                    &["meeting"],
                ),
                ("Wrote OKRs for Q2 on Tuesday", MemoryKind::Event, &["okrs"]),
                (
                    "Did interview prep on Friday",
                    MemoryKind::Event,
                    &["interview"],
                ),
            ],
            supersede: None,
            query: "what did I do for OKRs",
            query_tags: &["okrs"],
            gold_substring: "OKRs",
        },
    ]
}

#[derive(Default)]
struct CategoryResult {
    total: usize,
    hits: usize,
    top1: usize,
    total_tokens_used: usize,
    total_tokens_budget: usize,
    total_latency_us: u128,
}

fn run_case_verbose(case: &EvalCase, verbose: bool) -> (bool, bool, usize, usize, u128) {
    let mut s = Smriti::open(":memory:").expect("smriti open");

    // Optionally turn on the real fastembed-rs (MiniLM-L6-v2 quantized) layer.
    // First call downloads the model (~50 MB); subsequent calls reuse the
    // local cache. Gated on env var so we can compare zero-ML vs embedded.
    #[cfg(feature = "embeddings")]
    if std::env::var("SMRITI_BENCH_EMBEDDINGS").is_ok() {
        s.enable_embeddings().expect("enable embeddings");
    }

    // Insert memories with ids tracked so we can supersede. Sleep tiny
    // amounts between inserts so created_at gives a reliable ordering.
    let mut ids: Vec<uuid::Uuid> = Vec::new();
    for (text, kind, tags) in &case.setup {
        let id = s
            .remember(*text)
            .kind(*kind)
            .tags(tags.iter().copied())
            .commit()
            .expect("remember");
        ids.push(id);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // Apply supersedes if any.
    if let Some((old, new)) = case.supersede {
        s.supersede(ids[old], ids[new]).expect("supersede");
    }

    s.consolidate().expect("consolidate");

    // Run the query.
    let started = Instant::now();
    let result = s
        .recall(case.query)
        .budget(500)
        .tags(case.query_tags.iter().copied())
        .execute()
        .expect("recall");
    let latency_us = started.elapsed().as_micros();

    // Score.
    let hit = result
        .hits
        .iter()
        .any(|h| h.node.text.contains(case.gold_substring));
    let top1 = result
        .hits
        .first()
        .map(|h| h.node.text.contains(case.gold_substring))
        .unwrap_or(false);

    if verbose && !top1 {
        eprintln!("\n  Q[{}]: {}", case.category, case.query);
        eprintln!("  GOLD: {}", case.gold_substring);
        for (i, h) in result.hits.iter().take(3).enumerate() {
            eprintln!(
                "    {}. [{:.3}] {}",
                i + 1,
                h.final_score,
                &h.node.text[..h.node.text.len().min(80)]
            );
        }
    }

    (
        hit,
        top1,
        result.tokens_used,
        result.tokens_budget,
        latency_us,
    )
}

fn run_case(case: &EvalCase) -> (bool, bool, usize, usize, u128) {
    let verbose = std::env::var("SMRITI_BENCH_VERBOSE").is_ok();
    run_case_verbose(case, verbose)
}

fn main() {
    let cases = cases();
    let mut by_category: std::collections::BTreeMap<&'static str, CategoryResult> =
        std::collections::BTreeMap::new();

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Smriti Benchmark — LongMemEval-style evaluation                      ║");
    println!("║  स्मृति · structured memory engine                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    for case in &cases {
        let (hit, top1, tu, tb, lat) = run_case(case);
        let entry = by_category.entry(case.category).or_default();
        entry.total += 1;
        if hit {
            entry.hits += 1;
        }
        if top1 {
            entry.top1 += 1;
        }
        entry.total_tokens_used += tu;
        entry.total_tokens_budget += tb;
        entry.total_latency_us += lat;
    }

    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>14} {:>10}",
        "Category", "Cases", "Hit %", "Top-1 %", "Avg tok used", "Avg µs"
    );
    println!("{}", "─".repeat(80));

    let mut grand_total = 0;
    let mut grand_hits = 0;
    let mut grand_top1 = 0;
    let mut grand_tu = 0;
    let mut grand_tb = 0;
    let mut grand_lat = 0u128;

    for (cat, r) in &by_category {
        let hit_pct = 100.0 * r.hits as f32 / r.total as f32;
        let top1_pct = 100.0 * r.top1 as f32 / r.total as f32;
        let avg_tu = r.total_tokens_used as f32 / r.total as f32;
        let avg_lat = r.total_latency_us / r.total as u128;
        println!(
            "{:<22} {:>10} {:>9.1}% {:>9.1}% {:>14.0} {:>10}",
            cat, r.total, hit_pct, top1_pct, avg_tu, avg_lat
        );
        grand_total += r.total;
        grand_hits += r.hits;
        grand_top1 += r.top1;
        grand_tu += r.total_tokens_used;
        grand_tb += r.total_tokens_budget;
        grand_lat += r.total_latency_us;
    }

    println!("{}", "─".repeat(80));
    let overall_hit = 100.0 * grand_hits as f32 / grand_total as f32;
    let overall_top1 = 100.0 * grand_top1 as f32 / grand_total as f32;
    let avg_tu = grand_tu as f32 / grand_total as f32;
    let efficiency = 100.0 * (1.0 - grand_tu as f32 / grand_tb as f32);
    let avg_lat = grand_lat / grand_total as u128;
    println!(
        "{:<22} {:>10} {:>9.1}% {:>9.1}% {:>14.0} {:>10}",
        "OVERALL", grand_total, overall_hit, overall_top1, avg_tu, avg_lat
    );
    println!();
    println!("┌─ Token efficiency ──────────────────────────────────────────────┐");
    println!(
        "│  Used {} / {} budget tokens ({:.1}% headroom remaining)            │",
        grand_tu, grand_tb, efficiency
    );
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    let mode = if cfg!(feature = "embeddings") && std::env::var("SMRITI_BENCH_EMBEDDINGS").is_ok() {
        "with fastembed-rs MiniLM-L6-v2 (quantized, 384-d)"
    } else {
        "zero-ML (no embeddings)"
    };
    println!("Configuration: {}", mode);
    println!("Build: $(rustc --version | cut -d' ' -f2)");

    // Exit non-zero if hit rate is below a sanity floor — useful for CI.
    if overall_hit < 50.0 {
        eprintln!(
            "\n⚠️  Overall hit rate {:.1}% is below the 50% sanity floor",
            overall_hit
        );
        std::process::exit(2);
    }
}
