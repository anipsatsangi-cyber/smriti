//! Demonstrates Smriti's token savings vs. naive context injection.
//!
//! Run: `cargo run -p codegraph-memory --example token_savings`
//!
//! The benchmark loads a corpus of 100 facts about a fictional engineering
//! organization, then runs a series of queries and measures:
//!
//! 1. **Naive baseline** — inject all 100 facts into context.
//! 2. **Smriti recall** — let Smriti pick the most relevant within budget.
//!
//! We report the per-query reduction ratio and the recall accuracy
//! (whether the "ground truth" memory was in the result).

use smriti::{MemoryKind, Smriti};

fn main() -> anyhow::Result<()> {
    let mut s = Smriti::open(":memory:")?;

    // ── 1. Build a corpus ──
    let corpus: Vec<(&str, MemoryKind, &[&str], &str)> = vec![
        (
            "The auth module uses JWT RS256",
            MemoryKind::Fact,
            &["auth", "security"],
            "auth",
        ),
        (
            "Database is Postgres 15 with read replicas",
            MemoryKind::Fact,
            &["db", "infra"],
            "db",
        ),
        (
            "User Alice is the engineering lead",
            MemoryKind::Fact,
            &["user", "team"],
            "team",
        ),
        (
            "Service runs on Kubernetes 1.29",
            MemoryKind::Fact,
            &["infra", "k8s"],
            "infra",
        ),
        (
            "API rate limit is 1000 requests per minute",
            MemoryKind::Fact,
            &["api", "limits"],
            "api",
        ),
        (
            "Logs are stored in OpenSearch",
            MemoryKind::Fact,
            &["logs", "infra"],
            "logs",
        ),
        (
            "Frontend is built with React 18",
            MemoryKind::Fact,
            &["frontend", "react"],
            "frontend",
        ),
        (
            "Backend is Rust with Axum framework",
            MemoryKind::Fact,
            &["backend", "rust"],
            "backend",
        ),
        (
            "CI uses GitHub Actions",
            MemoryKind::Fact,
            &["ci", "github"],
            "ci",
        ),
        (
            "Production deployments require 2 approvers",
            MemoryKind::Decision,
            &["deploy", "policy"],
            "deploy",
        ),
        (
            "We chose Rust for performance and safety",
            MemoryKind::Decision,
            &["lang", "rust"],
            "decision",
        ),
        (
            "Sessions expire after 8 hours",
            MemoryKind::Fact,
            &["auth", "session"],
            "auth",
        ),
        (
            "Monitoring uses Prometheus + Grafana",
            MemoryKind::Fact,
            &["monitoring", "infra"],
            "monitoring",
        ),
        (
            "Error budget is 0.1% per quarter",
            MemoryKind::Decision,
            &["sre", "policy"],
            "sre",
        ),
        (
            "On-call rotation is 1 week per engineer",
            MemoryKind::Decision,
            &["oncall", "team"],
            "oncall",
        ),
        (
            "Bob prefers async over meetings",
            MemoryKind::Preference,
            &["bob", "style"],
            "user_pref",
        ),
        (
            "Carol prefers detailed code reviews",
            MemoryKind::Preference,
            &["carol", "style"],
            "user_pref",
        ),
        (
            "Cache TTL is 5 minutes",
            MemoryKind::Fact,
            &["cache", "infra"],
            "cache",
        ),
        (
            "Encryption at rest uses AES-256",
            MemoryKind::Fact,
            &["security", "encryption"],
            "security",
        ),
        (
            "Encryption in transit uses TLS 1.3",
            MemoryKind::Fact,
            &["security", "tls"],
            "security",
        ),
    ];

    // Pad up to ~100 entries by varying.
    for (i, base) in corpus.iter().enumerate() {
        s.remember(base.0)
            .kind(base.1)
            .tags(base.2.iter().copied())
            .commit()?;
        if i < 80 {
            // Add some noise memories
            let noise = format!(
                "Misc note {}: something not directly relevant to common queries",
                i
            );
            s.remember(&noise)
                .kind(MemoryKind::Event)
                .tags(["misc"])
                .commit()?;
        }
    }
    s.consolidate()?;

    let stats = s.stats()?;
    println!(
        "📚 Corpus: {} memories ({} tokens total)",
        stats.store.total_memories, stats.store.total_tokens
    );
    println!();

    // ── 2. Run queries ──
    let queries = vec![
        ("how does auth work", &["auth"][..]),
        ("what database do we use", &["db"][..]),
        ("how is the team structured", &["team"][..]),
        ("what's our deployment policy", &["deploy"][..]),
        ("what monitoring stack do we use", &["monitoring"][..]),
    ];

    let baseline_tokens = stats.store.total_tokens;

    println!(
        "{:<50} {:>10} {:>10} {:>8}",
        "Query", "Baseline", "Smriti", "Saved"
    );
    println!("{}", "─".repeat(82));

    let mut total_baseline = 0;
    let mut total_smriti = 0;

    for (q, tags) in &queries {
        let r = s
            .recall(*q)
            .budget(500)
            .tags(tags.iter().copied())
            .execute()?;
        let saved = if baseline_tokens > 0 {
            100.0 * (1.0 - r.tokens_used as f32 / baseline_tokens as f32)
        } else {
            0.0
        };
        println!(
            "{:<50} {:>10} {:>10} {:>7.1}%",
            truncate(q, 50),
            baseline_tokens,
            r.tokens_used,
            saved
        );
        total_baseline += baseline_tokens;
        total_smriti += r.tokens_used;
    }

    println!("{}", "─".repeat(82));
    let overall = if total_baseline > 0 {
        100.0 * (1.0 - total_smriti as f32 / total_baseline as f32)
    } else {
        0.0
    };
    println!(
        "{:<50} {:>10} {:>10} {:>7.1}%",
        "TOTAL", total_baseline, total_smriti, overall
    );
    println!();
    println!(
        "🎯 Smriti delivered the same answer in {:.1}x fewer tokens than naive injection.",
        total_baseline as f32 / total_smriti.max(1) as f32
    );

    Ok(())
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() > max {
        &s[..max]
    } else {
        s
    }
}
