//! Continuity benchmark — measures the lift from spreading-activation
//! priming when a sequence of queries is correlated (the real-agent case)
//! vs. when each query is treated as independent (the unit-test case).
//!
//! # Why this benchmark exists
//!
//! `smriti-bench-500` fires 47 unrelated questions against the same
//! corpus. There is no conversation. Each question is an island. In
//! that environment, priming is strictly noise — it can only
//! contaminate, it cannot help, because there's no signal of continuity
//! to amplify.
//!
//! That's an artifact of the bench design, not of the real world. In a
//! real agent loop the user typically asks several correlated questions
//! in a row: debugging an outage, planning a trip, reading a paper.
//! Smriti's persistent activation map is supposed to make those follow-
//! up questions cheaper and more accurate by carrying forward the
//! relevant subgraph from earlier queries.
//!
//! This bench measures that lift directly. We define 5 themed bursts of
//! 5 correlated queries each (25 total) and run them in two configs:
//!
//!   * **Cold**: `clear_activation()` before *every* query. Stateless.
//!   * **Primed**: `clear_activation()` only *between* bursts. Real-loop
//!     behavior — within a thread, priming carries forward; across
//!     unrelated topics, the agent (or harness) explicitly resets.
//!
//! The reported delta — primed_hit% − cold_hit% — is the honest answer
//! to "does cross-query priming work?". Hypothesis: lift starts at
//! ~0 for query 1 of each burst (no priming has built up yet) and
//! grows through queries 2-5 as residual activation seeds related
//! memories.
//!
//! # Substring eval caveat
//!
//! We use the same gold-substring metric as bench-500. It's brittle in
//! absolute terms (substring match is too strict on paraphrase, too
//! loose on short numerics). But this bench reports a *delta* between
//! two configs against the same metric — so the noise mostly cancels.
//! If the delta is positive across burst positions, priming works. If
//! it's flat, the architectural feature isn't earning its place.

use smriti::{MemoryKind, Smriti};
use std::time::Instant;

/// One themed burst — 5 correlated questions about the same topic.
struct Burst {
    topic: &'static str,
    /// Five `(query, gold_substring)` pairs in order. Order matters:
    /// the priming benefit accumulates across the sequence.
    queries: [(&'static str, &'static str); 5],
}

fn corpus() -> Vec<(&'static str, MemoryKind, &'static [&'static str])> {
    let mut base: Vec<(&'static str, MemoryKind, &'static [&'static str])> = signal_corpus();
    // Add procedural noise so cold queries have to fight through it. The
    // cold/primed delta is most visible when neither config gets 100%
    // — noise is what creates the headroom for priming to help.
    for i in 0..150 {
        let text = Box::leak(format!(
            "System log entry {}: routine background worker handled queue items relating \
             to scaling and operational maintenance. Generic user {} updated preferences via API.",
            i, i % 50
        ).into_boxed_str());
        let tags: &'static [&'static str] =
            Box::leak(vec!["noise", "synthetic"].into_boxed_slice());
        base.push((text, MemoryKind::Event, tags));
    }
    base
}

fn signal_corpus() -> Vec<(&'static str, MemoryKind, &'static [&'static str])> {
    vec![
        // ─── Auth ──────────────────────────────────────────────────────
        (
            "Authentication uses JWT with RS256 keypairs and 1-hour expiry",
            MemoryKind::Fact,
            &["auth", "security", "jwt"][..],
        ),
        (
            "Sessions are stored in Redis with an 8-hour TTL",
            MemoryKind::Fact,
            &["auth", "session", "redis"][..],
        ),
        (
            "Refresh tokens rotate every 7 days via the /refresh endpoint",
            MemoryKind::Fact,
            &["auth", "refresh"][..],
        ),
        (
            "Password hashing uses Argon2id with memory cost 64 MB",
            MemoryKind::Fact,
            &["auth", "password", "security"][..],
        ),
        (
            "MFA via TOTP is mandatory for admin users",
            MemoryKind::Decision,
            &["auth", "mfa", "policy"][..],
        ),
        // ─── Database ─────────────────────────────────────────────────
        (
            "Primary database is Postgres 15 with three read replicas",
            MemoryKind::Fact,
            &["db", "postgres", "infra"][..],
        ),
        (
            "Connection pooling via PgBouncer in transaction mode",
            MemoryKind::Fact,
            &["db", "postgres", "pooling"][..],
        ),
        (
            "Database backups run every 4 hours to S3 with 30-day retention",
            MemoryKind::Fact,
            &["db", "backup"][..],
        ),
        (
            "We migrated from MongoDB to Postgres in Q3 2025 for ACID guarantees",
            MemoryKind::Decision,
            &["db", "decision", "migration"][..],
        ),
        (
            "Slow-query log threshold is 200 ms; alerts go to #db-alerts",
            MemoryKind::Fact,
            &["db", "monitoring"][..],
        ),
        // ─── Backend ──────────────────────────────────────────────────
        (
            "Backend is Rust with the Axum framework on Tokio runtime",
            MemoryKind::Fact,
            &["backend", "rust", "axum"][..],
        ),
        (
            "We chose Rust for memory safety and predictable latency",
            MemoryKind::Decision,
            &["backend", "rust", "decision"][..],
        ),
        (
            "Error handling uses thiserror for libraries and anyhow for binaries",
            MemoryKind::Decision,
            &["backend", "rust", "errors"][..],
        ),
        (
            "Tracing uses opentelemetry-rust with traces shipped to Jaeger",
            MemoryKind::Fact,
            &["backend", "observability"][..],
        ),
        (
            "Graceful shutdown waits 30 seconds for in-flight requests",
            MemoryKind::Fact,
            &["backend", "deploy"][..],
        ),
        // ─── Frontend ─────────────────────────────────────────────────
        (
            "Frontend is a React 18 SPA with TypeScript strict mode",
            MemoryKind::Fact,
            &["frontend", "react", "typescript"][..],
        ),
        (
            "State management uses Zustand instead of Redux",
            MemoryKind::Decision,
            &["frontend", "state"][..],
        ),
        (
            "Build tool is Vite with esbuild for production bundles",
            MemoryKind::Fact,
            &["frontend", "build", "vite"][..],
        ),
        (
            "Component styling uses Tailwind CSS with a custom design-token theme",
            MemoryKind::Fact,
            &["frontend", "css"][..],
        ),
        (
            "End-to-end tests run via Playwright in CI on every PR",
            MemoryKind::Fact,
            &["frontend", "testing"][..],
        ),
        // ─── Infra ────────────────────────────────────────────────────
        (
            "Production runs on AWS us-west-2 with active-passive failover to us-east-1",
            MemoryKind::Fact,
            &["infra", "aws"][..],
        ),
        (
            "Kubernetes version is 1.29; we upgrade one minor every quarter",
            MemoryKind::Fact,
            &["infra", "k8s"][..],
        ),
        (
            "Cluster autoscaling targets 70% CPU; min 3 nodes, max 50",
            MemoryKind::Fact,
            &["infra", "scaling"][..],
        ),
        (
            "Secrets live in AWS Secrets Manager; rotation every 90 days",
            MemoryKind::Fact,
            &["infra", "secrets"][..],
        ),
        (
            "We use Linkerd as the service mesh for mTLS between services",
            MemoryKind::Fact,
            &["infra", "mesh"][..],
        ),
    ]
}

fn bursts() -> Vec<Burst> {
    // Each burst follows the same pattern:
    //   - Q1 is a *direct* keyword question about a memory in the cluster.
    //     This grounds the activation map in the right subgraph.
    //   - Q2-Q5 are increasingly *paraphrase-heavy* — they share the
    //     subgraph but use different vocabulary than the gold memory.
    //     A cold engine has nothing to lean on; a primed engine carries
    //     the subgraph forward and can find the right memory through
    //     graph proximity rather than keyword overlap.
    vec![
        Burst {
            topic: "auth",
            queries: [
                // Q1 grounds: "auth" + "JWT" both lexically present.
                ("how does authentication work in this system", "JWT"),
                // Q2-Q5: paraphrase. None of the gold strings appear
                // in the question text.
                ("where do user logins live in memory", "Redis"),
                ("how are credentials renewed without re-login", "/refresh"),
                ("how is the user secret encoded at rest", "Argon2id"),
                ("what extra step do operators need beyond a password", "TOTP"),
            ],
        },
        Burst {
            topic: "db",
            queries: [
                ("what database do we use", "Postgres"),
                ("how is the wire-level traffic to it managed", "PgBouncer"),
                ("when does the data get copied off the primary", "4 hours"),
                ("why did we leave the previous data store", "MongoDB"),
                ("when do operators get paged for sluggish queries", "200 ms"),
            ],
        },
        Burst {
            topic: "backend",
            queries: [
                ("what is the backend stack", "Axum"),
                ("why is this the language of choice for our services", "memory safety"),
                ("how do library failures surface vs binary failures", "thiserror"),
                ("how do operators correlate spans across requests", "opentelemetry"),
                ("how do running requests survive a redeploy", "30 seconds"),
            ],
        },
        Burst {
            topic: "frontend",
            queries: [
                ("what is the frontend framework", "React 18"),
                ("how do components share data without redux boilerplate", "Zustand"),
                ("how is the production js bundle assembled", "Vite"),
                ("how are visual styles applied across the app", "Tailwind"),
                ("how do we exercise the UI from the user's perspective", "Playwright"),
            ],
        },
        Burst {
            topic: "infra",
            queries: [
                ("where does production run", "us-west-2"),
                ("what container orchestration version are we on", "1.29"),
                ("when does the cluster decide to add capacity", "70%"),
                ("where do credentials get stored centrally", "Secrets Manager"),
                ("how is encrypted traffic enforced between services", "Linkerd"),
            ],
        },
    ]
}

#[derive(Default, Clone)]
struct PositionStats {
    hits: usize,
    total: usize,
    tokens: usize,
    latency_us: u128,
}

fn run_one(
    s: &mut Smriti,
    bursts: &[Burst],
    cold: bool,
) -> [PositionStats; 5] {
    let mut by_pos: [PositionStats; 5] = Default::default();
    for burst in bursts {
        if !cold {
            // Primed mode: clear once between bursts so the next topic
            // starts stateless, but within the burst priming carries.
            s.clear_activation();
        }
        for (i, (q, gold)) in burst.queries.iter().enumerate() {
            if cold {
                // Cold mode: every query starts stateless.
                s.clear_activation();
            }
            let started = Instant::now();
            let result = s
                .recall(*q)
                .budget(500)
                .execute()
                .expect("recall");
            let lat = started.elapsed().as_micros();
            let hit = result
                .hits
                .iter()
                .any(|h| h.node.text.contains(gold));
            by_pos[i].total += 1;
            if hit {
                by_pos[i].hits += 1;
            }
            by_pos[i].tokens += result.tokens_used;
            by_pos[i].latency_us += lat;
        }
    }
    by_pos
}

fn pct(p: &PositionStats) -> f32 {
    if p.total == 0 {
        0.0
    } else {
        100.0 * p.hits as f32 / p.total as f32
    }
}

fn avg_tokens(p: &PositionStats) -> f32 {
    if p.total == 0 {
        0.0
    } else {
        p.tokens as f32 / p.total as f32
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Smriti Continuity Benchmark — does priming help follow-up queries?    ║");
    println!("║  स्मृति · 5 themed bursts × 5 correlated queries · cold vs primed      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Build corpus once, then reuse the same Smriti instance for both
    // runs (clearing the activation state before each run gives them
    // identical starting states; the corpus, neocortex graph, and
    // fingerprints are unchanged).
    let mut s = Smriti::open(":memory:").expect("smriti open");
    let load_start = Instant::now();
    for (text, kind, tags) in corpus() {
        s.remember(text)
            .kind(kind)
            .tags(tags.iter().copied())
            .commit()
            .expect("remember");
    }
    s.consolidate().expect("consolidate");
    let load_ms = load_start.elapsed().as_millis();
    let stats = s.stats().expect("stats");
    println!(
        "Corpus: {} memories ({} tokens stored), loaded in {} ms",
        stats.store.total_memories, stats.store.total_tokens, load_ms
    );
    println!();

    let bursts = bursts();

    // ── Cold pass ──
    s.clear_activation();
    let cold = run_one(&mut s, &bursts, true);
    // ── Primed pass ──
    s.clear_activation();
    let primed = run_one(&mut s, &bursts, false);

    // ── Report ──
    println!("Per-query-position results (averaged across 5 bursts):");
    println!();
    println!(
        "{:>9}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "position", "cold hit%", "primed hit%", "lift", "cold tok", "primed tok"
    );
    println!("{}", "─".repeat(72));
    let mut total_cold = 0.0f32;
    let mut total_primed = 0.0f32;
    for i in 0..5 {
        let c = pct(&cold[i]);
        let p = pct(&primed[i]);
        let lift = p - c;
        let ct = avg_tokens(&cold[i]);
        let pt = avg_tokens(&primed[i]);
        println!(
            "{:>9}  {:>9.1}%  {:>10.1}%  {:>+7.1}  {:>10.1}  {:>10.1}",
            i + 1,
            c,
            p,
            lift,
            ct,
            pt
        );
        total_cold += c;
        total_primed += p;
    }
    println!("{}", "─".repeat(72));
    println!(
        "{:>9}  {:>9.1}%  {:>10.1}%  {:>+7.1}",
        "overall",
        total_cold / 5.0,
        total_primed / 5.0,
        (total_primed - total_cold) / 5.0
    );

    println!();
    println!("Interpretation:");
    println!("  - Position 1 lift should be ~0 — no priming has built up yet.");
    println!("  - Positions 2-5 lift > 0 means cross-query priming actually helps.");
    println!("  - Negative lift means residual activation is hurting (mis-tuned");
    println!("    decay/gain) — the bench-500 regression we just fixed was a case");
    println!("    of this happening at scale.");
    println!();
    println!("Reproduce:  cargo run --bin smriti-bench-continuity --release");
}
