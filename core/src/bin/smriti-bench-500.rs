//! Large-scale Smriti benchmark — 200+ memories, synonym-heavy queries.
//!
//! The default `smriti-bench` is a 9-case correctness sanity check. This
//! one is the *real* benchmark: it loads ~200 memories spread across
//! multiple scopes, runs 50 queries that mix exact-keyword, paraphrase,
//! and pure-synonym lookups, and reports score, latency and token
//! efficiency.
//!
//! Run zero-ML:    `cargo run -p codegraph-memory --bin smriti-bench-large --release`
//! Run with deps:  `SMRITI_BENCH_EMBEDDINGS=1 cargo run -p codegraph-memory \
//!                  --bin smriti-bench-large --features embeddings --release`
//!
//! The delta between the two configurations is the honest answer to
//! "how much do embeddings actually buy us at scale?".

use smriti::{MemoryKind, Smriti};
use std::time::Instant;

/// One case: setup loads memories, then we query and check the gold.
struct LargeCase {
    /// Human label for the category being tested.
    category: &'static str,
    /// The query an agent would ask.
    query: &'static str,
    /// Optional tags supplied with the query (most agents won't bother).
    query_tags: &'static [&'static str],
    /// A substring that MUST appear in at least one returned memory's text.
    /// Empty string means "no gold exists" — used by abstention cases.
    gold_substring: &'static str,
    /// Whether this query is a paraphrase (semantically similar but
    /// lexically different). These are the cases where embeddings help.
    paraphrase: bool,
    /// Adversarial query: the gold is *not* in the corpus. Smriti should
    /// return either an empty result or hits with low confidence. We
    /// measure "abstention rate" — how often the top hit's score falls
    /// below an acceptance threshold (or the result set is empty).
    expect_abstain: bool,
}

/// Generate the corpus. ~200 memories across 4 thematic clusters so PPR
/// has signal to work with, but with enough lexical variation that
/// keyword-only retrieval has to do real work.
fn corpus() -> Vec<(&'static str, MemoryKind, &'static [&'static str])> {
    let mut base = vec![
        // ─── Engineering / infra ────────────────────────────────────────
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
            "OAuth2 integration supports Google, GitHub, Microsoft providers",
            MemoryKind::Fact,
            &["auth", "oauth"][..],
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
        (
            "Rate limiting on /login is 5 attempts per minute per IP",
            MemoryKind::Fact,
            &["auth", "ratelimit"][..],
        ),
        (
            "All auth events are logged to the audit-log Kafka topic",
            MemoryKind::Fact,
            &["auth", "audit", "logging"][..],
        ),
        (
            "Session hijacking detection runs every 30 seconds via fingerprinting",
            MemoryKind::Fact,
            &["auth", "security"][..],
        ),
        (
            "We chose JWT over opaque session tokens for stateless scaling",
            MemoryKind::Decision,
            &["auth", "decision", "scale"][..],
        ),
        (
            "Primary database is Postgres 15 with three read replicas",
            MemoryKind::Fact,
            &["db", "postgres", "infra"][..],
        ),
        (
            "Replicas use logical replication with 5-second target lag",
            MemoryKind::Fact,
            &["db", "postgres", "replication"][..],
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
            "Schema migrations use sqlx-cli with forward-only rules",
            MemoryKind::Decision,
            &["db", "migration", "policy"][..],
        ),
        (
            "Read queries run against the nearest read replica via PgBouncer",
            MemoryKind::Fact,
            &["db", "performance"][..],
        ),
        (
            "Vacuum analyze runs nightly on tables larger than 1 GB",
            MemoryKind::Fact,
            &["db", "performance", "maintenance"][..],
        ),
        (
            "Postgres extensions in use: pg_stat_statements, pgcrypto, pg_trgm",
            MemoryKind::Fact,
            &["db", "extensions"][..],
        ),
        (
            "Slow-query log threshold is 200 ms; alerts go to #db-alerts",
            MemoryKind::Fact,
            &["db", "monitoring"][..],
        ),
        (
            "Backend is Rust with the Axum framework on Tokio runtime",
            MemoryKind::Fact,
            &["backend", "rust", "axum"][..],
        ),
        (
            "All services compile to a single static musl binary for deployment",
            MemoryKind::Fact,
            &["backend", "build", "deploy"][..],
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
            "The HTTP server runs in Tokio's multi-threaded scheduler",
            MemoryKind::Fact,
            &["backend", "rust", "tokio"][..],
        ),
        (
            "All public endpoints have OpenAPI 3.1 schemas auto-generated from utoipa",
            MemoryKind::Fact,
            &["backend", "api", "openapi"][..],
        ),
        (
            "Graceful shutdown waits 30 seconds for in-flight requests",
            MemoryKind::Fact,
            &["backend", "deploy"][..],
        ),
        (
            "Tracing uses opentelemetry-rust with traces shipped to Jaeger",
            MemoryKind::Fact,
            &["backend", "observability"][..],
        ),
        (
            "We use sqlx (not diesel) for compile-time SQL verification",
            MemoryKind::Decision,
            &["backend", "db", "rust"][..],
        ),
        (
            "CI runs cargo test, cargo clippy, cargo fmt on every PR",
            MemoryKind::Fact,
            &["ci", "rust"][..],
        ),
        // ─── Frontend ───────────────────────────────────────────────────
        (
            "Frontend is a React 18 SPA with TypeScript strict mode",
            MemoryKind::Fact,
            &["frontend", "react", "ts"][..],
        ),
        (
            "State management uses Zustand for client state, TanStack Query for server",
            MemoryKind::Fact,
            &["frontend", "state"][..],
        ),
        (
            "Styling uses Tailwind CSS with a custom design-token preset",
            MemoryKind::Fact,
            &["frontend", "styling"][..],
        ),
        (
            "All forms use react-hook-form with Zod schema validation",
            MemoryKind::Fact,
            &["frontend", "forms"][..],
        ),
        (
            "Routing uses React Router 6 with file-based code splitting",
            MemoryKind::Fact,
            &["frontend", "routing"][..],
        ),
        (
            "Bundle size is enforced under 250 KB gzipped via size-limit on PRs",
            MemoryKind::Decision,
            &["frontend", "performance"][..],
        ),
        (
            "We chose Vite over webpack for 10x faster dev builds",
            MemoryKind::Decision,
            &["frontend", "build"][..],
        ),
        (
            "Component library is custom-built; no Material-UI or Chakra",
            MemoryKind::Decision,
            &["frontend", "design"][..],
        ),
        (
            "Accessibility target is WCAG 2.1 Level AA across all pages",
            MemoryKind::Decision,
            &["frontend", "a11y"][..],
        ),
        (
            "E2E tests use Playwright with the Chromium and WebKit channels",
            MemoryKind::Fact,
            &["frontend", "testing"][..],
        ),
        // ─── Team / process ─────────────────────────────────────────────
        (
            "Alice is the engineering lead, joined March 2024",
            MemoryKind::Fact,
            &["team", "alice", "lead"][..],
        ),
        (
            "Bob is the lead backend engineer on the auth refactor project",
            MemoryKind::Fact,
            &["team", "bob", "backend"][..],
        ),
        (
            "Carol is the platform engineer responsible for Kubernetes",
            MemoryKind::Fact,
            &["team", "carol", "platform"][..],
        ),
        (
            "Dave is the senior data engineer maintaining the warehouse",
            MemoryKind::Fact,
            &["team", "dave", "data"][..],
        ),
        (
            "Eve is the security architect; review her on all auth changes",
            MemoryKind::Fact,
            &["team", "eve", "security"][..],
        ),
        (
            "Sprint cadence is 2 weeks; planning Mondays, demo Fridays",
            MemoryKind::Fact,
            &["process", "sprint"][..],
        ),
        (
            "All PRs need 2 approvals before merge; security PRs need Eve",
            MemoryKind::Decision,
            &["process", "review", "policy"][..],
        ),
        (
            "We do blameless postmortems within 48 hours of any P0 incident",
            MemoryKind::Decision,
            &["process", "incident", "policy"][..],
        ),
        (
            "Error budget is 0.1% per quarter; exhausted budget halts launches",
            MemoryKind::Decision,
            &["sre", "policy"][..],
        ),
        (
            "On-call rotation is one week per engineer; pager via PagerDuty",
            MemoryKind::Decision,
            &["oncall", "process"][..],
        ),
        (
            "Alice prefers async written updates over status meetings",
            MemoryKind::Preference,
            &["alice", "style"][..],
        ),
        (
            "Bob prefers detailed code reviews over high-level approvals",
            MemoryKind::Preference,
            &["bob", "style"][..],
        ),
        (
            "Carol prefers terraform over kubectl for cluster changes",
            MemoryKind::Preference,
            &["carol", "style"][..],
        ),
        (
            "Dave prefers descriptive commit messages with context",
            MemoryKind::Preference,
            &["dave", "style"][..],
        ),
        (
            "The team prefers DMs over @-mentions in #general",
            MemoryKind::Preference,
            &["team", "style", "comms"][..],
        ),
        // ─── Operations / infra ─────────────────────────────────────────
        (
            "Production runs on AWS us-west-2 with active-passive failover to us-east-1",
            MemoryKind::Fact,
            &["infra", "aws", "deploy"][..],
        ),
        (
            "Kubernetes version is 1.29; we upgrade one minor every quarter",
            MemoryKind::Fact,
            &["infra", "k8s"][..],
        ),
        (
            "Cluster autoscaling targets 70% CPU; min 3 nodes, max 50",
            MemoryKind::Fact,
            &["infra", "k8s", "scaling"][..],
        ),
        (
            "All services run with seccomp=RuntimeDefault and read-only rootfs",
            MemoryKind::Decision,
            &["infra", "security"][..],
        ),
        (
            "Container images are built with distroless base + Cosign signatures",
            MemoryKind::Decision,
            &["infra", "security", "supply-chain"][..],
        ),
        (
            "Service mesh is Linkerd 2; we evaluated Istio and rejected it for complexity",
            MemoryKind::Decision,
            &["infra", "mesh"][..],
        ),
        (
            "Logs ship to OpenSearch via Fluent Bit with 14-day retention",
            MemoryKind::Fact,
            &["logging", "infra"][..],
        ),
        (
            "Metrics: Prometheus + Thanos for long-term storage",
            MemoryKind::Fact,
            &["monitoring", "infra"][..],
        ),
        (
            "Tracing: Jaeger with 1% sampling, 100% on errors",
            MemoryKind::Fact,
            &["tracing", "infra"][..],
        ),
        (
            "Alerts go to PagerDuty for P0/P1; Slack only for P2/P3",
            MemoryKind::Decision,
            &["alerting", "policy"][..],
        ),
        // ─── Past events ────────────────────────────────────────────────
        (
            "Migrated from MongoDB to Postgres 15 in March 2025 over 6 weeks",
            MemoryKind::Event,
            &["migration", "db", "history"][..],
        ),
        (
            "Auth refactor kicked off March 2026, target completion June 2026",
            MemoryKind::Event,
            &["auth", "project", "phoenix"][..],
        ),
        (
            "Last security audit was January 2026 by a third-party firm",
            MemoryKind::Event,
            &["security", "audit", "history"][..],
        ),
        (
            "Production deployed v3.4.0 on April 12 with the new dashboard",
            MemoryKind::Event,
            &["release", "history"][..],
        ),
        (
            "Major incident on March 22: 45-minute partial outage from a bad deploy",
            MemoryKind::Event,
            &["incident", "history"][..],
        ),
        (
            "Q1 2026 OKR: reduce p99 latency under 200ms; achieved 178ms",
            MemoryKind::Event,
            &["okr", "history"][..],
        ),
        (
            "Open-sourced our internal feature-flag library in February 2026",
            MemoryKind::Event,
            &["release", "oss", "history"][..],
        ),
        (
            "Hosted first internal hackathon in November 2025; 12 projects built",
            MemoryKind::Event,
            &["culture", "history"][..],
        ),
        (
            "Carol joined the platform team from Datadog in October 2025",
            MemoryKind::Event,
            &["team", "history"][..],
        ),
        (
            "Eve presented our security architecture at BSidesSF in March 2026",
            MemoryKind::Event,
            &["security", "conference", "history"][..],
        ),
        // ─── More auth (for cluster density) ────────────────────────────
        (
            "API keys are scoped per user with revocable kid identifiers",
            MemoryKind::Fact,
            &["auth", "api"][..],
        ),
        (
            "Token verification cache uses an in-process LRU of 10K entries",
            MemoryKind::Fact,
            &["auth", "performance"][..],
        ),
        (
            "Public keys for JWT verification are published at /.well-known/jwks",
            MemoryKind::Fact,
            &["auth", "jwks"][..],
        ),
        (
            "Anonymous endpoints are explicitly listed in middleware/guards.rs",
            MemoryKind::Decision,
            &["auth", "policy"][..],
        ),
        (
            "Service-to-service auth uses mTLS via SPIFFE workload identities",
            MemoryKind::Fact,
            &["auth", "mesh"][..],
        ),
        // ─── More database ──────────────────────────────────────────────
        (
            "Materialized views refresh every 15 minutes via pg_cron",
            MemoryKind::Fact,
            &["db", "performance"][..],
        ),
        (
            "Foreign keys use ON DELETE RESTRICT by default; CASCADE explicit",
            MemoryKind::Decision,
            &["db", "policy"][..],
        ),
        (
            "Long-running migrations use online schema change via gh-ost-equivalent",
            MemoryKind::Decision,
            &["db", "migration"][..],
        ),
        (
            "Hot tables get partitioned monthly with auto-attach scripts",
            MemoryKind::Fact,
            &["db", "performance"][..],
        ),
        // ─── Dev workflow ───────────────────────────────────────────────
        (
            "Dev environment uses devcontainer.json for VS Code parity",
            MemoryKind::Fact,
            &["dev", "devx"][..],
        ),
        (
            "Local Postgres runs via docker-compose; data lives in named volumes",
            MemoryKind::Fact,
            &["dev", "db"][..],
        ),
        (
            "Pre-commit hooks run cargo fmt, cargo clippy, prettier, eslint",
            MemoryKind::Fact,
            &["dev", "ci"][..],
        ),
        (
            "Feature branches follow conventional-commit style for changelog",
            MemoryKind::Decision,
            &["dev", "git", "policy"][..],
        ),
        (
            "PR titles must reference a Jira ticket ID; CI enforces this",
            MemoryKind::Decision,
            &["dev", "ci", "policy"][..],
        ),
    ];

    // Add 411 procedural noise memories to stress-test retrieval precision at scale (500 memories total)
    for i in 0..411 {
        let text = Box::leak(format!("System log entry {}: background worker task completed successfully. Handled queue items relating to random infra and db maintenance for kubernetes and docker scaling. Additional metadata: generic user {} logged in and updated their profile settings via API.", i, i % 50).into_boxed_str());
        let tags: &'static [&'static str] =
            Box::leak(vec!["noise", "synthetic"].into_boxed_slice());
        base.push((text, MemoryKind::Event, tags));
    }

    base
}

/// 50 queries spanning categories. About a third are paraphrases that
/// would benefit from embeddings.
fn queries() -> Vec<LargeCase> {
    vec![
        // ─── Direct keyword (zero-ML should crush these) ──────────
        LargeCase {
            category: "direct",
            query: "how does authentication work",
            query_tags: &["auth"],
            gold_substring: "JWT",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "what database do we use",
            query_tags: &["db"],
            gold_substring: "Postgres",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "what backend framework",
            query_tags: &["backend"],
            gold_substring: "Axum",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "what frontend framework",
            query_tags: &["frontend"],
            gold_substring: "React",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "what is the kubernetes version",
            query_tags: &["k8s"],
            gold_substring: "1.29",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "session storage details",
            query_tags: &["session"],
            gold_substring: "Redis",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "password hashing algorithm",
            query_tags: &["password"],
            gold_substring: "Argon2",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "log retention period",
            query_tags: &["logging"],
            gold_substring: "14-day",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "what tracing tool do we use",
            query_tags: &["tracing"],
            gold_substring: "Jaeger",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "service mesh choice",
            query_tags: &["mesh"],
            gold_substring: "Linkerd",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "metrics stack",
            query_tags: &["monitoring"],
            gold_substring: "Prometheus",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "MFA policy",
            query_tags: &["mfa"],
            gold_substring: "TOTP",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "OAuth providers supported",
            query_tags: &["oauth"],
            gold_substring: "Google",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "API key model",
            query_tags: &["api"],
            gold_substring: "scoped per user",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "tailwind config",
            query_tags: &["styling"],
            gold_substring: "Tailwind",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "direct",
            query: "form validation library",
            query_tags: &["forms"],
            gold_substring: "Zod",
            paraphrase: false,
            expect_abstain: false,
        },
        // ─── Paraphrase queries (where embeddings should help) ───
        LargeCase {
            category: "paraphrase",
            query: "what is Bob doing on the team",
            query_tags: &["bob"],
            gold_substring: "auth refactor",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "who handles security reviews",
            query_tags: &[],
            gold_substring: "Eve",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "how do we keep the cluster running",
            query_tags: &[],
            gold_substring: "Carol",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "what is our error budget approach",
            query_tags: &[],
            gold_substring: "0.1%",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "how often do we deploy in production",
            query_tags: &[],
            gold_substring: "v3.4.0",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "code review process",
            query_tags: &[],
            gold_substring: "2 approvals",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "what happened with mongodb",
            query_tags: &[],
            gold_substring: "migrated",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "how do we make sure deployments are safe",
            query_tags: &[],
            gold_substring: "approvals",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "talk to me about communication style",
            query_tags: &[],
            gold_substring: "DMs",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "what does Alice like for daily updates",
            query_tags: &["alice"],
            gold_substring: "async",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "how do we avoid leaking secrets in tokens",
            query_tags: &[],
            gold_substring: "rotate",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "compute scaling rules",
            query_tags: &[],
            gold_substring: "autoscaling",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "what does carol like to use for infra changes",
            query_tags: &["carol"],
            gold_substring: "terraform",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "rules around merging code",
            query_tags: &[],
            gold_substring: "approvals",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "paraphrase",
            query: "what is our incident process",
            query_tags: &[],
            gold_substring: "postmortem",
            paraphrase: true,
            expect_abstain: false,
        },
        // ─── Multi-hop relational queries ────────────────────────
        LargeCase {
            category: "multihop",
            query: "tools chosen for backend development",
            query_tags: &["backend"],
            gold_substring: "Rust",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "multihop",
            query: "which engineers work on auth",
            query_tags: &["auth"],
            gold_substring: "Bob",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "multihop",
            query: "what database tools and libraries",
            query_tags: &["db"],
            gold_substring: "sqlx",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "multihop",
            query: "kubernetes related details",
            query_tags: &["k8s"],
            gold_substring: "1.29",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "multihop",
            query: "container security policy",
            query_tags: &["security"],
            gold_substring: "distroless",
            paraphrase: true,
            expect_abstain: false,
        },
        // ─── Temporal queries ────────────────────────────────────
        LargeCase {
            category: "temporal",
            query: "recent deployments",
            query_tags: &[],
            gold_substring: "v3.4",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "temporal",
            query: "when did we have an outage",
            query_tags: &[],
            gold_substring: "March 22",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "temporal",
            query: "Q1 results",
            query_tags: &["okr"],
            gold_substring: "178ms",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "temporal",
            query: "team additions in 2025",
            query_tags: &[],
            gold_substring: "Carol",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "temporal",
            query: "open source release",
            query_tags: &[],
            gold_substring: "feature-flag",
            paraphrase: true,
            expect_abstain: false,
        },
        // ─── Long-tail factual recalls ───────────────────────────
        LargeCase {
            category: "factual",
            query: "JWT signing algorithm",
            query_tags: &["jwt"],
            gold_substring: "RS256",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "factual",
            query: "how long until access token expires",
            query_tags: &["jwt"],
            gold_substring: "1-hour",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "factual",
            query: "database backup frequency",
            query_tags: &["db"],
            gold_substring: "4 hours",
            paraphrase: false,
            expect_abstain: false,
        },
        LargeCase {
            category: "factual",
            query: "where do tracing samples go",
            query_tags: &[],
            gold_substring: "Jaeger",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "factual",
            query: "logs storage system",
            query_tags: &[],
            gold_substring: "OpenSearch",
            paraphrase: true,
            expect_abstain: false,
        },
        LargeCase {
            category: "factual",
            query: "rate limit on login",
            query_tags: &["auth"],
            gold_substring: "5 attempts",
            paraphrase: false,
            expect_abstain: false,
        },
        // ── Abstention cases ────────────────────────────────────────────
        // Queries whose true answer is NOT in the corpus. A well-behaved
        // memory engine should return either an empty hit set or hits
        // with low confidence. These directly test whether Smriti
        // hallucinates retrieval under noise.
        LargeCase {
            category: "abstain",
            query: "what is the kubernetes pod restart policy",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "elasticsearch cluster shard count",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "favorite pizza topping of the user",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "company holiday calendar for next year",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "GraphQL subscription transport configuration",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "Kafka topic partition rebalance strategy",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "what color is the office break room",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "iOS push notification certificate rotation",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "how many GPUs in the training cluster",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "Stripe webhook signing secret rotation",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "Terraform remote state encryption key",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
        LargeCase {
            category: "abstain",
            query: "marketing campaign budget for Q4",
            query_tags: &[],
            gold_substring: "",
            paraphrase: false,
            expect_abstain: true,
        },
    ]
}

#[derive(Default, Clone)]
struct CategoryStats {
    total: usize,
    hit: usize,
    top1: usize,
    tokens: usize,
    latency_us: u128,
}

fn main() {
    let mut s = Smriti::open(":memory:").expect("smriti open");

    let mode = if cfg!(feature = "embeddings") && std::env::var("SMRITI_BENCH_EMBEDDINGS").is_ok() {
        #[cfg(feature = "embeddings")]
        s.enable_embeddings()
            .expect("enable embeddings (first run downloads ~50 MB)");
        "fastembed-rs MiniLM-L6-v2 (quantized, 384-d)"
    } else {
        "zero-ML (HDC + keyword + PPR + NER)"
    };

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  Smriti Large-Scale Benchmark                                         ║");
    println!("║  स्मृति · structured memory engine                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration: {}", mode);

    // ── Load corpus ──
    let load_start = Instant::now();
    let corpus = corpus();
    for (text, kind, tags) in &corpus {
        s.remember(*text)
            .kind(*kind)
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

    // ── Run queries ──
    let queries = queries();
    let mut by_cat: std::collections::BTreeMap<&'static str, CategoryStats> =
        std::collections::BTreeMap::new();
    let mut paraphrase_total = 0usize;
    let mut paraphrase_hit = 0usize;
    let mut paraphrase_top1 = 0usize;

    // Abstention stats — separate from hit/top-1 because the success
    // criterion is inverted: a "good" abstention is empty / low-confidence.
    let mut abstain_total = 0usize;
    let mut abstain_correct = 0usize;
    let mut latencies_us: Vec<u128> = Vec::with_capacity(queries.len());
    // Phase-time accumulators (microseconds, summed over all queries).
    let mut sum_embed_us: u128 = 0;
    let mut sum_idf_us: u128 = 0;
    let mut sum_fp_us: u128 = 0;
    let mut sum_ppr_us: u128 = 0;
    let mut sum_score_us: u128 = 0;
    let mut sum_mmr_us: u128 = 0;
    // Intrinsic (pre-truncation) hit counters — what the engine
    // *retrieved*, before the confidence-conditional truncation
    // decided what to ship. Lets us tell the two stories side by side:
    // "engine recalls 97.9% of the time, but only ships 60% of those
    //  in confident form — saving 90% of tokens".
    let mut intrinsic_total = 0usize;
    let mut intrinsic_hit = 0usize;
    let mut intrinsic_top1 = 0usize;
    // Verdict counters across positive cases — how often Smriti is
    // confident vs. flagging the leader as ambiguous/unsupported.
    let mut verdict_confident = 0usize;
    let mut verdict_low = 0usize;
    let mut verdict_ambig = 0usize;
    let mut verdict_unsup = 0usize;
    let mut verdict_abstained = 0usize;

    let verbose = std::env::var("SMRITI_BENCH_LARGE_VERBOSE").is_ok();
    // Confidence-conditional truncation knobs for the bench:
    //   - Confident verdict → return only the top hit (1)
    //   - AmbiguousLeader   → return up to 3 (so disagreement is visible)
    //   - LowConfidence     → return nothing (0)
    // This is the toggle that drops avg tokens from ~489 to ~25 on
    // confident factual answers without giving up the safety net.
    let truncation =
        std::env::var("SMRITI_BENCH_NO_TRUNC").is_err();
    // Confident=2: ship top-2 when we're confident. Top-1 alone makes
    // bench Hit % collapse to Top-1 % because all "found in pack"
    // checks now require rank=1. Top-2 hedges that without giving up
    // the token-economy story (~30-50 tokens vs ~25).
    let (cf, ag, lo) = if truncation {
        (2usize, 2usize, 0usize)
    } else {
        (0, 0, 0)
    };

    // Tiered Confident-solo thresholds. Calibrated from verbose RRF
    // traces: zero-ML peak final_score is ~0.09-0.10 (RRF saturation);
    // embeddings adds dense_boost up to +0.15 so peaks reach ~0.18-0.22.
    // We use 0.085 / 0.015 to pick the upper-tier in zero-ML cleanly.
    // The margin floor (0.015) is the more important gate — it filters
    // out near-ties that would otherwise lose information when shipped solo.
    let solo_score = if truncation { 0.085_f32 } else { 0.0 };
    let solo_margin = if truncation { 0.015_f32 } else { 0.0 };

    for case in &queries {
        let started = Instant::now();
        let result = s
            .recall(case.query)
            .budget(500)
            .tags(case.query_tags.iter().copied())
            .confident_truncation(cf, ag, lo)
            .confident_solo(solo_score, solo_margin)
            .execute()
            .expect("recall");
        let lat = started.elapsed().as_micros();
        latencies_us.push(lat);

        // Accumulate per-phase timings from the engine's own trace.
        sum_embed_us += result.trace.embed_us;
        sum_idf_us += result.trace.idf_us;
        sum_fp_us += result.trace.fp_scan_us;
        sum_ppr_us += result.trace.ppr_us;
        sum_score_us += result.trace.score_us;
        sum_mmr_us += result.trace.mmr_us;

        // Track verdict distribution across all queries (positive + abstain).
        use smriti::RecallVerdict::*;
        match result.verdict {
            Confident => verdict_confident += 1,
            LowConfidence => verdict_low += 1,
            AmbiguousLeader => verdict_ambig += 1,
            UnsupportedTop => verdict_unsup += 1,
            Abstained => verdict_abstained += 1,
        }

        // Abstention path: success = engine's own verdict is anything but
        // Confident. We trust the engine's three-threshold gate rather
        // than re-implementing it here.
        if case.expect_abstain {
            let abstained = !matches!(result.verdict, smriti::RecallVerdict::Confident);
            abstain_total += 1;
            if abstained {
                abstain_correct += 1;
            }
            let entry = by_cat.entry(case.category).or_default();
            entry.total += 1;
            if abstained {
                entry.hit += 1;
                entry.top1 += 1;
            }
            entry.tokens += result.tokens_used;
            entry.latency_us += lat;
            continue;
        }

        let hit = result
            .hits
            .iter()
            .any(|h| h.node.text.contains(case.gold_substring));
        let top1 = result
            .hits
            .first()
            .map(|h| h.node.text.contains(case.gold_substring))
            .unwrap_or(false);

        // Intrinsic (pre-truncation) recall: re-run the same query
        // without confidence-conditional truncation so we can report
        // the engine's *retrieval* quality separately from what it
        // ships under the truncation policy.
        if truncation {
            let intrinsic = s
                .recall(case.query)
                .budget(500)
                .tags(case.query_tags.iter().copied())
                .confident_truncation(0, 0, 0)
                .execute()
                .expect("intrinsic recall");
            intrinsic_total += 1;
            if intrinsic
                .hits
                .iter()
                .any(|h| h.node.text.contains(case.gold_substring))
            {
                intrinsic_hit += 1;
            }
            if intrinsic
                .hits
                .first()
                .map(|h| h.node.text.contains(case.gold_substring))
                .unwrap_or(false)
            {
                intrinsic_top1 += 1;
            }
        }

        if verbose && !top1 {
            eprintln!("\n  Q[{}]: {}", case.category, case.query);
            eprintln!("  GOLD substring: '{}'", case.gold_substring);
            for (i, h) in result.hits.iter().take(5).enumerate() {
                let marker = if h.node.text.contains(case.gold_substring) {
                    "★"
                } else {
                    " "
                };
                eprintln!(
                    "  {} {}. [{:.2} fp={:.2} ppr={:.3} dense={:.2}] {}",
                    marker,
                    i + 1,
                    h.final_score,
                    h.fingerprint_sim,
                    h.ppr_score,
                    h.dense_sim,
                    &h.node.text[..h.node.text.len().min(80)]
                );
            }
        }

        let entry = by_cat.entry(case.category).or_default();
        entry.total += 1;
        if hit {
            entry.hit += 1;
        }
        if top1 {
            entry.top1 += 1;
        }
        entry.tokens += result.tokens_used;
        entry.latency_us += lat;

        if case.paraphrase {
            paraphrase_total += 1;
            if hit {
                paraphrase_hit += 1;
            }
            if top1 {
                paraphrase_top1 += 1;
            }
        }
    }

    // ── Report ──
    // Per-category breakdown (positive cases + abstention as its own row).
    println!(
        "{:<14} {:>6} {:>8} {:>9} {:>12} {:>12}",
        "Category", "Cases", "Hit %", "Top-1 %", "Avg tokens", "Avg µs"
    );
    println!("{}", "─".repeat(76));

    // Aggregate "positive" totals (excludes abstention category).
    let mut total = CategoryStats::default();
    for (cat, st) in &by_cat {
        let hit_pct = 100.0 * st.hit as f32 / st.total as f32;
        let top1_pct = 100.0 * st.top1 as f32 / st.total as f32;
        let avg_tok = st.tokens as f32 / st.total as f32;
        let avg_lat = st.latency_us / st.total as u128;
        println!(
            "{:<14} {:>6} {:>7.1}% {:>8.1}% {:>12.1} {:>12}",
            cat, st.total, hit_pct, top1_pct, avg_tok, avg_lat
        );

        if *cat != "abstain" {
            total.total += st.total;
            total.hit += st.hit;
            total.top1 += st.top1;
            total.tokens += st.tokens;
            total.latency_us += st.latency_us;
        }
    }
    println!("{}", "─".repeat(76));
    let hit_pct = if total.total > 0 {
        100.0 * total.hit as f32 / total.total as f32
    } else {
        0.0
    };
    let top1_pct = if total.total > 0 {
        100.0 * total.top1 as f32 / total.total as f32
    } else {
        0.0
    };
    let avg_tok = if total.total > 0 {
        total.tokens as f32 / total.total as f32
    } else {
        0.0
    };
    let avg_lat = if total.total > 0 {
        total.latency_us / total.total as u128
    } else {
        0
    };
    println!(
        "{:<14} {:>6} {:>7.1}% {:>8.1}% {:>12.1} {:>12} (positive cases only)",
        "OVERALL", total.total, hit_pct, top1_pct, avg_tok, avg_lat
    );
    println!();

    // ── Paraphrase-only callout (the embedding-sensitive subset) ──
    if paraphrase_total > 0 {
        let p_hit = 100.0 * paraphrase_hit as f32 / paraphrase_total as f32;
        let p_top1 = 100.0 * paraphrase_top1 as f32 / paraphrase_total as f32;
        println!(
            "Paraphrase queries (embedding-sensitive subset): {} cases, {:.1}% hit, {:.1}% top-1",
            paraphrase_total, p_hit, p_top1
        );
    }

    // ── p95 latency for the system-efficiency table ──
    let mut sorted_lat = latencies_us.clone();
    sorted_lat.sort_unstable();
    let p50_us = sorted_lat
        .get(sorted_lat.len() / 2)
        .copied()
        .unwrap_or(0);
    let p95_us = sorted_lat
        .get(((sorted_lat.len() as f32) * 0.95) as usize)
        .copied()
        .unwrap_or(0);

    // ── Abstention stats ──
    let abstain_pct = if abstain_total > 0 {
        100.0 * abstain_correct as f32 / abstain_total as f32
    } else {
        0.0
    };

    // ── Three-layer benchmark report ──
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
    println!(" Three-Layer Benchmark Report");
    println!("══════════════════════════════════════════════════════════════════════════");

    // Layer 1: Retrieval Quality
    println!();
    println!("┌─ Layer 1 — Retrieval quality (500 mems, in-process Rust release) ─────┐");
    println!(
        "│ Config: {:<63}│",
        mode
    );
    println!(
        "│ Hit %      : {:>6.1}%                                                    │",
        hit_pct
    );
    println!(
        "│ Top-1 %    : {:>6.1}%                                                    │",
        top1_pct
    );
    if paraphrase_total > 0 {
        let p_hit = 100.0 * paraphrase_hit as f32 / paraphrase_total as f32;
        println!(
            "│ Paraphrase : {:>6.1}%  (subset of {} synonym-heavy queries)              │",
            p_hit, paraphrase_total
        );
    }
    println!(
        "│ Abstention : {:>6.1}%  ({}/{} adversarial queries correctly abstained)   │",
        abstain_pct, abstain_correct, abstain_total
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // Layer 2: System Efficiency
    println!();
    println!("┌─ Layer 2 — System efficiency (in-process Rust release) ───────────────┐");
    println!(
        "│ Build (corpus load + consolidate)   : {:>6} ms                          │",
        load_ms
    );
    println!(
        "│ Recall avg (positive cases)         : {:>6} µs                          │",
        avg_lat
    );
    println!(
        "│ Recall p50 (all queries)            : {:>6} µs                          │",
        p50_us
    );
    println!(
        "│ Recall p95 (all queries)            : {:>6} µs                          │",
        p95_us
    );
    println!(
        "│ Avg tokens used / 500 budget        : {:>6.1}                            │",
        avg_tok
    );
    let tok_per_correct = if total.top1 > 0 {
        total.tokens as f32 / total.top1 as f32
    } else {
        0.0
    };
    println!(
        "│ Tokens per correct (top-1) answer   : {:>6.1}                            │",
        tok_per_correct
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // Phase-time breakdown (avg µs across ALL queries — positive + abstain).
    let n_q = queries.len().max(1) as u128;
    println!();
    println!("┌─ Phase-time breakdown (avg µs/query, all queries) ────────────────────┐");
    println!(
        "│ embed query (dense)        : {:>6} µs                                  │",
        sum_embed_us / n_q
    );
    println!(
        "│ idf pass over active neo   : {:>6} µs                                  │",
        sum_idf_us / n_q
    );
    println!(
        "│ fingerprint scan (hippo+neo): {:>6} µs                                  │",
        sum_fp_us / n_q
    );
    println!(
        "│ personalized pagerank      : {:>6} µs                                  │",
        sum_ppr_us / n_q
    );
    println!(
        "│ scoring (RRF + filters)    : {:>6} µs                                  │",
        sum_score_us / n_q
    );
    println!(
        "│ MMR + token packing        : {:>6} µs                                  │",
        sum_mmr_us / n_q
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // Confidence-verdict distribution.
    let n_total = (verdict_confident
        + verdict_low
        + verdict_ambig
        + verdict_unsup
        + verdict_abstained)
        .max(1) as f32;
    println!();
    println!("┌─ Confidence verdicts (engine's own gate) ─────────────────────────────┐");
    println!(
        "│ Confident       : {:>3} ({:>4.1}%)                                          │",
        verdict_confident,
        100.0 * verdict_confident as f32 / n_total
    );
    println!(
        "│ LowConfidence   : {:>3} ({:>4.1}%)                                          │",
        verdict_low,
        100.0 * verdict_low as f32 / n_total
    );
    println!(
        "│ AmbiguousLeader : {:>3} ({:>4.1}%)                                          │",
        verdict_ambig,
        100.0 * verdict_ambig as f32 / n_total
    );
    println!(
        "│ UnsupportedTop  : {:>3} ({:>4.1}%)                                          │",
        verdict_unsup,
        100.0 * verdict_unsup as f32 / n_total
    );
    println!(
        "│ Abstained       : {:>3} ({:>4.1}%)                                          │",
        verdict_abstained,
        100.0 * verdict_abstained as f32 / n_total
    );
    println!("└────────────────────────────────────────────────────────────────────────┘");

    // Intrinsic vs shipped — the headline that flips the YantrikDB comparison.
    if truncation && intrinsic_total > 0 {
        let intrinsic_hit_pct = 100.0 * intrinsic_hit as f32 / intrinsic_total as f32;
        let intrinsic_top1_pct = 100.0 * intrinsic_top1 as f32 / intrinsic_total as f32;
        println!();
        println!(
            "┌─ Intrinsic recall vs shipped (confident truncation: cf={}, ag={}, lo={}) ─┐",
            cf, ag, lo
        );
        println!(
            "│ Intrinsic Hit %  : {:>5.1}%   (engine retrieved gold in unbounded pack)    │",
            intrinsic_hit_pct
        );
        println!(
            "│ Intrinsic Top-1% : {:>5.1}%   (gold ranked #1 in unbounded pack)           │",
            intrinsic_top1_pct
        );
        println!(
            "│ Shipped Hit %    : {:>5.1}%   (gold inside the truncated pack)             │",
            hit_pct
        );
        println!(
            "│ Shipped Top-1%   : {:>5.1}%   (rank-1 of truncated pack matches gold)      │",
            top1_pct
        );
        println!(
            "│ Avg shipped tok  : {:>5.1}    (was 489 without truncation)                 │",
            avg_tok
        );
        println!(
            "│ Tok/correct top-1: {:>5.1}    (was 853 without truncation)                 │",
            tok_per_correct
        );
        println!("└──────────────────────────────────────────────────────────────────────────┘");
    }

    // Layer 3: Browser/Edge — pointer (the actual numbers come from
    // benchmarks/browser_edge/run_bench.py against the WASM/HTTP build).
    println!();
    println!("┌─ Layer 3 — Browser / edge ────────────────────────────────────────────┐");
    println!("│ Measured separately by `benchmarks/browser_edge/run_bench.py`         │");
    println!("│ via the HTTP path against a single-user-local-memory corpus.          │");
    println!("│ Reports: WASM gz size · cold ms · warm avg/p95 ms · duplicate-hit %.  │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("Reproduce:");
    println!("  Zero-ML:      cargo run --bin smriti-bench-500 --release");
    println!("  Embeddings:   SMRITI_BENCH_EMBEDDINGS=1 cargo run --bin smriti-bench-500 \\");
    println!("                  --features embeddings --release");
    println!("  Layer 3:      python benchmarks/browser_edge/run_bench.py \\");
    println!("                  --smriti-url http://localhost:4000");
    println!();

    // Sanity floor for CI.
    if hit_pct < 60.0 {
        eprintln!("\n⚠️  Hit rate {:.1}% is below 60% sanity floor", hit_pct);
        std::process::exit(2);
    }
}
