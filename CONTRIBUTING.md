# Contributing to Smriti

Thanks for considering a contribution. This is a serious-but-friendly
project: the architecture is opinionated, the tests are thorough, and
review is honest. If you ship something here, you'll probably be a
better engineer afterwards.

## Quickstart

```bash
git clone https://github.com/fork-demon/smriti
cd smriti

# Build + run all 83 tests + 1 integration test
cargo test --release --lib
cargo test --release --tests

# Run the canonical retrieval-quality benchmark
cargo run --release --bin smriti-bench-500

# Confirm cognitive-feature integration test still passes
cargo test --release --test agi_integration_test -- --nocapture
```

If `cargo test` passes locally, you're ready to open a PR. If it
doesn't, file an issue — that's a higher-priority signal than any
feature request.

## What we welcome

- **Benchmark corpora.** Real-world memory corpora with multi-session
  structure. LongMemEval and LOCOMO are wired in; head-to-heads
  against Mem0 / Letta / Zep on the same harness are on the v0.3
  roadmap and would land especially well.
- **Bindings.** Python is partial. Rust + WASM is shipped. JavaScript
  / TypeScript wrappers around the WASM build, plus Go / Swift / Kotlin
  HTTP clients, are all welcome.
- **MCP integrations.** Smriti speaks Model Context Protocol natively;
  drop-in configs for Claude Code, Cursor, Zed, and other MCP-aware
  clients are useful contributions even without code changes.
- **Bug reports** with a reproducer. The engine is deterministic — if
  you can show two recalls returning different results from the same
  inputs, that's a real bug and we want to see it.
- **Documentation fixes** when the API drifts. The README and
  `docs/capabilities.md` are pinned to a doctest
  (`core/tests/readme_example.rs`) — if that test fails, a doc is
  stale. Fixing the docs to match the API is a perfectly good PR.

## What we'd push back on

- **New cognitive-architecture features without a benchmark.** The
  engine already has Salience, persistent activation, goal priming,
  surprise mechanics, causal trajectories, and reconsolidation. A
  new feature should come with a measurable lift on a real corpus
  (or a clear capability gap a measured workload exposes).
- **Embedding-model dependencies in the default build.** Embeddings
  live behind the `embeddings` feature flag. Anything that pulls a
  model into the default path breaks the WASM / edge story.
- **Surface-area expansion without docs.** New public API ⇒ new
  example in `docs/capabilities.md` and a corresponding line in the
  drift-pinned `readme_example.rs`. We'd rather have one well-documented
  primitive than five undocumented ones.

## How review works

PRs get an honest read. Expect:

1. **Architectural feedback first** — does the change fit the
   primitives-first model, or does it bake an LLM into the engine?
2. **Test coverage second** — what new test pins the behavior, and is
   it deterministic on a small corpus?
3. **Docs third** — capability doc updated, README example still
   compiles, MCP tool list current if you added one.
4. **Performance fourth** — bench-500 numbers pre/post, especially
   if the change touches `core::recall`, `core::neocortex`, or
   `core::consolidation`.

Review is direct. We point out what's wrong, what's strong, and what
would land. Disagreement is fine and expected.

## Code style

- `cargo fmt` before pushing.
- `cargo clippy --release --all-targets -- -D warnings` should pass.
- Rustdoc on every public item. If the public item is interesting,
  the doc explains *why* it exists, not just what it does.
- Prefer `anyhow::Result` for binaries, `thiserror` for libraries.
  We've stayed consistent on this; please do too.

## Tests

- New public function ⇒ at least one unit test exercising the happy
  path and one edge case (empty input, type mismatch, scope isolation,
  whichever applies).
- New scoring change ⇒ a corresponding bench-500 run pasted in the PR
  description showing pre/post numbers.
- New cognitive feature ⇒ either a unit test showing the mechanism
  fires, or a small dedicated integration test (see
  `core/tests/agi_integration_test.rs` as the pattern).

We currently sit at 83 unit tests + 1 integration test, all green on
both `--features ""` and `--features "embeddings http"`. Don't break
that.

## Contributor License Agreement (CLA)

For any non-trivial PR (more than a typo / single-line fix), we ask
contributors to sign a lightweight CLA. The CLA grants Smriti the
right to relicense the codebase later — currently MIT, but if the
project ever needs to dual-license under a copyleft variant for
commercial sustainability, the CLA gives us that option without
having to track down every contributor for permission.

The CLA does **not**:

- Transfer copyright. You retain ownership of your contribution.
- Limit how you use your own code outside Smriti.
- Restrict what the project can do for you under MIT (your
  contribution stays MIT-available forever).

CLA signing is automated via [cla-assistant.io](https://cla-assistant.io).
First non-trivial PR triggers it; subsequent PRs reuse the signature.

If the CLA is a blocker for your contribution, open an issue and we'll
talk through it. We'd rather get the contribution than enforce
paperwork.

## Project values, in plain English

1. **Be honest about numbers.** Don't claim a benchmark we didn't
   run. Don't compare against a competitor we didn't measure. The
   bench-500 binary is the source of truth.
2. **Don't bake an LLM into the engine.** Smriti is a memory engine,
   not a memory model. The LLM is the agent's reasoning layer. Keep
   that separation clean.
3. **Determinism is a feature.** If a recall produces different results
   from the same inputs, that's a bug — not a quirk to document.
4. **The WASM build is load-bearing.** Edge / browser deployment is
   one of the three things that make Smriti different. Anything that
   makes the WASM bundle bigger or slower needs to justify itself.
5. **Tests over PRs.** A 50-line PR with three tests beats a 500-line
   PR with none.

## Filing issues

Good issues are gold. Try to include:

- The Smriti version (`smriti --version` or `Cargo.lock` line).
- A minimal reproducer (5-10 lines of Rust ideal; harness
  configuration if it's a benchmark anomaly).
- What you expected vs. what happened.
- Whether the failure is deterministic.

Bad issues we'll still try to help with:

- "Doesn't work."
- "How do I do X?" — issues are fine for this if there's no doc on
  it; we'll either answer or fix the doc.

## Communication

- **GitHub Issues** for bugs and feature requests.
- **GitHub Discussions** for design questions and open-ended
  conversations.
- **Email** `hello@smriti.ai` for security disclosures, commercial
  licensing inquiries, or anything that doesn't fit GitHub.

## A personal note

This project started because the AI memory industry collectively
skipped 30 years of cognitive science. If you're contributing, you
probably already know that's a strange thing to be true in 2026.
Welcome.

Build something with it. Tell us what breaks. We read every issue.

— Arvind & contributors
