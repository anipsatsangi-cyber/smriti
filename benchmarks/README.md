# Smriti Memory Benchmarks

This directory contains the evaluation harness for running industry-standard long-term memory benchmarks against the Smriti engine. We use these to validate that Smriti's HDC + PPR recall architecture outperforms or matches embedding-based systems (like Mem0 and Letta).

## Benchmarks Included

1. **[LOCOMO](locomo/README.md)**: Long-term Conversational Memory benchmark (Maharana et al., ACL 2024). Evaluates cross-session QA, temporal reasoning, and event extraction.
2. **[LongMemEval](longmemeval/README.md)**: Evaluates information extraction, multi-session reasoning, temporal reasoning, and knowledge updates.
3. **[Browser + Edge](browser_edge/README.md)**: Measures WASM artifact footprint plus cold/warm recall latency for local edge deployments.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Smriti server locally from this repo
cargo run --manifest-path ../core/Cargo.toml --features http --bin smriti-http -- --db /tmp/smriti-bench.db --port 4000
```

## Running Evaluations

Each benchmark has its own dataset loader and evaluation script. See the individual READMEs in `locomo/` and `longmemeval/` for instructions on downloading the datasets and running the harness.

Results are automatically saved to the `results/` directory as JSON files.
