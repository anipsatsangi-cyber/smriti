# Browser + Edge Efficiency Benchmark

This suite measures the part of Smriti's moat that traditional hosted memory
products usually avoid:

- **WASM artifact footprint** for in-browser deployments
- **Cold and warm recall latency** against a local HTTP server
- **Token efficiency** and exact duplicate rates under tight budgets

## What it measures

1. `wasm_bytes` and `wasm_gzip_bytes`
2. `cold_recall_ms`
3. `warm_recall_avg_ms` and `warm_recall_p95_ms`
4. `avg_tokens_used`
5. `avg_duplicate_hit_rate`

## Run it

Start the standalone Smriti HTTP server from this repo:

```bash
cargo run --manifest-path ../core/Cargo.toml --features http --bin smriti-http -- --db /tmp/smriti-browser-edge.db --port 4000
```

Then run the benchmark:

```bash
python browser_edge/run_bench.py --smriti-url http://localhost:4000
```

Results are saved into `benchmarks/results/` as JSON.

## Notes

- The benchmark intentionally uses a **small synthetic corpus** so the suite is
  quick to run while still exercising browser-sized memory workloads.
- The browser half of the suite is currently based on **artifact footprint**.
  Runtime browser benchmarks can be layered on top later with Playwright or the
  in-browser demo harness.
