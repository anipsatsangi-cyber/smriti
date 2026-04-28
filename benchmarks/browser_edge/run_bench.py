import argparse
import gzip
import json
import os
import statistics
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import BenchmarkHarness


SYNTHETIC_MEMORIES = [
    "The auth layer uses JWT RS256 with a 1-hour expiry.",
    "Refresh tokens are stored server-side and rotated on every use.",
    "The production database runs on a managed Postgres cluster.",
    "The edge deployment target is a small ARM device with 4 GB of RAM.",
    "User preferences should stay local and never leave the browser.",
    "The browser build ships as a WebAssembly module compiled from Rust.",
    "Open telemetry traces are sampled at 10 percent in staging.",
    "The billing worker retries failed jobs with exponential backoff.",
    "The search API uses a 500-token recall budget by default.",
    "Embeddings are optional and only enabled for synonym-heavy workloads.",
    "The edge cache expires session summaries every 15 minutes.",
    "The memory engine should run without a GPU on developer laptops.",
]

SYNTHETIC_QUERIES = [
    "how does auth work",
    "where do preferences live",
    "what runs on small ARM devices",
    "do we need embeddings",
    "what is the default recall budget",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser/edge efficiency benchmarks on Smriti")
    parser.add_argument("--smriti-url", type=str, default="http://localhost:4000", help="URL of Smriti server")
    parser.add_argument(
        "--wasm-path",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "target/wasm32-unknown-unknown/release/smriti.wasm"),
        help="Path to the compiled WASM artifact",
    )
    parser.add_argument("--warm-runs", type=int, default=25, help="Number of warm recall iterations")
    parser.add_argument("--budget", type=int, default=500, help="Recall budget for efficiency runs")
    return parser.parse_args()


def artifact_metrics(wasm_path: Path) -> dict:
    if not wasm_path.exists():
        return {
            "wasm_path": str(wasm_path),
            "wasm_present": False,
            "wasm_bytes": None,
            "wasm_gzip_bytes": None,
        }

    wasm_bytes = wasm_path.read_bytes()
    return {
        "wasm_path": str(wasm_path),
        "wasm_present": True,
        "wasm_bytes": len(wasm_bytes),
        "wasm_gzip_bytes": len(gzip.compress(wasm_bytes, compresslevel=9)),
    }


def duplicate_hit_rate(lines: list[str]) -> float:
    if not lines:
        return 0.0
    return (len(lines) - len(set(lines))) / len(lines)


def percentile(samples: list[float], p: int) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    rank = (len(ordered) - 1) * (p / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_server(args: argparse.Namespace) -> dict:
    harness = BenchmarkHarness(name="browser_edge", smriti_url=args.smriti_url)
    client = harness.client
    run_scope = f"browser-edge-{uuid.uuid4().hex[:8]}"
    client.set_scope(agent="browser-edge", user=run_scope)

    remember_latencies = []
    for memory in SYNTHETIC_MEMORIES:
        start = time.perf_counter()
        client.remember(memory, tags=["browser-edge", "efficiency"], kind="fact", importance=0.5)
        remember_latencies.append((time.perf_counter() - start) * 1000)

    cold_start = time.perf_counter()
    cold_result = client.recall(SYNTHETIC_QUERIES[0], budget=args.budget)
    cold_recall_ms = (time.perf_counter() - cold_start) * 1000

    warm_latencies = []
    warm_tokens = []
    warm_hits = []
    warm_duplicate_rates = []
    for i in range(args.warm_runs):
        query = SYNTHETIC_QUERIES[i % len(SYNTHETIC_QUERIES)]
        start = time.perf_counter()
        result = client.recall(query, budget=args.budget)
        warm_latencies.append((time.perf_counter() - start) * 1000)
        warm_tokens.append(result.tokens_used)
        warm_hits.append(len(result.hits))
        warm_duplicate_rates.append(duplicate_hit_rate([hit.node.text for hit in result.hits]))

    return {
        "cold_recall_ms": cold_recall_ms,
        "cold_tokens_used": cold_result.tokens_used,
        "remember_avg_ms": statistics.mean(remember_latencies),
        "warm_recall_avg_ms": statistics.mean(warm_latencies),
        "warm_recall_p95_ms": percentile(warm_latencies, 95),
        "avg_tokens_used": statistics.mean(warm_tokens) if warm_tokens else 0.0,
        "avg_hits": statistics.mean(warm_hits) if warm_hits else 0.0,
        "avg_duplicate_hit_rate": statistics.mean(warm_duplicate_rates) if warm_duplicate_rates else 0.0,
    }


def save_results(report: dict) -> Path:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"browser_edge_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path


def main() -> None:
    args = parse_args()
    report = {
        "benchmark": "browser_edge",
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "warm_runs": args.warm_runs,
            "budget": args.budget,
        },
        "artifacts": artifact_metrics(Path(args.wasm_path)),
        "metrics": benchmark_server(args),
    }
    output_path = save_results(report)
    print(json.dumps(report, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
