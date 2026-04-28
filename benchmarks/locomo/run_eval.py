"""LOCOMO evaluation harness.

Real dataset (snap-research/locomo, locomo10.json: 10 conversations,
~1700 QA pairs total, sessions per conversation up to 35).

This benchmark uses a grouped runner: each conversation's haystack is
ingested ONCE into Smriti, then all of its QA pairs are queried back-
to-back. This mirrors how a real long-running agent would use Smriti
(persistent memory, many recalls) rather than re-ingesting per query.

Substring eval — see longmemeval/run_eval.py for the same caveat.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import BenchmarkHarness, EvalTask  # noqa: E402

from loader import LocomoConversation, load_locomo_grouped  # noqa: E402


DEFAULT_DATA = "data/locomo10.json"


def evaluate_locomo_task(task: EvalTask, context: str) -> float:
    if not context:
        return 0.0
    gt = str(task.ground_truth).strip().lower()
    if not gt:
        return 0.0
    return 1.0 if gt in context.lower() else 0.0


def _duplicate_stats(lines: List[str]) -> Dict[str, Any]:
    if not lines:
        return {"unique_hits": 0, "duplicate_hits": 0, "duplicate_hit_rate": 0.0}
    unique = len(set(lines))
    return {
        "unique_hits": unique,
        "duplicate_hits": len(lines) - unique,
        "duplicate_hit_rate": (len(lines) - unique) / len(lines),
    }


def run_grouped(
    convs: List[LocomoConversation],
    smriti_url: str,
    limit: int = 0,
    category: str | None = None,
) -> Dict[str, Any]:
    """Ingest each conversation once, then run all its QA pairs against
    the same persisted scope. This is the LOCOMO-correct evaluation
    pattern."""
    harness = BenchmarkHarness(name="locomo", smriti_url=smriti_url)
    client = harness.client

    results = []
    cat_scores: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"total": 0, "hit": 0})
    total_score = 0.0
    total_tokens = 0
    total_dup_hits = 0
    total_hits = 0
    n_run = 0

    for conv in convs:
        # Filter QA list (category + limit are applied across the
        # global stream, not per conv). We still ingest the haystack
        # because the loader has already tied each task to its conv.
        relevant = conv.tasks
        if category:
            relevant = [t for t in relevant if t.metadata.get("category") == category]
        if not relevant:
            continue

        # Stop early if global limit reached.
        if limit and n_run >= limit:
            break

        # Ingest haystack ONCE under the conversation's scope.
        client.set_scope(agent="benchmark-locomo", user=conv.id)
        ingest_start = time.perf_counter()
        for turn in conv.history:
            content = turn.get("content", "")
            if not content:
                continue
            client.remember(
                content,
                tags=["locomo", f"role:{turn.get('role','user')}", f"session:{turn.get('session_id','?')}"],
                kind="event",
                importance=0.4,
            )
        ingest_ms = (time.perf_counter() - ingest_start) * 1000
        harness.logger.info(
            f"Conversation {conv.id}: ingested {len(conv.history)} turns "
            f"in {ingest_ms:.0f} ms; running {len(relevant)} QA"
        )

        for task in relevant:
            if limit and n_run >= limit:
                break
            n_run += 1

            recall = client.recall(task.query, budget=2000)
            lines = [hit.node.text for hit in recall.hits]
            context_text = "\n".join(lines)
            score = evaluate_locomo_task(task, context_text)
            dup = _duplicate_stats(lines)

            total_score += score
            total_tokens += recall.tokens_used
            total_dup_hits += dup["duplicate_hits"]
            total_hits += len(lines)
            cat = task.metadata.get("category", "unknown")
            cat_scores[cat]["total"] += 1
            if score >= 1.0:
                cat_scores[cat]["hit"] += 1

            results.append(
                {
                    "task_id": task.id,
                    "conv_id": conv.id,
                    "query": task.query,
                    "ground_truth": task.ground_truth,
                    "score": score,
                    "category": cat,
                    "recall_stats": {
                        "hits": len(recall.hits),
                        "tokens_used": recall.tokens_used,
                        **dup,
                    },
                }
            )

    avg = total_score / n_run if n_run else 0.0
    by_cat = {
        cat: {
            "total": v["total"],
            "hit": v["hit"],
            "score_pct": (100.0 * v["hit"] / v["total"]) if v["total"] else 0.0,
        }
        for cat, v in sorted(cat_scores.items())
    }

    report = {
        "benchmark": "locomo",
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": {
            "average_score": avg,
            "total_tasks": n_run,
            "total_tokens_used": total_tokens,
            "average_tokens_used": total_tokens / n_run if n_run else 0.0,
            "average_duplicate_hit_rate": total_dup_hits / total_hits if total_hits else 0.0,
        },
        "by_category": by_cat,
        "results": results,
    }
    harness.save_results(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run LOCOMO evaluation on Smriti")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA,
        help=f"Path to dataset JSON (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--smriti-url",
        type=str,
        default="http://localhost:4000",
        help="Smriti HTTP server URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N QA pairs across all conversations (0 = all).",
    )
    parser.add_argument(
        "--category",
        type=int,
        default=None,
        help="Filter to a single QA category id (1-5).",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

    try:
        convs = load_locomo_grouped(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Download the dataset first:\n"
            "  curl -L -o benchmarks/locomo/data/locomo10.json "
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            file=sys.stderr,
        )
        sys.exit(1)

    n_qa = sum(len(c.tasks) for c in convs)
    n_turns = sum(len(c.history) for c in convs)
    print(
        f"Loaded {len(convs)} conversations, {n_qa} QA pairs, "
        f"{n_turns} total turns from {os.path.basename(data_path)}"
    )

    report = run_grouped(
        convs,
        smriti_url=args.smriti_url,
        limit=args.limit,
        category=args.category,
    )
    m = report["metrics"]
    print(
        f"\nDone. {m['total_tasks']} tasks scored. "
        f"avg={m['average_score']:.3f}  "
        f"avg_tokens={m['average_tokens_used']:.1f}  "
        f"dup_rate={m['average_duplicate_hit_rate']:.3f}"
    )
    print("By category:")
    for cat, v in report["by_category"].items():
        print(f"  {cat}: {v['hit']}/{v['total']} = {v['score_pct']:.1f}%")


if __name__ == "__main__":
    main()
