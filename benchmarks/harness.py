"""Base harness for running memory benchmarks against Smriti."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from smriti import SmritiClient


@dataclass
class EvalTask:
    """A single evaluation task."""
    id: str
    history: List[Dict[str, str]]  # list of {"role": "...", "content": "..."}
    query: str
    ground_truth: Any
    metadata: Dict[str, Any]


class BenchmarkHarness:
    """Base harness for running memory benchmarks."""

    def __init__(self, name: str, smriti_url: str = "http://localhost:4000"):
        self.name = name
        self.client = SmritiClient(base_url=smriti_url)
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"smriti.bench.{name}")

    def _set_task_scope(self, session_id: str) -> None:
        """Isolate each benchmark task in its own namespace."""
        self.client.set_scope(agent="benchmark", user=session_id)

    @staticmethod
    def _duplicate_stats(lines: List[str]) -> Dict[str, Any]:
        """Compute exact duplicate rates for retrieved context lines."""
        if not lines:
            return {
                "unique_hits": 0,
                "duplicate_hits": 0,
                "duplicate_hit_rate": 0.0,
            }

        unique_hits = len(set(lines))
        duplicate_hits = len(lines) - unique_hits
        duplicate_hit_rate = duplicate_hits / len(lines)
        return {
            "unique_hits": unique_hits,
            "duplicate_hits": duplicate_hits,
            "duplicate_hit_rate": duplicate_hit_rate,
        }

    def ingest_history(self, session_id: str, history: List[Dict[str, str]]) -> None:
        """Ingest conversation history into Smriti."""
        self._set_task_scope(session_id)

        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not content:
                continue

            self.client.remember(
                f"{role}: {content}",
                tags=[f"role:{role}", "conversation"],
                kind="event",
                importance=0.4 if role == "user" else 0.3,
            )

    def evaluate(self, tasks: List[EvalTask], eval_fn: Callable[[EvalTask, str], float]) -> Dict[str, Any]:
        """Run evaluation across all tasks and compute scores."""
        self.logger.info(f"Starting evaluation of {len(tasks)} tasks...")

        results = []
        total_score = 0.0
        total_tokens_used = 0
        total_duplicate_hits = 0
        total_hits = 0

        for i, task in enumerate(tasks):
            if i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(tasks)}")

            # 1. Isolate namespace
            self._set_task_scope(task.id)

            # 2. Ingest history
            self.ingest_history(task.id, task.history)

            # 3. Recall context for the query
            recall_res = self.client.recall(task.query, budget=2000)
            context_lines = [hit.node.text for hit in recall_res.hits]
            context = "\n".join(context_lines)
            duplicate_stats = self._duplicate_stats(context_lines)

            # 4. Evaluate using the provided metric function
            score = eval_fn(task, context)
            total_score += score
            total_tokens_used += recall_res.tokens_used
            total_duplicate_hits += duplicate_stats["duplicate_hits"]
            total_hits += len(context_lines)

            tokens_per_correct_answer = (
                recall_res.tokens_used / score if score > 0 else None
            )

            results.append({
                "task_id": task.id,
                "query": task.query,
                "retrieved_context": context,
                "retrieved_context_lines": context_lines,
                "ground_truth": task.ground_truth,
                "score": score,
                "metadata": task.metadata,
                "recall_stats": {
                    "hits": len(recall_res.hits),
                    "tokens_used": recall_res.tokens_used,
                    **duplicate_stats,
                    "tokens_per_correct_answer": tokens_per_correct_answer,
                }
            })

        avg_score = total_score / len(tasks) if tasks else 0.0
        self.logger.info(f"Evaluation complete. Average score: {avg_score:.4f}")

        report = {
            "benchmark": self.name,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {
                "average_score": avg_score,
                "total_tasks": len(tasks),
                "total_tokens_used": total_tokens_used,
                "average_tokens_used": (total_tokens_used / len(tasks)) if tasks else 0.0,
                "token_per_correct_answer": (
                    total_tokens_used / total_score if total_score > 0 else None
                ),
                "total_duplicate_hits": total_duplicate_hits,
                "average_duplicate_hit_rate": (
                    total_duplicate_hits / total_hits if total_hits else 0.0
                ),
            },
            "results": results,
        }

        self.save_results(report)
        return report

    def save_results(self, report: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{self.name}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"Results saved to {filename}")
