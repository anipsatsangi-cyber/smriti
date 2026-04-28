"""LongMemEval-S evaluation harness.

Real dataset (xiaowu0162/longmemeval-cleaned, longmemeval_s_cleaned.json,
500 questions across 6 categories). Each task ingests its own haystack
(~50 sessions) into a fresh Smriti scope, runs one query, and scores.

Substring eval — the gold answer phrase must appear verbatim in at
least one returned memory. This is a *floor* on quality: real LLM-as-
judge eval would credit paraphrased / extracted answers too. Treat
the substring score as a lower bound.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import BenchmarkHarness, EvalTask  # noqa: E402

from loader import load_longmemeval_dataset  # noqa: E402


DEFAULT_DATA = "data/longmemeval_s_cleaned.json"


def evaluate_longmemeval_task(task: EvalTask, context: str) -> float:
    """Substring-style eval. Returns 1.0 if any hit's text contains the
    ground-truth answer, else 0.0. Insensitive to case."""
    if not context:
        return 0.0
    gt = str(task.ground_truth).strip().lower()
    if not gt:
        return 0.0
    return 1.0 if gt in context.lower() else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Run LongMemEval-S evaluation against Smriti"
    )
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
        help="Run only the first N tasks (0 = all). Recommended first run: 50.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter to a single question_type (e.g. 'single-session-user').",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_path)

    try:
        tasks = load_longmemeval_dataset(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "Download the dataset first via huggingface_hub:\n"
            "  python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('xiaowu0162/longmemeval-cleaned', "
            "'longmemeval_s_cleaned.json', repo_type='dataset', "
            "local_dir='benchmarks/longmemeval/data')\"",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.category:
        tasks = [t for t in tasks if t.metadata.get("category") == args.category]
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    print(
        f"Loaded {len(tasks)} tasks from {os.path.basename(data_path)}"
        f"{' (filtered category=' + args.category + ')' if args.category else ''}"
    )

    harness = BenchmarkHarness(name="longmemeval", smriti_url=args.smriti_url)
    harness.evaluate(tasks, evaluate_longmemeval_task)


if __name__ == "__main__":
    main()
