# LongMemEval Benchmark Evaluation

This directory contains the harness for running the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark against Smriti.

## Dataset

LongMemEval evaluates LLM-driven chat assistants in sustained, multi-session interactions on tasks like information extraction, temporal reasoning, and knowledge updates.

To run this evaluation:
1. Download the dataset from the GitHub repository linked above.
2. Place the JSON file in a `data/` directory (e.g., `benchmarks/memory/longmemeval/data/longmemeval.json`).

## Running the Evaluation

```bash
python run_eval.py --data-path data/longmemeval.json
```
