# LOCOMO Benchmark Evaluation

This directory contains the harness for running the [LOCOMO (Long-term Conversational Memory)](https://arxiv.org/abs/2402.11522) benchmark against Smriti.

## Dataset

LOCOMO is a dataset of long-term conversational memory tasks. It involves multi-session dialogues where the agent must recall information from past sessions to answer questions.

To run this evaluation, you need to download the LOCOMO dataset.
The dataset is available on Hugging Face: `hf.co/datasets/locomo` (placeholder).

1. Download the dataset JSON files.
2. Place them in a `data/` directory (e.g., `benchmarks/memory/locomo/data/locomo.json`).

## Running the Evaluation

Once the dataset is downloaded and the Smriti server is running:

```bash
python run_eval.py --data-path data/locomo.json
```
