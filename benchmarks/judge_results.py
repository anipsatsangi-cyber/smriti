"""Re-score a cached benchmark result JSON with the LLM judge.

Reads the JSON Smriti's harness already wrote (which contains question,
gold, retrieved context per task) and adds a `judge_score` to each
entry. Writes a new file alongside the input named `*_judged.json`.

Usage:
    python judge_results.py results/longmemeval_strat_xxx.json
    python judge_results.py results/locomo_xxx.json --limit 30 --model meta/llama-3.3-70b-instruct

Why standalone: the Smriti server doesn't need to run again. Re-judging
is a function of (question, gold, retrieved_context) — all of which
are already captured in the result file. Cheap, repeatable, no
ingestion cost.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

# Allow `judge.py` import whether invoked from benchmarks/ or repo root
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import DEFAULT_MODEL, NimJudge  # noqa: E402


def _context_for(entry: dict) -> str:
    """Recover the retrieved context that the harness packed."""
    if entry.get("retrieved_context"):
        return entry["retrieved_context"]
    lines = entry.get("retrieved_context_lines") or []
    return "\n".join(lines)


def _category_of(entry: dict) -> str:
    md = entry.get("metadata") or {}
    return str(md.get("category", entry.get("category", "?")))


def main():
    p = argparse.ArgumentParser(description="Re-score a cached benchmark JSON with NIM LLM judge.")
    p.add_argument("input", type=str, help="Path to cached *.json result")
    p.add_argument("--limit", type=int, default=0, help="Judge only the first N entries (0=all)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--rate-sleep",
        type=float,
        default=1.6,
        help="Sleep seconds after each call. Default 1.6 ≈ 37 req/min, under NIM's 40/min free-tier cap.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Default: <input>_judged.json next to the input.",
    )
    p.add_argument(
        "--filter-category",
        type=str,
        default=None,
        help="Only judge entries whose category matches this string.",
    )
    args = p.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(src.read_text())

    entries = data.get("results") or []
    if args.filter_category:
        entries = [e for e in entries if _category_of(e) == args.filter_category]
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    print(
        f"Loaded {len(entries)} entries from {src.name}"
        f"{' (filtered category=' + args.filter_category + ')' if args.filter_category else ''}"
    )
    if not entries:
        print("nothing to judge")
        return

    judge = NimJudge(model=args.model, rate_limit_sleep=args.rate_sleep)

    judged = []
    cat_score = defaultdict(lambda: {"n": 0, "judge": 0.0, "substr": 0.0})
    t0 = time.perf_counter()
    for i, entry in enumerate(entries):
        q = entry.get("query", "")
        gold = str(entry.get("ground_truth", ""))
        ctx = _context_for(entry)
        if not q or not gold:
            continue
        try:
            v = judge.judge_support(q, gold, ctx)
        except Exception as e:
            print(f"  [{i+1}/{len(entries)}] judge error: {e}", file=sys.stderr)
            v_score, v_reason = 0.0, f"ERROR: {e!r}"
            v = type("V", (), {"score": v_score, "reason": v_reason, "raw": "", "prompt_tokens": 0, "completion_tokens": 0})()
        cat = _category_of(entry)
        substr = float(entry.get("score", 0.0))
        cat_score[cat]["n"] += 1
        cat_score[cat]["judge"] += v.score
        cat_score[cat]["substr"] += substr

        e = dict(entry)
        e["judge_score"] = v.score
        e["judge_reason"] = v.reason
        e["substring_score"] = substr  # preserve old number explicitly
        judged.append(e)
        if (i + 1) % 5 == 0 or i + 1 == len(entries):
            elapsed = time.perf_counter() - t0
            print(
                f"  {i+1}/{len(entries)} done | "
                f"elapsed {elapsed:.0f}s | "
                f"calls={judge.calls} prompt_tok={judge.total_prompt_tokens} "
                f"completion_tok={judge.total_completion_tokens} errors={judge.errors}"
            )

    # ── Aggregate ──
    n = len(judged)
    avg_judge = sum(e["judge_score"] for e in judged) / n if n else 0.0
    avg_substr = sum(e["substring_score"] for e in judged) / n if n else 0.0

    print()
    print(f"=== Judged: {n} entries with {args.model} ===")
    print(f"  Substring  avg : {avg_substr:.3f}")
    print(f"  LLM-judge  avg : {avg_judge:.3f}")
    print(f"  Lift           : {(avg_judge - avg_substr)*100:+.1f} percentage points")
    print(f"  Total tokens   : prompt={judge.total_prompt_tokens}, completion={judge.total_completion_tokens}")
    print()
    print(f"  {'category':<32} {'n':>4}  substr%  judge%   lift")
    for cat in sorted(cat_score):
        v = cat_score[cat]
        if not v["n"]:
            continue
        s = 100.0 * v["substr"] / v["n"]
        j = 100.0 * v["judge"] / v["n"]
        print(f"  {cat:<32} {v['n']:>4}  {s:>6.1f}%  {j:>6.1f}%  {j-s:+6.1f}")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_judged.json")
    payload = {
        "source": str(src),
        "judged_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "summary": {
            "total": n,
            "substring_avg": avg_substr,
            "judge_avg": avg_judge,
            "by_category": {
                cat: {
                    "n": v["n"],
                    "substring_pct": 100.0 * v["substr"] / v["n"] if v["n"] else 0.0,
                    "judge_pct": 100.0 * v["judge"] / v["n"] if v["n"] else 0.0,
                }
                for cat, v in cat_score.items()
            },
            "judge_total_prompt_tokens": judge.total_prompt_tokens,
            "judge_total_completion_tokens": judge.total_completion_tokens,
            "judge_errors": judge.errors,
        },
        "results": judged,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
