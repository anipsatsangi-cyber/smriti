"""LOCOMO loader (real schema, snap-research/locomo).

LOCOMO is structured as 10 conversations, each with up to 35 sessions
and 100-200 QA pairs. The QA pairs share the same conversation history
— so the natural unit is "ingest-once-per-conversation, query many
times" rather than "ingest per task" (which is what the harness's
default `evaluate()` does).

We expose two things:
  - `load_locomo_grouped(path)` -> list[Conversation], each with its
    full message history and a list of EvalTasks that share it.
  - `load_locomo_dataset(path)` -> flat list[EvalTask] for compatibility
    with the existing harness. The history field contains the FULL
    conversation; the harness will redundantly re-ingest it. This
    works for small runs; for the full 1.7k QA pair eval, prefer the
    grouped runner in run_eval.py.

Schema (per conversation):
    sample_id            - str (e.g. "conv-26")
    qa                   - list[{question, answer, evidence, category}]
    conversation         - dict with `speaker_a`, `speaker_b`,
                           `session_N` (list[{speaker,dia_id,text}]),
                           `session_N_date_time` (str)
    event_summary        - dict per session
    observation          - dict per session
    session_summary      - dict per session
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import EvalTask  # noqa: E402


@dataclass
class LocomoConversation:
    """One LOCOMO conversation with its shared haystack and QA set."""

    id: str
    speaker_a: str
    speaker_b: str
    history: List[dict]  # ordered list of {role, content, session_id, date}
    tasks: List[EvalTask]  # QA pairs grounded in `history`


_SESSION_RE = re.compile(r"^session_(\d+)$")


def _iter_session_keys(conv: dict) -> Iterable[tuple[int, str, str]]:
    """Yield (idx, session_text_key, date_key) for each session that has
    actual message turns. LOCOMO files include `session_N_date_time`
    keys for sessions with no message body — we skip those."""
    for k in conv.keys():
        m = _SESSION_RE.match(k)
        if not m:
            continue
        idx = int(m.group(1))
        yield idx, k, f"session_{idx}_date_time"


def _conversation_history(conv: dict) -> List[dict]:
    """Flatten all sessions into a chronological message list."""
    out = []
    sessions = sorted(_iter_session_keys(conv), key=lambda t: t[0])
    for idx, sess_key, date_key in sessions:
        turns = conv.get(sess_key) or []
        date = conv.get(date_key, "")
        sid = f"session_{idx}"
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if not text:
                continue
            out.append(
                {
                    # Map LOCOMO speakers to a coarse role for the
                    # harness. Both speakers are "users" in dialogue
                    # terms; we keep the speaker name in `content` so
                    # the recall pass can match on it.
                    "role": "user",
                    "content": f"{speaker}: {text}" if speaker else text,
                    "session_id": sid,
                    "date": date,
                    "dia_id": turn.get("dia_id", ""),
                }
            )
    return out


def load_locomo_grouped(file_path: str | Path) -> List[LocomoConversation]:
    """Load LOCOMO as conversation groups (recommended runner)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    raw = json.load(open(path))

    convs: List[LocomoConversation] = []
    for item in raw:
        sample_id = item.get("sample_id", f"conv-{len(convs)}")
        conversation = item.get("conversation", {}) or {}
        history = _conversation_history(conversation)
        speaker_a = conversation.get("speaker_a", "")
        speaker_b = conversation.get("speaker_b", "")

        tasks: List[EvalTask] = []
        for qa in item.get("qa", []) or []:
            question = qa.get("question") or ""
            answer = qa.get("answer")
            if not question or answer is None:
                # Some categories (adversarial / unanswerable) carry no
                # answer field — skip for substring eval.
                continue
            tasks.append(
                EvalTask(
                    id=f"{sample_id}_q{len(tasks)}",
                    history=history,  # shared reference; harness re-uses
                    query=question,
                    ground_truth=str(answer),
                    metadata={
                        "category": qa.get("category", "unknown"),
                        "evidence": qa.get("evidence", []),
                        "n_turns": len(history),
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "sample_id": sample_id,
                    },
                )
            )

        convs.append(
            LocomoConversation(
                id=sample_id,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                history=history,
                tasks=tasks,
            )
        )
    return convs


def load_locomo_dataset(file_path: str | Path) -> List[EvalTask]:
    """Flat list[EvalTask] view — compatible with the default harness."""
    return [t for c in load_locomo_grouped(file_path) for t in c.tasks]
