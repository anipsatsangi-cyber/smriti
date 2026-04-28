"""LongMemEval loader (real schema).

Schema: list of question objects, each with
    question_id          - str
    question_type        - str (single-session-user, multi-session, ...)
    question             - str
    question_date        - str (e.g. "2023/05/30 (Tue) 23:40")
    answer               - str (gold)
    answer_session_ids   - list[str] (sessions holding the gold)
    haystack_dates       - list[str] (per-session timestamps)
    haystack_session_ids - list[str]
    haystack_sessions    - list[list[{role, content}]]

Each question maps to one EvalTask. The full haystack (every session
in the question's history) is ingested before we issue the query —
this is what makes it long-context: the answer lives among ~50 other
unrelated sessions.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import EvalTask  # noqa: E402


def load_longmemeval_dataset(file_path: str | Path) -> List[EvalTask]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)

    tasks = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue

        # Flatten the per-session message lists into a single ordered
        # `history` list. The Smriti harness ingests this as a stream
        # of role/content turns; session boundaries are preserved as
        # tag metadata on the memory if we want to add that later.
        history = []
        sessions = item.get("haystack_sessions", []) or []
        session_ids = item.get("haystack_session_ids", []) or []
        for sess_idx, session in enumerate(sessions):
            sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"s{sess_idx}"
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if not content:
                    continue
                history.append(
                    {
                        "role": role,
                        "content": content,
                        # Carry the source session id so downstream
                        # eval / future LLM-as-judge can cite it.
                        "session_id": sid,
                    }
                )

        tasks.append(
            EvalTask(
                id=item.get("question_id", str(len(tasks))),
                history=history,
                query=question,
                ground_truth=answer,
                metadata={
                    "category": item.get("question_type", "unknown"),
                    "answer_session_ids": item.get("answer_session_ids", []),
                    "n_sessions": len(sessions),
                    "n_turns": len(history),
                },
            )
        )

    return tasks
