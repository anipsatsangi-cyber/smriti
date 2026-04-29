"""LLM-as-judge for memory benchmarks, NIM-backed.

Uses NVIDIA NIM (build.nvidia.com) for free hosted Llama-3.3-70B. The
endpoint is OpenAI-compatible, so the OpenAI SDK works as-is.

Two scoring modes:

1. **support** — given the question, the gold answer, and the retrieved
   context, decide whether the context contains enough information to
   reach the gold answer. This is the right metric for retrieval-only
   benchmarks (Smriti returns memories; an LLM downstream would use
   them). Score in {0.0, 0.5, 1.0}: full / partial / none.

2. **answer** — generate an answer from the context, then judge against
   the gold. Closer to end-to-end QA but introduces generator bias.
   Use only when comparing against systems that also do generation.

The default is `support`, which is the fairer metric for evaluating
Smriti as a retrieval engine.

Reads the API key from `NVIDIA_API_KEY` env var or `/tmp/nv.key`. The
file path is the convenience for shell sessions where the env doesn't
inherit cleanly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import urllib.error
import urllib.request


NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"


def _load_api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_KEY")
    if key:
        return key
    p = Path("/tmp/nv.key")
    if p.is_file():
        return p.read_text().strip()
    raise RuntimeError(
        "NVIDIA_API_KEY not set and /tmp/nv.key not readable. "
        "Either `export NVIDIA_API_KEY=...` or write the key to /tmp/nv.key."
    )


SUPPORT_SYSTEM = (
    "You are a strict, impartial judge evaluating a memory-retrieval system. "
    "You will be given a question, the gold (correct) answer, and the context "
    "the system retrieved. Your job is to decide whether the context contains "
    "enough information to derive the gold answer.\n\n"
    "Rules:\n"
    "- Output ONLY a single JSON object with two fields: \"score\" (one of "
    "1.0, 0.5, 0.0) and \"reason\" (one short sentence).\n"
    "- Score 1.0 if the context fully supports the gold answer (paraphrase OK).\n"
    "- Score 0.5 if the context partially supports it (related but missing key detail).\n"
    "- Score 0.0 if the context does not support the gold answer at all.\n"
    "- Do NOT include any text outside the JSON object."
)


SUPPORT_USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "GOLD ANSWER:\n{gold}\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "Output the JSON now."
)


@dataclass
class JudgeVerdict:
    score: float
    reason: str
    raw: str
    prompt_tokens: int
    completion_tokens: int


class NimJudge:
    """Thin OpenAI-compatible judge over NVIDIA NIM. No SDK dep — uses urllib."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        rate_limit_sleep: float = 1.6,
        max_tokens: int = 200,
        temperature: float = 0.0,
        truncate_context_chars: int = 8000,
    ):
        self.api_key = _load_api_key()
        self.model = model
        self.max_retries = max_retries
        self.rate_limit_sleep = rate_limit_sleep
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.truncate_context_chars = truncate_context_chars
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0
        self.errors = 0

    def judge_support(self, question: str, gold: str, context: str) -> JudgeVerdict:
        """Score a single (question, gold, context) tuple. See module docstring."""
        if self.truncate_context_chars and len(context) > self.truncate_context_chars:
            # Keep head + tail; summarize the middle. This protects the gold
            # if it lives anywhere in the pack while staying within token budget.
            half = self.truncate_context_chars // 2
            context = context[:half] + "\n[... truncated ...]\n" + context[-half:]

        user = SUPPORT_USER_TEMPLATE.format(
            question=question, gold=gold, context=context
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SUPPORT_SYSTEM},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 1.0,
        }
        # Try to ask for JSON object output where supported. NIM's Llama-3
        # endpoints accept response_format; if rejected, retry without.
        body_with_json = dict(body)
        body_with_json["response_format"] = {"type": "json_object"}

        verdict_text, usage = self._post_with_retry(body_with_json, body)
        self.calls += 1

        score, reason = _parse_judge_output(verdict_text)
        if score is None:
            self.errors += 1
            return JudgeVerdict(
                score=0.0,
                reason=f"PARSE_ERROR: {verdict_text[:200]}",
                raw=verdict_text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        return JudgeVerdict(
            score=score,
            reason=reason,
            raw=verdict_text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _post_with_retry(self, body_primary: dict, body_fallback: dict) -> tuple[str, dict]:
        """POST with simple retry/backoff. Falls back to body_fallback on
        first 400 (response_format rejection)."""
        url = f"{NIM_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        bodies = [body_primary, body_fallback]
        last_err = None
        for attempt in range(self.max_retries):
            for body in bodies:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = json.loads(r.read().decode("utf-8"))
                    text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {}) or {}
                    # Be polite — NIM free tier ~40 req/min
                    time.sleep(self.rate_limit_sleep)
                    return text, usage
                except urllib.error.HTTPError as e:
                    last_err = e
                    body_text = ""
                    try:
                        body_text = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    if e.code in (429, 500, 502, 503, 504):
                        # Backoff, retry primary
                        time.sleep(2.0 * (attempt + 1))
                        break  # break inner for, retry attempt
                    if e.code == 400 and body is body_primary:
                        # response_format unsupported → fall through to fallback body
                        continue
                    raise
                except Exception as e:
                    last_err = e
                    time.sleep(2.0 * (attempt + 1))
                    break
        raise RuntimeError(f"judge call failed after {self.max_retries} attempts: {last_err}")


def _parse_judge_output(text: str) -> tuple[float | None, str]:
    """Extract score+reason from the model's reply. Tolerates stray prose."""
    if not text:
        return None, ""
    # Find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, ""
    blob = text[start : end + 1]
    try:
        obj = json.loads(blob)
    except Exception:
        return None, ""
    score = obj.get("score")
    reason = str(obj.get("reason", ""))
    if isinstance(score, (int, float)) and score in (0.0, 0.5, 1.0):
        return float(score), reason
    # Some models output 0/1 — coerce.
    if isinstance(score, (int, float)):
        if score >= 0.75:
            return 1.0, reason
        if score >= 0.25:
            return 0.5, reason
        return 0.0, reason
    return None, reason
