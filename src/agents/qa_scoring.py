"""QA scoring agent: five-dimension rubric with deterministic overall score."""

from __future__ import annotations

import time

from src.graph.state import QAScoreResult, SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss

DIMENSION_WEIGHTS = {
    "professionalism": 0.15,
    "empathy": 0.20,
    "problem_resolution": 0.30,
    "compliance": 0.20,
    "communication_clarity": 0.15,
}


class QAScoringError(Exception):
    """Raised when QA scoring fails after all retry attempts."""


_SYSTEM_PROMPT = """\
You are a senior call center QA coach. Score the agent's handling of this
call on five dimensions, 1-5 each. Scoring philosophy: 3 is the baseline
for competent, unremarkable handling - do not inflate scores. A short call
that resolves the issue efficiently is not deficient; do not penalize
brevity.

Dimension rubrics (1-5, low to high):
- Professionalism (15%): language quality, greeting and closing, composure
  under pressure, no interruptions.
- Empathy (20%): active listening, acknowledging customer feelings,
  building rapport, personalized responses.
- Problem Resolution (30%): root cause identification, solution quality,
  customer confirmation of understanding.
- Compliance (20%): required disclosures, identity verification steps,
  hold procedures, data safety.
- Communication Clarity (15%): clear explanations, minimal jargon,
  structured delivery, confirmed comprehension.

Justification guidelines: cite specific transcript timestamps in MM:SS
format and write like a real coach giving feedback, not a rubric checklist.

Compliance flags: only flag genuine procedural violations (e.g. skipped
identity verification before disclosing account data), never mere style
preferences. Assign severity low/medium/high/critical.
"""


def _format_transcript(transcript: TranscriptionResult) -> str:
    lines = [
        f"[{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}] {seg.speaker}: {seg.text}"
        for seg in transcript.segments
    ]
    return "\n".join(lines) if lines else transcript.full_text


def _recompute_overall_score(result: QAScoreResult) -> float:
    """Deterministic guardrail: always overrides whatever the LLM proposed."""
    weighted = sum(
        getattr(result, dimension).score * weight for dimension, weight in DIMENSION_WEIGHTS.items()
    )
    return round(weighted, 2)


def run_qa_scoring(
    transcript: TranscriptionResult,
    summary: SummaryResult,
    llm,
    max_retries: int = 3,
) -> QAScoreResult:
    """Score the call, then discard the LLM's overall_score for a computed one."""
    structured_llm = llm.with_structured_output(QAScoreResult)
    formatted = _format_transcript(transcript)

    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "user",
            f"Transcript:\n{formatted}\n\n"
            f"Call summary context:\n{summary.model_dump_json(indent=2)}\n\n"
            "Score all five dimensions and list any compliance flags.",
        ),
    ]

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: QAScoreResult = structured_llm.invoke(messages)
            result.call_id = transcript.call_id
            result.overall_score = _recompute_overall_score(result)
            return result
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 10))

    raise QAScoringError(
        f"QA scoring failed after {max_retries} attempts: {last_exc}"
    ) from last_exc
