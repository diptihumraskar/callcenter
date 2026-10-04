"""Summarization agent: structured extraction with exponential backoff retry."""

from __future__ import annotations

import time

from src.graph.state import SummaryResult, TranscriptionResult
from src.utils.formatters import secs_to_mmss


class SummarizationError(Exception):
    """Raised when summarization fails after all retry attempts."""


_SYSTEM_PROMPT = """\
You are a call center analyst. Read the speaker-labeled transcript and
extract a factual, grounded summary. Never invent details that are not
present in the transcript. If something is unclear, say so plainly.
"""


def _format_transcript(transcript: TranscriptionResult) -> str:
    lines = [
        f"[{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}] {seg.speaker}: {seg.text}"
        for seg in transcript.segments
    ]
    return "\n".join(lines) if lines else transcript.full_text


def run_summarization(
    transcript: TranscriptionResult,
    llm,
    max_retries: int = 3,
) -> SummaryResult:
    """Summarize a redacted transcript, retrying transient failures with backoff."""
    structured_llm = llm.with_structured_output(SummaryResult)
    formatted = _format_transcript(transcript)

    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "user",
            f"Transcript:\n{formatted}\n\n"
            "Extract call_purpose, key_discussion_points (3-7 items), "
            "action_items (with owner and optional deadline), resolution_status "
            "(resolved/unresolved/escalated), sentiment_trajectory, and entities.",
        ),
    ]

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            result: SummaryResult = structured_llm.invoke(messages)
            result.call_id = transcript.call_id
            return result
        except Exception as exc:  # noqa: BLE001 - any transient API error triggers retry
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(2**attempt, 10))

    raise SummarizationError(
        f"Summarization failed after {max_retries} attempts: {last_exc}"
    ) from last_exc
