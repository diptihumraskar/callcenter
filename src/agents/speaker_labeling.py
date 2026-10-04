"""Speaker role assignment agent.

The transcription stage's turn-boundary detector (silence gaps,
question/answer alternation, short replies after a long turn) is only a
heuristic, and it under- or over-fires in real calls: a single person's
monologue commonly gets sliced into several short Whisper segments by
natural mid-sentence pauses, and each of those pauses can look like a
"gap" worth flipping speaker on. Once that heuristic flips one time too
many (or too few), every segment after the mistake is attributed to the
wrong bucket - a single early misfire silently inverts the rest of the
call.

Assigning a single Agent/Customer role to each of the heuristic's two
"buckets" (as an earlier version of this agent did) cannot fix that: it
inherits whatever local desync already happened. So instead, this agent
makes ONE LLM call with the full (already PII-redacted) transcript and
asks it to label every segment individually, using the whole conversation
as context - who greets, who verifies identity, who pitches a product,
who raises a concern about their own account - rather than trusting the
heuristic's turn-change guesses at all.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field

from src.graph.state import TranscriptionResult
from src.utils.formatters import secs_to_mmss


class SpeakerLabelingError(Exception):
    """Raised when speaker role assignment fails after all retry attempts."""


class SegmentRole(BaseModel):
    """The role speaking in one transcript segment."""

    segment_index: int = Field(
        description="0-based position of this segment in the transcript, matching the input order."
    )
    speaker: Literal["Agent", "Customer"] = Field(
        description="Who is actually speaking in this segment."
    )


class SpeakerRoleAssignment(BaseModel):
    """One role assignment per transcript segment."""

    segments: list[SegmentRole] = Field(
        description=(
            "Exactly one entry per input segment, in order, covering every "
            "segment index from 0 to the last one."
        )
    )


_SYSTEM_PROMPT = """\
You are analyzing a call center transcript. Speech was split into numbered \
segments by an imperfect automatic process, and segment boundaries do NOT \
reliably match real turn boundaries - a single person's turn is often split \
across several consecutive segments, especially where they pause \
mid-sentence or mid-thought. Do not assume every segment boundary is a \
speaker change.

Read the whole conversation and label every segment with who is actually \
speaking: Agent (the call center employee) or Customer (the caller). Use \
context - who greets and offers help, who asks to verify identity \
information, who states their own account/address/phone details, who \
pitches a product or price, who raises a concern about affording \
something, who follows call-handling procedure. It is normal and expected \
for the SAME role to appear across many consecutive segments in a row \
(e.g. one long pitch broken into five short segments) - only change \
speaker where the conversation content genuinely shows a change of who is \
talking. You must return exactly one label per segment index given below.
"""


def _format_transcript(transcript: TranscriptionResult) -> str:
    return "\n".join(
        f"[{i}] [{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}] {seg.text}"
        for i, seg in enumerate(transcript.segments)
    )


def run_speaker_labeling(
    transcript: TranscriptionResult,
    llm,
    max_retries: int = 3,
) -> TranscriptionResult:
    """Label every segment Agent/Customer via one LLM call over the full transcript."""
    segment_count = len(transcript.segments)
    if segment_count == 0:
        return transcript

    structured_llm = llm.with_structured_output(SpeakerRoleAssignment)
    formatted = _format_transcript(transcript)

    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "user",
            f"Transcript segments (indexed 0 to {segment_count - 1}):\n{formatted}\n\n"
            f"Label all {segment_count} segments.",
        ),
    ]

    last_exc: Exception | None = None
    roles_by_index: dict[int, str] | None = None
    for attempt in range(max_retries):
        try:
            candidate = structured_llm.invoke(messages)
            by_index = {entry.segment_index: entry.speaker for entry in candidate.segments}
            missing = [i for i in range(segment_count) if i not in by_index]
            if missing:
                last_exc = ValueError(
                    f"LLM omitted {len(missing)} of {segment_count} segment labels: {missing[:5]}"
                )
            else:
                roles_by_index = by_index
                break
        except Exception as exc:  # noqa: BLE001 - any transient API error triggers retry
            last_exc = exc
        if attempt < max_retries - 1:
            time.sleep(min(2**attempt, 10))

    if roles_by_index is None:
        raise SpeakerLabelingError(
            f"Speaker role assignment failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    for i, segment in enumerate(transcript.segments):
        segment.speaker = roles_by_index[i]

    return transcript
