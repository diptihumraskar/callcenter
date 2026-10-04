"""Pipeline service: bridges the Gradio UI and the compiled LangGraph workflow."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from src.graph.state import AudioInput
from src.utils.formatters import format_qa, format_summary, secs_to_mmss

_temp_files: list[str] = []
_MAX_TEMP_FILES = 50


@dataclass
class PipelineResult:
    status: str
    transcript_text: str = ""
    summary_markdown: str = ""
    qa_markdown: str = ""
    pdf_path: str | None = None
    json_path: str | None = None
    error: str | None = None


def _track_temp_file(path: str) -> None:
    """Keep a rolling cap of temp files so disk usage never grows unbounded."""
    _temp_files.append(path)
    while len(_temp_files) > _MAX_TEMP_FILES:
        old_path = _temp_files.pop(0)
        try:
            os.remove(old_path)
        except OSError:
            pass


def _audio_to_wav_bytes(audio) -> bytes:
    """Convert a Gradio (sample_rate, np.ndarray) tuple into WAV bytes."""
    sample_rate, array = audio
    array = np.asarray(array)
    if array.dtype != np.int16:
        # Normalize floats in [-1, 1] to int16 PCM.
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, -1.0, 1.0)
            array = (array * 32767).astype(np.int16)
        else:
            array = array.astype(np.int16)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, array, sample_rate, subtype="PCM_16")
        _track_temp_file(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()


def _format_transcript_display(transcription) -> str:
    lines = []
    for seg in transcription.segments:
        marker = " [LOW CONF]" if seg.flagged_low_confidence else ""
        lines.append(
            f"[{secs_to_mmss(seg.start)}-{secs_to_mmss(seg.end)}] {seg.speaker}: {seg.text}{marker}"
        )
    return "\n".join(lines) if lines else transcription.full_text


def process_call(
    workflow,
    audio,
    caller_id: str | None,
    department: str | None,
    langfuse_handler=None,
) -> PipelineResult:
    """Run one call through the compiled workflow and shape the UI-facing result.

    ``langfuse_handler``, when provided, is a LangChain callback handler
    passed once at the top-level ``invoke`` call - LangChain propagates it
    down through every node and LLM call automatically (see
    src.utils.observability_providers for why this differs from LangSmith,
    which needs no explicit wiring).
    """
    if audio is None:
        return PipelineResult(status="failed", error="No audio provided.")

    if isinstance(audio, tuple):
        wav_bytes = _audio_to_wav_bytes(audio)
        filename = "recording.wav"
    else:
        # A file path from a Gradio file-upload component.
        with open(audio, "rb") as f:
            wav_bytes = f.read()
        filename = os.path.basename(audio)

    audio_input = AudioInput(
        audio_data=wav_bytes,
        filename=filename,
        caller_id=caller_id or None,
        department=department or None,
    )

    invoke_config = {"callbacks": [langfuse_handler]} if langfuse_handler else None
    result_state = workflow.invoke({"audio_input": audio_input}, config=invoke_config)
    status = result_state.get("status", "failed")
    report = result_state.get("report")

    if status not in ("completed",):
        if status == "flagged_for_review":
            default_error = (
                "This call was flagged for supervisor review (e.g. a critical "
                "compliance issue was detected) and requires manual review "
                "before a report can be generated."
            )
        else:
            default_error = "Call processing failed."
        error = result_state.get("error") or default_error
        return PipelineResult(status=status, error=error)

    transcription = result_state["transcription"]
    summary = result_state["summary"]
    qa_scores = result_state["qa_scores"]

    from src.agents.report import generate_report_json, generate_report_pdf

    pdf_bytes = generate_report_pdf(report)
    json_text = generate_report_json(report)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf_file:
        pdf_file.write(pdf_bytes)
        pdf_path = pdf_file.name
        _track_temp_file(pdf_path)

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", mode="w", encoding="utf-8"
    ) as json_file:
        json_file.write(json_text)
        json_path = json_file.name
        _track_temp_file(json_path)

    return PipelineResult(
        status=status,
        transcript_text=_format_transcript_display(transcription),
        summary_markdown=format_summary(summary),
        qa_markdown=format_qa(qa_scores),
        pdf_path=pdf_path,
        json_path=json_path,
    )
