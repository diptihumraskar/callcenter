"""Intake agent - the gatekeeper that runs before any Whisper or LLM call."""

from __future__ import annotations

import io
import re
import tempfile
import uuid
import wave

from src.graph.state import AudioInput, AudioProperties, IntakeResult, PIIScanResult
from src.utils.audio import (
    MAX_DURATION_SECONDS,
    AudioValidationError,
    extract_audio_properties,
    validate_audio_file,
)

_EMPTY_AUDIO_PROPS = AudioProperties()
_EMPTY_PII = PIIScanResult()

# Metadata PII patterns - applied to caller_id / department strings only.
# (Full-transcript PII redaction lives in src/security/pii_redactor.py.)
_METADATA_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){16}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}


def _make_failed_result(
    call_id: str, error: str, detected_format: str | None = None
) -> IntakeResult:
    return IntakeResult(
        call_id=call_id,
        validation_passed=False,
        validation_error=error,
        detected_format=detected_format,
        audio_properties=_EMPTY_AUDIO_PROPS,
        pii_scan=_EMPTY_PII,
    )


def _scan_metadata_for_pii(caller_id: str | None, department: str | None) -> PIIScanResult:
    affected: list[str] = []
    for field_name, value in (("caller_id", caller_id), ("department", department)):
        if not value:
            continue
        for pattern in _METADATA_PII_PATTERNS.values():
            if pattern.search(value):
                affected.append(field_name)
                break
    return PIIScanResult(pii_detected=bool(affected), affected_fields=affected)


def _wav_duration_from_header(data: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate) if rate else None
    except (wave.Error, EOFError):
        return None


def run_intake(audio_input: AudioInput) -> IntakeResult:
    """Validate an uploaded call, extract its properties, and stage it on disk."""
    call_id = str(uuid.uuid4())
    data = audio_input.audio_data

    # WAV duration is checked from the header BEFORE the general size gate,
    # so an oversized-but-valid WAV reports "duration exceeds" rather than a
    # misleading "file too large" error.
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        duration = _wav_duration_from_header(data)
        if duration is not None and duration > MAX_DURATION_SECONDS:
            return _make_failed_result(
                call_id,
                f"Audio duration {duration:.1f}s exceeds maximum of "
                f"{MAX_DURATION_SECONDS}s (60 minutes).",
                detected_format="wav",
            )

    validation = validate_audio_file(data, audio_input.filename)
    if not validation.is_valid:
        return _make_failed_result(call_id, validation.error, validation.detected_format)

    try:
        raw_properties = extract_audio_properties(data, validation.detected_format)
    except AudioValidationError as exc:
        return _make_failed_result(call_id, str(exc), validation.detected_format)

    properties = AudioProperties(
        duration_seconds=raw_properties.duration_seconds,
        sample_rate=raw_properties.sample_rate,
        channels=raw_properties.channels,
    )

    if properties.duration_seconds > MAX_DURATION_SECONDS:
        return _make_failed_result(
            call_id,
            f"Audio duration {properties.duration_seconds:.1f}s exceeds maximum of "
            f"{MAX_DURATION_SECONDS}s (60 minutes).",
            validation.detected_format,
        )

    suffix = f".{validation.detected_format}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        temp_path = tmp.name

    pii_scan = _scan_metadata_for_pii(audio_input.caller_id, audio_input.department)

    return IntakeResult(
        call_id=call_id,
        validation_passed=True,
        validation_error=None,
        detected_format=validation.detected_format,
        audio_properties=properties,
        pii_scan=pii_scan,
        temp_file_path=temp_path,
        caller_id=audio_input.caller_id,
        department=audio_input.department,
    )
