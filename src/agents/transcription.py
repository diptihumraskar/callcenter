"""faster-whisper transcription agent: model singleton, diarization, caching."""

from __future__ import annotations

import hashlib
import json
import re

from sqlalchemy import Engine, select

from src.database.connection import session_scope
from src.database.models import TranscriptionCache
from src.graph.state import IntakeResult, TranscriptionResult, TranscriptionSegment

_model = None
_model_size: str | None = None

_CLEAN_PATTERNS = [
    (re.compile(r"\[BLANK_AUDIO\]", re.I), ""),
    (re.compile(r"\.{4,}"), "..."),
    (re.compile(r"\bthanks? for watching\b.*", re.I), ""),
    (re.compile(r"\[(music|applause|laughter|silence)\]", re.I), ""),
]
_REPEATED_PHRASE = re.compile(r"\b(\w+(?:\s+\w+){0,4})(?:\s+\1\b)+", re.I)


def _get_whisper_model(model_size: str):
    """Load faster-whisper once as a module-level singleton, auto-detecting device."""
    global _model, _model_size
    if _model is not None and _model_size == model_size:
        return _model

    device = "cpu"
    compute_type = "int8"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        # MPS (Apple Silicon) falls back to CPU - faster-whisper has no MPS backend.
    except ImportError:
        pass

    from faster_whisper import WhisperModel  # noqa: PLC0415

    _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_size = model_size
    return _model


def _clean_transcript_text(text: str) -> str:
    """Strip Whisper hallucination artifacts and collapse repeated phrases."""
    cleaned = text
    for pattern, replacement in _CLEAN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _REPEATED_PHRASE.sub(r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _segment_confidence(avg_logprob: float, no_speech_prob: float) -> float:
    logprob_conf = max(0.0, min(1.0, 1 + avg_logprob))
    speech_conf = 1 - no_speech_prob
    return round(logprob_conf * 0.7 + speech_conf * 0.3, 4)


class SpeakerDiarizer:
    """Turn-boundary detector - separates speech into two anonymous tracks.

    This only decides WHEN the speaker probably changed (silence gaps,
    question/answer alternation, a short reply after a long turn). It
    deliberately does NOT decide which track is the Agent and which is the
    Customer - guessing that from a handful of trigger phrases on every
    individual segment is brittle and, when the very first guess is wrong,
    flips the entire call. That role assignment is made once, correctly,
    later in the pipeline by an LLM with the full conversation in view (see
    src.agents.speaker_labeling.run_speaker_labeling).
    """

    def __init__(self) -> None:
        self._last_speaker: str | None = None
        self._last_end: float | None = None
        self._last_text: str = ""
        self._last_word_count: int = 0

    @staticmethod
    def _other(speaker: str) -> str:
        return "Speaker 2" if speaker == "Speaker 1" else "Speaker 1"

    def label(self, start: float, end: float, text: str) -> str:
        if self._last_speaker is None:
            speaker = "Speaker 1"
        # 1. Question -> answer switch.
        elif self._last_text.strip().endswith("?"):
            speaker = self._other(self._last_speaker)
        # 2. Gap-based switch (gap measured from the PREVIOUS segment's END,
        #    not its start - otherwise every long segment looks like a pause).
        elif self._last_end is not None and start - self._last_end > 1.2:
            speaker = self._other(self._last_speaker)
        # 3. Short affirmation (<=3 words) immediately after a long segment
        #    (>6 words) from the other party -> treat as a reply, flip.
        elif len(text.split()) <= 3 and self._last_word_count > 6:
            speaker = self._other(self._last_speaker)
        else:
            speaker = self._last_speaker

        self._last_speaker = speaker
        self._last_end = end
        self._last_text = text
        self._last_word_count = len(text.split())
        return speaker


def _compute_audio_hash(file_path: str) -> str:
    """SHA-256 of the file, read in 8 KB chunks to avoid loading it whole."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        #8kb 
        for chunk in iter(lambda: f.read(8192), b""): 
            hasher.update(chunk)
    return hasher.hexdigest()


def _check_cache(engine: Engine, audio_hash: str) -> TranscriptionResult | None:
    with session_scope(engine) as session:
        row = session.execute(
            select(TranscriptionCache).where(TranscriptionCache.audio_hash == audio_hash)
        ).scalar_one_or_none()
        if row is None:
            return None
        data = json.loads(row.transcription_json)
        return TranscriptionResult.model_validate(data)


def _save_cache(engine: Engine, audio_hash: str, result: TranscriptionResult) -> None:
    with session_scope(engine) as session:
        session.add(
            TranscriptionCache(
                audio_hash=audio_hash,
                transcription_json=result.model_dump_json(),
            )
        )


def run_transcription(
    intake: IntakeResult,
    engine: Engine | None = None,
    model_size: str = "tiny",
) -> TranscriptionResult:
    """Transcribe validated audio, with SHA-256 caching and speaker diarization."""
    audio_hash = _compute_audio_hash(intake.temp_file_path) if intake.temp_file_path else None

    if engine is not None and audio_hash is not None:
        cached = _check_cache(engine, audio_hash)
        if cached is not None:
            cached.call_id = intake.call_id
            cached.from_cache = True
            return cached

    model = _get_whisper_model(model_size)
    segments_iter, _info = model.transcribe(
        intake.temp_file_path,
        beam_size=1,
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    diarizer = SpeakerDiarizer()
    segments: list[TranscriptionSegment] = []
    full_text_parts: list[str] = []

    for seg in segments_iter:
        text = _clean_transcript_text(seg.text)
        if not text:
            continue
        confidence = _segment_confidence(seg.avg_logprob, seg.no_speech_prob)
        speaker = diarizer.label(seg.start, seg.end, text)
        segments.append(
            TranscriptionSegment(
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                text=text,
                confidence=confidence,
                flagged_low_confidence=confidence < 0.55,
            )
        )
        full_text_parts.append(text)

    result = TranscriptionResult(
        call_id=intake.call_id,
        full_text=" ".join(full_text_parts),
        segments=segments,
        from_cache=False,
        audio_hash=audio_hash,
    )

    if engine is not None and audio_hash is not None:
        _save_cache(engine, audio_hash, result)

    return result
