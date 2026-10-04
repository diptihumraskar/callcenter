"""Audio format detection, validation, and property extraction.

Format is always determined by inspecting the first bytes of the file
(magic bytes), never by trusting a file extension - a spoofed or
mislabeled file must be caught before it reaches any downstream stage.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DURATION_SECONDS = 3600  # 60 minutes
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "m4a"}


class AudioValidationError(Exception):
    """Raised when audio properties cannot be extracted from a file."""


@dataclass
class ValidationResult:
    is_valid: bool
    error: str | None = None
    detected_format: str | None = None


@dataclass
class AudioProperties:
    duration_seconds: float
    sample_rate: int
    channels: int


def detect_audio_format(data: bytes) -> str | None:
    """Identify the audio container format from its first 12 bytes.

    - RIFF....WAVE at bytes 0-11            -> "wav"
    - ID3 header, or 0xFF/0xE0 sync bits     -> "mp3"
    - "fLaC" at bytes 0-3                    -> "flac"
    - "ftyp" at bytes 4-7                    -> "m4a"
    Returns None when nothing recognizable is present.
    """
    if len(data) < 12:
        return None

    header = data[:12]

    if header[0:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"

    if header[0:3] == b"ID3":
        return "mp3"
    if header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"

    if header[0:4] == b"fLaC":
        return "flac"

    if header[4:8] == b"ftyp":
        return "m4a"

    return None


def validate_audio_file(data: bytes, filename: str) -> ValidationResult:
    """Reject empty, oversized, or unsupported-format audio up front."""
    if not data:
        return ValidationResult(is_valid=False, error="Empty file provided.")

    if len(data) > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            error=(
                f"File size {len(data)} bytes exceeds maximum of "
                f"{MAX_FILE_SIZE_BYTES} bytes (50 MB)."
            ),
        )

    detected = detect_audio_format(data)
    if detected is None or detected not in SUPPORTED_FORMATS:
        return ValidationResult(
            is_valid=False,
            error=(
                "Unsupported audio format. Supported formats are: "
                f"{', '.join(sorted(SUPPORTED_FORMATS))}."
            ),
            detected_format=detected,
        )

    return ValidationResult(is_valid=True, detected_format=detected)


def get_wav_duration_seconds(data: bytes) -> float:
    """Read duration straight from a WAV file's header, before size checks."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate == 0:
                raise AudioValidationError("WAV file reports a 0 Hz sample rate.")
            return frames / float(rate)
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(f"Could not read WAV header: {exc}") from exc


def extract_audio_properties(data: bytes, audio_format: str) -> AudioProperties:
    """Extract duration, sample rate, and channel count for a known format."""
    try:
        if audio_format == "wav":
            with wave.open(io.BytesIO(data), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                duration = frames / float(rate) if rate else 0.0
                return AudioProperties(
                    duration_seconds=duration, sample_rate=rate, channels=channels
                )

        buffer = io.BytesIO(data)
        if audio_format == "mp3":
            info = MP3(buffer).info
        elif audio_format == "flac":
            info = FLAC(buffer).info
        elif audio_format == "m4a":
            info = MP4(buffer).info
        else:
            raise AudioValidationError(f"Unsupported format: {audio_format}")

        return AudioProperties(
            duration_seconds=float(info.length),
            sample_rate=int(getattr(info, "sample_rate", 0) or 0),
            channels=int(getattr(info, "channels", 0) or 0),
        )
    except AudioValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parse failure is a validation error
        raise AudioValidationError(
            f"Could not extract audio properties for {audio_format}: {exc}"
        ) from exc


def make_wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Build a minimal silent WAV byte string - used by tests and demos."""
    n_frames = max(1, int(duration_seconds * sample_rate))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)
    return buffer.getvalue()
