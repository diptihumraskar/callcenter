from src.utils.audio import (
    MAX_FILE_SIZE_BYTES,
    detect_audio_format,
    extract_audio_properties,
    make_wav_bytes,
    validate_audio_file,
)


def test_detect_wav():
    assert detect_audio_format(make_wav_bytes()) == "wav"


def test_detect_mp3_id3():
    data = b"ID3" + b"\x00" * 20
    assert detect_audio_format(data) == "mp3"


def test_detect_mp3_sync_bits():
    data = b"\xff\xfb\x90\x00" + b"\x00" * 100
    assert detect_audio_format(data) == "mp3"


def test_detect_flac():
    data = b"fLaC" + b"\x00" * 20
    assert detect_audio_format(data) == "flac"


def test_detect_m4a():
    data = b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 20
    assert detect_audio_format(data) == "m4a"


def test_detect_unsupported_returns_none():
    data = b"OggS" + b"\x00" * 20
    assert detect_audio_format(data) is None


def test_validate_empty_file():
    result = validate_audio_file(b"", "empty.wav")
    assert result.is_valid is False
    assert "empty" in (result.error or "").lower()


def test_validate_oversized_file():
    data = b"\x00" * (MAX_FILE_SIZE_BYTES + 1)
    result = validate_audio_file(data, "big.wav")
    assert result.is_valid is False
    assert "exceeds maximum" in result.error


def test_validate_unsupported_format():
    data = b"OggS" + b"\x00" * 20
    result = validate_audio_file(data, "bad.ogg")
    assert result.is_valid is False
    assert "Unsupported" in result.error


def test_validate_valid_wav():
    result = validate_audio_file(make_wav_bytes(2.0), "call.wav")
    assert result.is_valid is True
    assert result.detected_format == "wav"


def test_extract_wav_properties():
    data = make_wav_bytes(5.0, 16000)
    props = extract_audio_properties(data, "wav")
    assert 4.9 <= props.duration_seconds <= 5.1
    assert props.sample_rate == 16000
    assert props.channels == 1
