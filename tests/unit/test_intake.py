from src.agents.intake import run_intake
from src.graph.state import AudioInput
from src.utils.audio import make_wav_bytes


def test_valid_wav_intake():
    result = run_intake(AudioInput(audio_data=make_wav_bytes(5.0), filename="call.wav"))
    assert result.validation_passed is True
    assert result.temp_file_path is not None


def test_empty_file_rejected():
    result = run_intake(AudioInput(audio_data=b"", filename="empty.wav"))
    assert result.validation_passed is False
    assert "Empty file" in result.validation_error


def test_unsupported_format_rejected():
    data = b"\x00" * 100
    result = run_intake(AudioInput(audio_data=data, filename="bad.ogg"))
    assert result.validation_passed is False
    assert "Unsupported" in result.validation_error


def test_long_wav_rejected():
    result = run_intake(AudioInput(audio_data=make_wav_bytes(3601.0), filename="long.wav"))
    assert result.validation_passed is False
    assert "duration" in result.validation_error.lower()


def test_pii_in_caller_id_detected():
    result = run_intake(
        AudioInput(
            audio_data=make_wav_bytes(1.0),
            filename="call.wav",
            caller_id="SSN: 123-45-6789",
        )
    )
    assert result.pii_scan.pii_detected is True
    assert "caller_id" in result.pii_scan.affected_fields


def test_call_ids_are_unique():
    audio = AudioInput(audio_data=make_wav_bytes(1.0), filename="call.wav")
    r1 = run_intake(audio)
    r2 = run_intake(audio)
    assert r1.call_id != r2.call_id
