from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agents.intake import run_intake
from src.agents.transcription import run_transcription
from src.database.connection import get_engine, init_db
from src.graph.state import AudioInput
from src.utils.audio import make_wav_bytes


def _fake_segment(start, end, text, avg_logprob=-0.1, no_speech_prob=0.05):
    return SimpleNamespace(
        start=start, end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob
    )


def _make_intake():
    return run_intake(AudioInput(audio_data=make_wav_bytes(2.0), filename="call.wav"))


@patch("src.agents.transcription._get_whisper_model")
def test_transcription_returns_result_for_call_id(mock_get_model):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment(0.0, 1.0, "Thank you for calling support.")],
        SimpleNamespace(language="en"),
    )
    mock_get_model.return_value = mock_model

    intake = _make_intake()
    result = run_transcription(intake, engine=None, model_size="tiny")

    assert result.call_id == intake.call_id
    assert len(result.segments) == 1
    assert result.segments[0].speaker == "Speaker 1"


@patch("src.agents.transcription._get_whisper_model")
def test_cache_hit_skips_model_call(mock_get_model):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_fake_segment(0.0, 1.0, "Thank you for calling support.")],
        SimpleNamespace(language="en"),
    )
    mock_get_model.return_value = mock_model

    engine = get_engine(":memory:")
    init_db(engine)

    intake1 = _make_intake()
    result1 = run_transcription(intake1, engine=engine, model_size="tiny")
    assert result1.from_cache is False
    assert mock_model.transcribe.call_count == 1

    # Second intake re-hashes the SAME underlying audio bytes by reusing the
    # temp file content, so the cache should short-circuit the model call.
    intake2 = _make_intake()
    with open(intake2.temp_file_path, "wb") as f:
        with open(intake1.temp_file_path, "rb") as src:
            f.write(src.read())

    result2 = run_transcription(intake2, engine=engine, model_size="tiny")
    assert result2.from_cache is True
    assert result2.call_id == intake2.call_id
    assert mock_model.transcribe.call_count == 1


def test_clean_transcript_removes_artifacts():
    from src.agents.transcription import _clean_transcript_text

    assert "[BLANK_AUDIO]" not in _clean_transcript_text("[BLANK_AUDIO] hello")
    cleaned = _clean_transcript_text("thank you thank you thank you")
    assert cleaned.lower().count("thank you") == 1
