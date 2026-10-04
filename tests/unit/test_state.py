import pytest
from pydantic import ValidationError

from src.graph.state import QADimensionScore, TranscriptionSegment


def test_transcription_segment_rejects_high_confidence():
    with pytest.raises(ValidationError):
        TranscriptionSegment(start=0.0, end=1.0, speaker="Agent", text="hi", confidence=1.5)


def test_transcription_segment_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        TranscriptionSegment(start=0.0, end=1.0, speaker="Agent", text="hi", confidence=-0.1)


def test_qa_dimension_score_rejects_zero():
    with pytest.raises(ValidationError):
        QADimensionScore(score=0, justification="too low")


def test_qa_dimension_score_rejects_above_five():
    with pytest.raises(ValidationError):
        QADimensionScore(score=6, justification="too high")


def test_qa_dimension_score_accepts_valid_range():
    score = QADimensionScore(score=3, justification="baseline")
    assert score.score == 3
