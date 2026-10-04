from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.speaker_labeling import SpeakerLabelingError, run_speaker_labeling
from src.graph.state import TranscriptionResult, TranscriptionSegment


def _transcript(n: int) -> TranscriptionResult:
    segments = [
        TranscriptionSegment(
            start=float(i),
            end=float(i) + 1.0,
            speaker="Speaker 1",
            text=f"line {i}",
            confidence=0.9,
        )
        for i in range(n)
    ]
    full_text = " ".join(s.text for s in segments)
    return TranscriptionResult(call_id="call-123", full_text=full_text, segments=segments)


def _entry(index, speaker):
    return SimpleNamespace(segment_index=index, speaker=speaker)


def _assignment(*roles: str):
    return SimpleNamespace(segments=[_entry(i, role) for i, role in enumerate(roles)])


@patch("time.sleep", return_value=None)
def test_labels_each_segment_from_llm_response(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = _assignment(
        "Agent", "Agent", "Customer", "Agent"
    )

    result = run_speaker_labeling(_transcript(4), llm)

    assert [seg.speaker for seg in result.segments] == ["Agent", "Agent", "Customer", "Agent"]


@patch("time.sleep", return_value=None)
def test_allows_long_same_speaker_runs(_mock_sleep):
    """A long monologue split into many segments should stay one speaker."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = _assignment(
        "Agent", "Agent", "Agent", "Agent", "Agent", "Customer"
    )

    result = run_speaker_labeling(_transcript(6), llm)

    assert [seg.speaker for seg in result.segments] == [
        "Agent",
        "Agent",
        "Agent",
        "Agent",
        "Agent",
        "Customer",
    ]


def test_noop_when_no_segments():
    llm = MagicMock()

    result = run_speaker_labeling(_transcript(0), llm)

    assert result.segments == []
    llm.with_structured_output.assert_not_called()


@patch("time.sleep", return_value=None)
def test_retries_when_llm_omits_segments(_mock_sleep):
    llm = MagicMock()
    # First response is missing segment 1's label; second is complete.
    llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(segments=[_entry(0, "Agent")]),
        _assignment("Agent", "Customer"),
    ]

    result = run_speaker_labeling(_transcript(2), llm, max_retries=3)

    assert [seg.speaker for seg in result.segments] == ["Agent", "Customer"]
    assert llm.with_structured_output.return_value.invoke.call_count == 2


@patch("time.sleep", return_value=None)
def test_raises_after_max_retries(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = Exception("boom")

    with pytest.raises(SpeakerLabelingError):
        run_speaker_labeling(_transcript(2), llm, max_retries=3)

    assert llm.with_structured_output.return_value.invoke.call_count == 3
