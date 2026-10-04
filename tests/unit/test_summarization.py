from unittest.mock import MagicMock, patch

import pytest

from src.agents.summarization import SummarizationError, run_summarization
from src.graph.state import ResolutionStatus, SummaryResult, TranscriptionResult


def _transcript():
    return TranscriptionResult(call_id="call-123", full_text="hello world")


def _summary_result():
    return SummaryResult(
        call_purpose="Billing question",
        key_discussion_points=["Overcharge on last invoice"],
        action_items=[],
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="Frustrated -> Satisfied",
        entities=[],
    )


@patch("time.sleep", return_value=None)
def test_run_summarization_sets_call_id(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = _summary_result()

    result = run_summarization(_transcript(), llm)

    assert result.call_id == "call-123"


@patch("time.sleep", return_value=None)
def test_run_summarization_raises_after_max_retries(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = Exception("boom")

    with pytest.raises(SummarizationError):
        run_summarization(_transcript(), llm, max_retries=3)

    assert llm.with_structured_output.return_value.invoke.call_count == 3
