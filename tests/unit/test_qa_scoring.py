from unittest.mock import MagicMock, patch

import pytest

from src.agents.qa_scoring import QAScoringError, run_qa_scoring
from src.graph.state import (
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
)


def _transcript():
    return TranscriptionResult(call_id="call-123", full_text="hello world")


def _summary():
    return SummaryResult(
        call_purpose="Billing question",
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="Neutral",
    )


def _perfect_qa_but_llm_lowballs_overall():
    dim = QADimensionScore(score=5, justification="excellent")
    return QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=3.0,  # LLM's own (wrong) score - must be discarded
        compliance_flags=[],
    )


@patch("time.sleep", return_value=None)
def test_overall_score_is_recomputed_deterministically(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = (
        _perfect_qa_but_llm_lowballs_overall()
    )

    result = run_qa_scoring(_transcript(), _summary(), llm)

    assert result.overall_score == 5.0
    assert result.call_id == "call-123"


@patch("time.sleep", return_value=None)
def test_raises_after_max_retries(_mock_sleep):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = Exception("boom")

    with pytest.raises(QAScoringError):
        run_qa_scoring(_transcript(), _summary(), llm, max_retries=3)

    assert llm.with_structured_output.return_value.invoke.call_count == 3
