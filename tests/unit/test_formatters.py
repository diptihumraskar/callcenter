from src.graph.state import QADimensionScore, QAScoreResult
from src.utils.formatters import format_qa, secs_to_mmss


def test_secs_to_mmss_zero():
    assert secs_to_mmss(0.0) == "00:00"


def test_secs_to_mmss_ninety():
    assert secs_to_mmss(90.0) == "01:30"


def test_format_qa_no_compliance_flags():
    dim = QADimensionScore(score=4, justification="solid")
    qa = QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=4.0,
        compliance_flags=[],
    )
    output = format_qa(qa)
    assert "No compliance issues detected." in output
