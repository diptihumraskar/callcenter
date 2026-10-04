from src.graph.edges import route_after_intake, route_after_qa
from src.graph.state import ComplianceFlag, IntakeResult, QADimensionScore, QAScoreResult


def _intake(passed: bool) -> IntakeResult:
    return IntakeResult(
        call_id="c1", validation_passed=passed, validation_error=None if passed else "bad"
    )


def _qa(flags: list[ComplianceFlag]) -> QAScoreResult:
    dim = QADimensionScore(score=3, justification="ok")
    return QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=3.0,
        compliance_flags=flags,
    )


def test_route_after_intake_valid():
    assert route_after_intake(_intake(True)) == "transcribe"


def test_route_after_intake_invalid():
    assert route_after_intake(_intake(False)) == "error"


def test_route_after_qa_critical_flag():
    flags = [ComplianceFlag(description="skipped ID verification", severity="critical")]
    assert route_after_qa(_qa(flags)) == "supervisor_review"


def test_route_after_qa_no_flags():
    assert route_after_qa(_qa([])) == "report"
