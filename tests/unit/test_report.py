from sqlalchemy import select

from src.agents.report import (
    compile_report,
    generate_report_json,
    generate_report_pdf,
    persist_report,
)
from src.database.connection import get_engine, init_db, session_scope
from src.database.models import CallRecord
from src.graph.state import (
    QADimensionScore,
    QAScoreResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
)


def _summary():
    return SummaryResult(
        call_purpose="Test",
        resolution_status=ResolutionStatus.RESOLVED,
        sentiment_trajectory="Neutral",
    )


def _qa():
    dim = QADimensionScore(score=4, justification="fine")
    return QAScoreResult(
        professionalism=dim,
        empathy=dim,
        problem_resolution=dim,
        compliance=dim,
        communication_clarity=dim,
        overall_score=4.0,
    )


def test_compile_report_propagates_call_id():
    transcript = TranscriptionResult(call_id="call-1", full_text="hi")
    report = compile_report("call-1", "completed", transcript, _summary(), _qa())
    assert report.call_id == "call-1"
    assert report.transcript.call_id == "call-1"


def test_generate_report_pdf_returns_pdf_bytes():
    report = compile_report("call-1", "completed", None, _summary(), _qa())
    pdf_bytes = generate_report_pdf(report)
    assert pdf_bytes[:5] == b"%PDF-"


def test_generate_report_json_contains_call_id_and_summary():
    report = compile_report("call-1", "completed", None, _summary(), _qa())
    json_str = generate_report_json(report)
    assert '"call_id"' in json_str
    assert '"summary"' in json_str


def test_persist_report_writes_and_is_retrievable():
    engine = get_engine(":memory:")
    init_db(engine)
    report = compile_report("call-1", "completed", None, _summary(), _qa())

    persist_report(engine, report, audio_filename="call.wav")

    with session_scope(engine) as session:
        record = session.execute(
            select(CallRecord).where(CallRecord.call_id == "call-1")
        ).scalar_one()
        assert record.status == "completed"
