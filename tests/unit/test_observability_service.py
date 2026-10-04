from src.database.connection import get_engine, init_db, session_scope
from src.database.models import CallRecord
from src.security.audit import AuditLogger
from src.services.observability import get_observability_dashboard


def test_dashboard_reports_counts_and_success_rate():
    engine = get_engine(":memory:")
    init_db(engine)
    logger = AuditLogger(engine)

    with session_scope(engine) as session:
        session.add(
            CallRecord(
                call_id="c1",
                status="completed",
                qa_scores_json='{"overall_score": 4.0, "compliance_flags": []}',
            )
        )
        session.add(CallRecord(call_id="c2", status="failed"))

    logger.log("c1", "call_started")
    logger.log("c1", "call_completed")

    stats_html, tracing_html, rows = get_observability_dashboard(engine, False, None)

    assert ">2<" in stats_html  # Total Calls stat card value
    assert "Total Calls" in stats_html
    assert "Success Rate" in stats_html
    assert "<strong>LangSmith:</strong> Disabled" in tracing_html
    assert "<strong>Langfuse:</strong> Disabled" in tracing_html
    assert len(rows) == 2


def test_dashboard_reports_langfuse_enabled():
    engine = get_engine(":memory:")
    init_db(engine)

    _, tracing_html, _ = get_observability_dashboard(
        engine,
        langsmith_enabled=True,
        langsmith_project="my-project",
        langfuse_enabled=True,
        langfuse_host="https://cloud.langfuse.com",
    )

    assert "<strong>LangSmith:</strong> Enabled" in tracing_html
    assert "my-project" in tracing_html
    assert "<strong>Langfuse:</strong> Enabled" in tracing_html
    assert "https://cloud.langfuse.com" in tracing_html
