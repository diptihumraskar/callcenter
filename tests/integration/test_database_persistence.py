from sqlalchemy import select

from src.database.connection import get_engine, init_db, session_scope
from src.database.models import AuditLogEntry, CallRecord
from src.security.audit import AuditLogger


def test_call_record_insert_and_retrieve():
    engine = get_engine(":memory:")
    init_db(engine)

    with session_scope(engine) as session:
        session.add(CallRecord(call_id="abc-123", status="completed"))

    with session_scope(engine) as session:
        record = session.execute(
            select(CallRecord).where(CallRecord.call_id == "abc-123")
        ).scalar_one()
        assert record.status == "completed"


def test_audit_log_entry_insert_and_retrieve():
    engine = get_engine(":memory:")
    init_db(engine)
    logger = AuditLogger(engine)

    logger.log("abc-123", "call_started", details={"source": "test"})

    with session_scope(engine) as session:
        entries = (
            session.execute(select(AuditLogEntry).where(AuditLogEntry.call_id == "abc-123"))
            .scalars()
            .all()
        )
        assert len(entries) == 1
        assert entries[0].action == "call_started"
