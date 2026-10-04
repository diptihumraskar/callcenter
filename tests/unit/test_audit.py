from sqlalchemy import select

from src.database.connection import get_engine, init_db, session_scope
from src.database.models import AuditLogEntry
from src.security.audit import AuditLogger


def test_log_writes_details_as_json():
    engine = get_engine(":memory:")
    init_db(engine)
    logger = AuditLogger(engine)

    logger.log("call-1", "call_flagged", details={"reason": "injection"})

    with session_scope(engine) as session:
        entry = session.execute(select(AuditLogEntry)).scalar_one()
        assert entry.action == "call_flagged"
        assert "injection" in entry.details


def test_log_without_details_is_none():
    engine = get_engine(":memory:")
    init_db(engine)
    logger = AuditLogger(engine)

    logger.log("call-1", "call_started")

    with session_scope(engine) as session:
        entry = session.execute(select(AuditLogEntry)).scalar_one()
        assert entry.details is None


def test_multiple_log_entries_are_append_only():
    engine = get_engine(":memory:")
    init_db(engine)
    logger = AuditLogger(engine)

    logger.log("call-1", "call_started")
    logger.log("call-1", "call_completed")

    with session_scope(engine) as session:
        entries = session.execute(select(AuditLogEntry)).scalars().all()
        assert len(entries) == 2
