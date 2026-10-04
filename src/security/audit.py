"""Append-only audit logging for every pipeline event."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import Engine

from src.database.connection import session_scope
from src.database.models import AuditLogEntry


class AuditLogger:
    """Writes timestamped, immutable audit records. Never deletes or updates."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def log(
        self,
        call_id: str,
        action: str,
        user: str = "system",
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = AuditLogEntry(
            call_id=call_id,
            action=action,
            user=user,
            details=json.dumps(details) if details is not None else None,
        )
        with session_scope(self._engine) as session:
            session.add(entry)
