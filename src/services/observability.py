"""Observability service: aggregates pipeline health metrics for the dashboard."""

from __future__ import annotations

import json

from sqlalchemy import Engine, func, select

from src.database.connection import session_scope
from src.database.models import AuditLogEntry, CallRecord


def _stat_card(value: str, label: str, accent: str = "#4f46e5") -> str:
    return (
        '<div class="cci-stat-card">'
        f'<div class="cci-stat-value" style="color:{accent}">{value}</div>'
        f'<div class="cci-stat-label">{label}</div>'
        "</div>"
    )


def _tracing_card(enabled: bool, name: str, detail: str, link: str | None = None) -> str:
    dot_color = "#10b981" if enabled else "#94a3b8"
    status_text = "Enabled" if enabled else "Disabled"
    body = f"<strong>{name}:</strong> {status_text}"
    if enabled and detail:
        body += f"<br><span style='font-size:0.85rem;color:#64748b'>{detail}</span>"
    if enabled and link:
        body += f"<br><a href='{link}' target='_blank'>Open dashboard →</a>"
    return (
        '<div class="cci-tracing-card">'
        f'<span class="cci-status-dot" style="background:{dot_color}"></span>{body}'
        "</div>"
    )


def get_observability_dashboard(
    engine: Engine,
    langsmith_enabled: bool,
    langsmith_project: str | None,
    langfuse_enabled: bool = False,
    langfuse_host: str | None = None,
) -> tuple[str, str, list[list[str]]]:
    """Return (stats_html, tracing_html, recent_audit_rows)."""
    with session_scope(engine) as session:
        total = session.execute(select(func.count(CallRecord.id))).scalar_one()
        completed = session.execute(
            select(func.count(CallRecord.id)).where(CallRecord.status == "completed")
        ).scalar_one()
        failed = session.execute(
            select(func.count(CallRecord.id)).where(CallRecord.status == "failed")
        ).scalar_one()
        flagged = session.execute(
            select(func.count(CallRecord.id)).where(CallRecord.status == "flagged_for_review")
        ).scalar_one()

        avg_qa_scores = []
        for row in session.execute(
            select(CallRecord.qa_scores_json).where(CallRecord.qa_scores_json.is_not(None))
        ).scalars():
            try:
                avg_qa_scores.append(json.loads(row)["overall_score"])
            except (KeyError, TypeError, ValueError):
                continue
        avg_qa = sum(avg_qa_scores) / len(avg_qa_scores) if avg_qa_scores else 0.0

        total_flags = 0
        for row in session.execute(
            select(CallRecord.qa_scores_json).where(CallRecord.qa_scores_json.is_not(None))
        ).scalars():
            try:
                total_flags += len(json.loads(row).get("compliance_flags", []))
            except (TypeError, ValueError):
                continue

        total_events = session.execute(select(func.count(AuditLogEntry.id))).scalar_one()

        recent = (
            session.execute(
                select(AuditLogEntry).order_by(AuditLogEntry.timestamp.desc()).limit(20)
            )
            .scalars()
            .all()
        )
        recent_rows = [
            [
                entry.timestamp.isoformat(sep=" ", timespec="seconds"),
                entry.call_id,
                entry.action,
                entry.details or "",
            ]
            for entry in recent
        ]

    success_rate = (completed / total * 100) if total else 0.0

    stats_html = (
        '<div class="cci-stat-row">'
        + "".join(
            [
                _stat_card(str(total), "Total Calls", "#4f46e5"),
                _stat_card(f"{success_rate:.0f}%", "Success Rate", "#10b981"),
                _stat_card(f"{avg_qa:.2f} / 5.00", "Avg QA Score", "#4f46e5"),
                _stat_card(str(flagged), "Flagged for Review", "#f59e0b"),
                _stat_card(str(failed), "Failed", "#ef4444"),
                _stat_card(str(total_flags), "Compliance Flags", "#f59e0b"),
                _stat_card(str(total_events), "Audit Events", "#4f46e5"),
            ]
        )
        + "</div>"
    )

    tracing_html = (
        '<div class="cci-tracing-row">'
        + "".join(
            [
                _tracing_card(
                    langsmith_enabled,
                    "LangSmith",
                    f"project: {langsmith_project}" if langsmith_project else "",
                    "https://smith.langchain.com/" if langsmith_enabled else None,
                ),
                _tracing_card(
                    langfuse_enabled,
                    "Langfuse",
                    "",
                    langfuse_host if langfuse_enabled else None,
                ),
            ]
        )
        + "</div>"
    )

    return stats_html, tracing_html, recent_rows
