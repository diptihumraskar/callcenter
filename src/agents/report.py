"""Report agent: assembles, persists, and renders the final call report."""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sqlalchemy import Engine

from src.database.connection import session_scope
from src.database.models import CallRecord
from src.graph.state import CallReport, QAScoreResult, SummaryResult, TranscriptionResult


def compile_report(
    call_id: str,
    status: str,
    transcript: TranscriptionResult | None,
    summary: SummaryResult | None,
    qa_scores: QAScoreResult | None,
) -> CallReport:
    """Assemble a CallReport from every upstream stage's result."""
    return CallReport(
        call_id=call_id,
        status=status,
        transcript=transcript,
        summary=summary,
        qa_scores=qa_scores,
    )


def persist_report(engine: Engine, report: CallReport, audio_filename: str | None = None) -> None:
    """Write (or upsert) a CallRecord row for this report."""
    with session_scope(engine) as session:
        record = CallRecord(
            call_id=report.call_id,
            status=report.status,
            audio_filename=audio_filename,
            transcript_text=report.transcript.full_text if report.transcript else None,
            summary_json=report.summary.model_dump_json() if report.summary else None,
            qa_scores_json=report.qa_scores.model_dump_json() if report.qa_scores else None,
            report_json=report.model_dump_json(),
        )
        session.add(record)


def generate_report_pdf(report: CallReport) -> bytes:
    """Render summary, QA scores, and compliance flags to a PDF, as bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Call Report — {report.call_id}", styles["Title"]), Spacer(1, 12)]

    if report.summary:
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(Paragraph(f"Purpose: {report.summary.call_purpose}", styles["Normal"]))
        story.append(
            Paragraph(f"Resolution: {report.summary.resolution_status.value}", styles["Normal"])
        )
        story.append(
            Paragraph(f"Sentiment: {report.summary.sentiment_trajectory}", styles["Normal"])
        )
        story.append(Spacer(1, 12))

    if report.qa_scores:
        qa = report.qa_scores
        story.append(Paragraph("QA Scores", styles["Heading2"]))
        story.append(Paragraph(f"Overall: {qa.overall_score:.2f} / 5.00", styles["Normal"]))
        for label, dim in (
            ("Professionalism", qa.professionalism),
            ("Empathy", qa.empathy),
            ("Problem Resolution", qa.problem_resolution),
            ("Compliance", qa.compliance),
            ("Communication Clarity", qa.communication_clarity),
        ):
            story.append(
                Paragraph(f"{label}: {dim.score}/5 — {dim.justification}", styles["Normal"])
            )
        story.append(Spacer(1, 12))

        if qa.compliance_flags:
            story.append(Paragraph("Compliance Flags", styles["Heading2"]))
            for flag in qa.compliance_flags:
                story.append(
                    Paragraph(f"[{flag.severity.upper()}] {flag.description}", styles["Normal"])
                )

    doc.build(story)
    return buffer.getvalue()


def generate_report_json(report: CallReport) -> str:
    return report.model_dump_json(indent=2)
