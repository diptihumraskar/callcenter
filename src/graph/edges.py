"""Conditional routing functions for the LangGraph state machine."""

from __future__ import annotations

from src.graph.state import IntakeResult, PipelineState, QAScoreResult


def route_after_intake(intake: IntakeResult) -> str:
    return "transcribe" if intake.validation_passed else "error"


def route_after_transcription(_state: PipelineState) -> str:
    return "summarize"


def route_after_qa(qa: QAScoreResult) -> str:
    if any(flag.severity == "critical" for flag in qa.compliance_flags):
        return "supervisor_review"
    return "report"
