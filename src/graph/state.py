"""Typed data contracts shared between every pipeline stage.

Fourteen Pydantic models plus the LangGraph `PipelineState` TypedDict form
the boundary contracts between agents. Field constraints on scores and
confidences act as a schema-level guardrail: an out-of-range value raised by
an LLM or a bug never silently flows downstream.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #
class AudioInput(BaseModel):
    audio_data: bytes
    filename: str
    caller_id: str | None = None
    department: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AudioProperties(BaseModel):
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0


class PIIScanResult(BaseModel):
    pii_detected: bool = False
    affected_fields: list[str] = Field(default_factory=list)


class IntakeResult(BaseModel):
    call_id: str
    validation_passed: bool
    validation_error: str | None = None
    detected_format: str | None = None
    audio_properties: AudioProperties = Field(default_factory=AudioProperties)
    pii_scan: PIIScanResult = Field(default_factory=PIIScanResult)
    temp_file_path: str | None = None
    caller_id: str | None = None
    department: str | None = None


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class TranscriptionSegment(BaseModel):
    start: float
    end: float
    speaker: str  # "Agent" | "Customer"
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    flagged_low_confidence: bool = False


class TranscriptionResult(BaseModel):
    call_id: str
    full_text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    language: str = "en"
    from_cache: bool = False
    audio_hash: str | None = None
    injection_detected: bool = False
    matched_injection_patterns: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Summarization
# --------------------------------------------------------------------------- #
class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ESCALATED = "escalated"


class ActionItem(BaseModel):
    description: str
    owner: str
    deadline: str | None = None


class Entity(BaseModel):
    name: str
    type: str


class SummaryResult(BaseModel):
    call_id: str = ""
    call_purpose: str
    key_discussion_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    resolution_status: ResolutionStatus
    sentiment_trajectory: str
    entities: list[Entity] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# QA Scoring
# --------------------------------------------------------------------------- #
class QADimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    justification: str


class ComplianceFlag(BaseModel):
    description: str
    severity: str  # "low" | "medium" | "high" | "critical"
    timestamp_reference: str | None = None


class QAScoreResult(BaseModel):
    call_id: str = ""
    professionalism: QADimensionScore
    empathy: QADimensionScore
    problem_resolution: QADimensionScore
    compliance: QADimensionScore
    communication_clarity: QADimensionScore
    overall_score: float = Field(ge=1.0, le=5.0)
    compliance_flags: list[ComplianceFlag] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
class CallReport(BaseModel):
    call_id: str
    status: str
    transcript: TranscriptionResult | None = None
    summary: SummaryResult | None = None
    qa_scores: QAScoreResult | None = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# LangGraph pipeline state
# --------------------------------------------------------------------------- #
class PipelineState(TypedDict, total=False):
    audio_input: AudioInput
    intake: IntakeResult
    transcription: TranscriptionResult
    summary: SummaryResult
    qa_scores: QAScoreResult
    report: CallReport
    error: str
    status: str
    call_id: str
    metadata: dict[str, Any]
