"""LangGraph orchestration: seven pipeline stages plus error/supervisor nodes."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, StateGraph
from sqlalchemy import Engine

from src.agents.intake import run_intake
from src.agents.qa_scoring import QAScoringError, run_qa_scoring
from src.agents.report import compile_report, persist_report
from src.agents.speaker_labeling import SpeakerLabelingError, run_speaker_labeling
from src.agents.summarization import SummarizationError, run_summarization
from src.agents.transcription import run_transcription
from src.graph.edges import route_after_intake, route_after_qa
from src.graph.state import PipelineState
from src.security.audit import AuditLogger
from src.security.injection_detector import detect_injection
from src.security.pii_redactor import redact_pii

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover - optional dependency

    def traceable(*_args, **_kwargs):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn

        return _decorator


@dataclass
class WorkflowDeps:
    """Runtime dependencies injected into every node closure."""

    engine: Engine
    llm_provider: str
    whisper_model_size: str
    audit_logger: AuditLogger
    max_retries: int = 3


def compile_workflow(deps: WorkflowDeps):
    """Build and compile the seven-stage StateGraph with conditional routing."""
    from src.utils.llm_factory import get_llm  # local import keeps CLI/test cost low

    graph = StateGraph(PipelineState)

    @traceable(name="intake_step")
    def intake_step(state: PipelineState) -> dict:
        intake = run_intake(state["audio_input"])
        deps.audit_logger.log(intake.call_id, "call_started")
        if not intake.validation_passed:
            deps.audit_logger.log(intake.call_id, "call_failed", details={"stage": "intake"})
        return {"intake": intake, "call_id": intake.call_id}

    @traceable(name="transcription_node")
    def transcription_node(state: PipelineState) -> dict:
        transcription = run_transcription(
            state["intake"], engine=deps.engine, model_size=deps.whisper_model_size
        )
        return {"transcription": transcription}

    @traceable(name="injection_check_node")
    def injection_check_node(state: PipelineState) -> dict:
        transcription = state["transcription"]
        scan = detect_injection(transcription.full_text)
        transcription.injection_detected = scan.injection_detected
        transcription.matched_injection_patterns = scan.matched_patterns
        if scan.injection_detected:
            deps.audit_logger.log(
                state["call_id"],
                "call_flagged",
                details={"stage": "injection_detection", "patterns": scan.matched_patterns},
            )
            return {
                "transcription": transcription,
                "status": "flagged_for_review",
                "error": f"Prompt injection detected: {', '.join(scan.matched_patterns)}",
            }
        return {"transcription": transcription}

    @traceable(name="pii_redaction_node")
    def pii_redaction_node(state: PipelineState) -> dict:
        transcription = state["transcription"]
        full = redact_pii(transcription.full_text)
        transcription.full_text = full.redacted_text
        for segment in transcription.segments:
            segment.text = redact_pii(segment.text).redacted_text
        return {"transcription": transcription}

    @traceable(name="speaker_labeling_node")
    def speaker_labeling_node(state: PipelineState) -> dict:
        llm = get_llm(deps.llm_provider)
        try:
            transcription = run_speaker_labeling(
                state["transcription"], llm, max_retries=deps.max_retries
            )
        except SpeakerLabelingError as exc:
            deps.audit_logger.log(
                state["call_id"],
                "call_failed",
                details={"stage": "speaker_labeling", "error": str(exc)},
            )
            return {"status": "failed", "error": str(exc)}
        return {"transcription": transcription}

    @traceable(name="summarize_and_qa_node")
    def summarize_and_qa_node(state: PipelineState) -> dict:
        llm = get_llm(deps.llm_provider)
        try:
            summary = run_summarization(state["transcription"], llm, max_retries=deps.max_retries)
            qa_scores = run_qa_scoring(
                state["transcription"], summary, llm, max_retries=deps.max_retries
            )
        except (SummarizationError, QAScoringError) as exc:
            deps.audit_logger.log(
                state["call_id"],
                "call_failed",
                details={"stage": "llm_analysis", "error": str(exc)},
            )
            return {"status": "failed", "error": str(exc)}
        return {"summary": summary, "qa_scores": qa_scores}

    @traceable(name="report_node")
    def report_node(state: PipelineState) -> dict:
        report = compile_report(
            state["call_id"],
            "completed",
            state.get("transcription"),
            state.get("summary"),
            state.get("qa_scores"),
        )
        persist_report(deps.engine, report, audio_filename=state["audio_input"].filename)
        deps.audit_logger.log(state["call_id"], "call_completed")
        return {"report": report, "status": "completed"}

    @traceable(name="error_node")
    def error_node(state: PipelineState) -> dict:
        error = (
            state.get("error")
            or state.get("intake", None)
            and getattr(state.get("intake"), "validation_error", None)
            or "Validation failed"
        )
        # Preserve a status already set upstream (e.g. "flagged_for_review" from
        # the injection check) - only default to "failed" when nothing set one.
        status = state.get("status") or "failed"
        report = compile_report(
            state.get("call_id", "unknown"),
            status,
            state.get("transcription"),
            state.get("summary"),
            state.get("qa_scores"),
        )
        return {"report": report, "status": status, "error": error}

    @traceable(name="supervisor_review_node")
    def supervisor_review_node(state: PipelineState) -> dict:
        report = compile_report(
            state["call_id"],
            "flagged_for_review",
            state.get("transcription"),
            state.get("summary"),
            state.get("qa_scores"),
        )
        persist_report(deps.engine, report, audio_filename=state["audio_input"].filename)
        deps.audit_logger.log(state["call_id"], "call_flagged", details={"stage": "qa_scoring"})
        return {"report": report, "status": "flagged_for_review"}

    graph.add_node("intake_step", intake_step)
    graph.add_node("transcribe_step", transcription_node)
    graph.add_node("injection_check_step", injection_check_node)
    graph.add_node("pii_redact_step", pii_redaction_node)
    graph.add_node("speaker_labeling_step", speaker_labeling_node)
    graph.add_node("summarize_and_qa_step", summarize_and_qa_node)
    graph.add_node("report_step", report_node)
    graph.add_node("error_step", error_node)
    graph.add_node("supervisor_step", supervisor_review_node)

    graph.set_entry_point("intake_step")

    graph.add_conditional_edges(
        "intake_step",
        lambda state: route_after_intake(state["intake"]),
        {"transcribe": "transcribe_step", "error": "error_step"},
    )
    graph.add_edge("transcribe_step", "injection_check_step")
    graph.add_conditional_edges(
        "injection_check_step",
        lambda state: "error" if state.get("status") == "flagged_for_review" else "continue",
        {"continue": "pii_redact_step", "error": "error_step"},
    )
    graph.add_edge("pii_redact_step", "speaker_labeling_step")
    graph.add_conditional_edges(
        "speaker_labeling_step",
        lambda state: "error" if state.get("status") == "failed" else "continue",
        {"continue": "summarize_and_qa_step", "error": "error_step"},
    )
    graph.add_conditional_edges(
        "summarize_and_qa_step",
        lambda state: (
            "error" if state.get("status") == "failed" else route_after_qa(state["qa_scores"])
        ),
        {
            "report": "report_step",
            "supervisor_review": "supervisor_step",
            "error": "error_step",
        },
    )
    graph.add_edge("report_step", END)
    graph.add_edge("error_step", END)
    graph.add_edge("supervisor_step", END)

    return graph.compile()
