"""Call History tab: a searchable master-detail browser over every analyzed call."""

from __future__ import annotations

import json

import gradio as gr
from sqlalchemy import Engine, select

from src.database.connection import session_scope
from src.database.models import CallRecord
from src.ui.theme import status_banner

_HEADERS = ["Call ID", "Filename", "Status", "Processed At"]

_STATUS_ICONS = {"completed": "✅", "flagged_for_review": "🚩", "failed": "⚠️"}


def _load_call_rows(engine: Engine, search: str = "") -> list[list[str]]:
    with session_scope(engine) as session:
        records = (
            session.execute(select(CallRecord).order_by(CallRecord.processed_at.desc()))
            .scalars()
            .all()
        )
        rows = [
            [
                r.call_id,
                r.audio_filename or "",
                r.status,
                r.processed_at.isoformat(sep=" ", timespec="seconds"),
            ]
            for r in records
        ]
    if search:
        needle = search.strip().lower()
        rows = [row for row in rows if needle in row[0].lower() or needle in row[1].lower()]
    return rows


def _load_call_detail(engine: Engine, call_id: str) -> tuple[str, str, str, str]:
    if not call_id:
        return status_banner("neutral", "Paste a Call ID above to inspect a call."), "", "", ""
    with session_scope(engine) as session:
        record = session.execute(
            select(CallRecord).where(CallRecord.call_id == call_id)
        ).scalar_one_or_none()
        if record is None:
            return status_banner("failed", "No call found with that ID.", icon="⚠️"), "", "", ""

        status_kind = "success" if record.status == "completed" else record.status
        processed_at = record.processed_at.isoformat(sep=" ", timespec="seconds")
        banner = status_banner(
            status_kind,
            f"<strong>{record.audio_filename or record.call_id}</strong> — "
            f"status: {record.status}, processed {processed_at}",
            icon=_STATUS_ICONS.get(record.status, "ℹ️"),
        )

        transcript = record.transcript_text or "(no transcript)"
        summary_md = "_No summary available._"
        if record.summary_json:
            summary_md = f"```json\n{json.dumps(json.loads(record.summary_json), indent=2)}\n```"
        qa_md = "_No QA scores available._"
        if record.qa_scores_json:
            qa_md = f"```json\n{json.dumps(json.loads(record.qa_scores_json), indent=2)}\n```"
        return banner, transcript, summary_md, qa_md


def build_history_tab(engine: Engine) -> None:
    with gr.Tab("📁 Call History") as tab:
        gr.Markdown(
            '<div class="cci-section-title">📁 Every call ever analyzed</div>'
            "Search by Call ID or filename, then paste a Call ID below to inspect it."
        )

        with gr.Row():
            search_box = gr.Textbox(
                label="Search",
                placeholder="Filter by Call ID or filename…",
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        table = gr.Dataframe(
            headers=_HEADERS,
            interactive=False,
            row_count=(0, "dynamic"),
        )

        gr.Markdown('<div class="cci-section-title">🔎 Inspect a call</div>')
        call_id_box = gr.Textbox(
            label="Call ID", placeholder="Paste a Call ID from the table above"
        )
        detail_banner = gr.HTML()

        with gr.Tabs():
            with gr.Tab("📝 Transcript"):
                detail_transcript = gr.Textbox(show_label=False, lines=10, interactive=False)
            with gr.Tab("🗒️ Summary"):
                detail_summary = gr.Markdown()
            with gr.Tab("✅ QA Scores"):
                detail_qa = gr.Markdown()

        def _refresh(search):
            return _load_call_rows(engine, search)

        def _inspect(call_id):
            return _load_call_detail(engine, call_id)

        tab.select(_refresh, inputs=search_box, outputs=table)
        refresh_btn.click(_refresh, inputs=search_box, outputs=table)
        search_box.submit(_refresh, inputs=search_box, outputs=table)
        call_id_box.change(
            _inspect,
            inputs=call_id_box,
            outputs=[detail_banner, detail_transcript, detail_summary, detail_qa],
        )
