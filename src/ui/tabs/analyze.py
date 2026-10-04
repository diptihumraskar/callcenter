"""Analyze Call tab: upload/record audio, run the pipeline, view results."""

from __future__ import annotations

import gradio as gr

from src.services.pipeline import process_call
from src.ui.theme import status_banner

_EXAMPLE_CALLS = [
    "data/samples/sample_01.mp3",
    "data/samples/sample_02.mp3",
    "data/samples/amazon_order_delayed.mp3",
    "data/samples/amazon_order_wrong_item.mp3",
]

_STATUS_ICONS = {
    "flagged_for_review": "🚩",
    "failed": "⚠️",
}

_STATUS_LABELS = {
    "flagged_for_review": "Flagged for supervisor review",
    "failed": "Processing failed",
}


def build_analyze_tab(workflow, langfuse_handler=None) -> None:
    with gr.Tab("🎧 Analyze Call"):
        gr.Markdown(
            '<div class="cci-section-title">📞 Submit a call for analysis</div>'
            "Upload a recording, drop in one of the samples below, or record "
            "directly from your microphone."
        )

        with gr.Group():
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="Call Recording",
            )
            gr.Examples(
                examples=_EXAMPLE_CALLS,
                inputs=audio_input,
                label="Or try a sample call",
                examples_per_page=4,
            )
            with gr.Row():
                caller_id = gr.Textbox(
                    label="Caller ID",
                    placeholder="Optional — e.g. customer email or account number",
                )
                department = gr.Textbox(
                    label="Department",
                    placeholder="Optional — e.g. Billing, Technical Support",
                )

        analyze_btn = gr.Button("🔍  Analyze Call", variant="primary", size="lg")
        status_html = gr.HTML(visible=False)

        with gr.Column(visible=False) as results_col:
            with gr.Tabs():
                with gr.Tab("📝 Transcript"):
                    transcript_box = gr.Textbox(
                        show_label=False,
                        lines=16,
                        show_copy_button=True,
                        interactive=False,
                    )
                with gr.Tab("🗒️ Summary"):
                    summary_md = gr.Markdown()
                with gr.Tab("✅ QA Scorecard"):
                    qa_md = gr.Markdown()

            gr.Markdown('<div class="cci-section-title">⬇️ Reports</div>')
            with gr.Row():
                pdf_file = gr.File(label="PDF Report")
                json_file = gr.File(label="JSON Report")

        def _show_status():
            return (
                gr.update(
                    value=status_banner(
                        "processing",
                        "<strong>Processing your call…</strong> This can take up to a "
                        "minute depending on call length. Please don't refresh the page.",
                        icon="⏳",
                    ),
                    visible=True,
                ),
                gr.update(visible=False),
            )

        def _run(audio, cid, dept):
            result = process_call(workflow, audio, cid, dept, langfuse_handler=langfuse_handler)
            if result.status != "completed":
                message = result.error or _STATUS_LABELS.get(result.status, "Processing failed.")
                icon = _STATUS_ICONS.get(result.status, "⚠️")
                banner = status_banner(result.status, message, icon=icon)
                return (
                    gr.update(value=banner, visible=True),
                    gr.update(visible=False),
                    "",
                    "",
                    "",
                    None,
                    None,
                )
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                result.transcript_text,
                result.summary_markdown,
                result.qa_markdown,
                result.pdf_path,
                result.json_path,
            )

        analyze_btn.click(_show_status, outputs=[status_html, results_col]).then(
            _run,
            inputs=[audio_input, caller_id, department],
            outputs=[
                status_html,
                results_col,
                transcript_box,
                summary_md,
                qa_md,
                pdf_file,
                json_file,
            ],
        )
