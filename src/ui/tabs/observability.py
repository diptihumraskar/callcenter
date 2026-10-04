"""Observability tab: pipeline health dashboard, refreshes on tab select."""

from __future__ import annotations

import gradio as gr
from sqlalchemy import Engine

from src.services.observability import get_observability_dashboard


def build_observability_tab(
    engine: Engine,
    langsmith_enabled: bool,
    langsmith_project: str | None,
    langfuse_enabled: bool = False,
    langfuse_host: str | None = None,
) -> None:
    with gr.Tab("📊 Observability") as tab:
        gr.Markdown('<div class="cci-section-title">📊 Pipeline health</div>')
        stats_html = gr.HTML()

        gr.Markdown('<div class="cci-section-title">🔗 Tracing</div>')
        tracing_html = gr.HTML()

        gr.Markdown('<div class="cci-section-title">🧾 Recent audit log events</div>')
        audit_table = gr.Dataframe(
            headers=["Timestamp", "Call ID", "Action", "Details"], interactive=False
        )
        refresh_btn = gr.Button("🔄 Refresh")

        def _load():
            return get_observability_dashboard(
                engine, langsmith_enabled, langsmith_project, langfuse_enabled, langfuse_host
            )

        tab.select(_load, outputs=[stats_html, tracing_html, audit_table])
        refresh_btn.click(_load, outputs=[stats_html, tracing_html, audit_table])
