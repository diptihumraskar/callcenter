"""Builds the top-level three-tab Gradio Blocks application."""

from __future__ import annotations

import gradio as gr
from sqlalchemy import Engine

from src.ui.tabs.analyze import build_analyze_tab
from src.ui.tabs.history import build_history_tab
from src.ui.tabs.observability import build_observability_tab
from src.ui.theme import CUSTOM_CSS, THEME

_HEADER_HTML = """
<div class="cci-header">
  <h1>&#127908; Call Center Intelligence System</h1>
  <p>Transcription, compliance screening, and QA scoring for every recorded call.</p>
  <div class="cci-badge-row">
    <span class="cci-pill">Speech-to-text</span>
    <span class="cci-pill">PII redaction</span>
    <span class="cci-pill">Prompt-injection defense</span>
    <span class="cci-pill">Automated QA scoring</span>
  </div>
</div>
"""


def build_app(
    workflow,
    engine: Engine,
    langsmith_enabled: bool = False,
    langsmith_project: str | None = None,
    langfuse_enabled: bool = False,
    langfuse_host: str | None = None,
    langfuse_handler=None,
) -> gr.Blocks:
    with gr.Blocks(
        title="Call Center Intelligence System",
        theme=THEME,
        css=CUSTOM_CSS,
        fill_width=True,
    ) as demo:
        gr.HTML(_HEADER_HTML)
        with gr.Tabs():
            build_analyze_tab(workflow, langfuse_handler)
            build_history_tab(engine)
            build_observability_tab(
                engine, langsmith_enabled, langsmith_project, langfuse_enabled, langfuse_host
            )
    return demo
