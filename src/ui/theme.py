"""Shared visual identity: color theme + CSS used across every tab.

Centralized here so every tab's status banners, cards, and badges look like
one coherent product instead of default-Gradio grey boxes.
"""

from __future__ import annotations

import gradio as gr

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
    block_label_text_weight="500",
)

# Status colors reused by every HTML banner/badge across tabs, so "flagged"
# always means the same amber everywhere, "failed" the same red, etc.
STATUS_COLORS = {
    "processing": {"bg": "#eef2ff", "border": "#6366f1", "text": "#3730a3"},
    "success": {"bg": "#ecfdf5", "border": "#10b981", "text": "#065f46"},
    "flagged_for_review": {"bg": "#fffbeb", "border": "#f59e0b", "text": "#92400e"},
    "failed": {"bg": "#fef2f2", "border": "#ef4444", "text": "#991b1b"},
    "completed": {"bg": "#ecfdf5", "border": "#10b981", "text": "#065f46"},
    "neutral": {"bg": "#f8fafc", "border": "#94a3b8", "text": "#334155"},
}

CUSTOM_CSS = """
.cci-header {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    color: white !important;
}
.cci-header h1 {
    color: white !important;
    margin: 0 0 4px 0;
    font-size: 1.7rem;
    font-weight: 700;
}
.cci-header p {
    color: rgba(255,255,255,0.88) !important;
    margin: 0;
    font-size: 0.95rem;
}
.cci-badge-row { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
.cci-pill {
    display: inline-block;
    background: rgba(255,255,255,0.16);
    color: white;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
}
.cci-section-title {
    font-weight: 600;
    font-size: 1.02rem;
    margin: 4px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.cci-banner {
    border-left: 4px solid;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.95rem;
    margin-bottom: 8px;
}
.cci-stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }
.cci-stat-card {
    flex: 1 1 160px;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    min-width: 150px;
}
.cci-stat-value { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
.cci-stat-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.cci-tracing-row { display: flex; gap: 14px; flex-wrap: wrap; }
.cci-tracing-card {
    flex: 1 1 220px;
    border-radius: 12px;
    padding: 14px 18px;
    border: 1px solid #e2e8f0;
}
.cci-tracing-card a { color: inherit; font-weight: 600; text-decoration: underline; }
.cci-status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
"""


def status_banner(status: str, message: str, *, icon: str = "") -> str:
    """A colored HTML alert box - the same visual language used everywhere
    a call's status is surfaced (Analyze tab, History detail view)."""
    colors = STATUS_COLORS.get(status, STATUS_COLORS["neutral"])
    prefix = f"{icon} " if icon else ""
    return (
        f'<div class="cci-banner" style="background:{colors["bg"]};'
        f'border-color:{colors["border"]};color:{colors["text"]}">'
        f"{prefix}{message}</div>"
    )
