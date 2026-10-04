"""Display formatting helpers for timestamps, summaries, and QA scorecards.

Report output is HTML embedded in Markdown (Gradio renders both) rather than
plain bullet lists, so status/severity/score values carry color - this is the
single source of truth for that palette, shared with `src/ui/theme.py` so the
Gradio chrome and the report content it wraps stay visually consistent.
"""

from __future__ import annotations

from src.graph.state import QAScoreResult, SummaryResult

# Apple-style semantic palette: muted pastel background + a saturated
# foreground of the same hue, used for every colored badge/bar in a report.
PALETTE: dict[str, tuple[str, str]] = {
    "green": ("#1d7a46", "#e3f8ea"),
    "amber": ("#8a5300", "#fff2d9"),
    "red": ("#c62828", "#fde8e8"),
    "blue": ("#0059b3", "#e5f1ff"),
    "gray": ("#555555", "#eeeeee"),
}

_STATUS_STYLE: dict[str, tuple[str, str]] = {
    "completed": ("green", "Completed"),
    "flagged_for_review": ("amber", "Flagged for Review"),
    "failed": ("red", "Failed"),
    "error": ("red", "Error"),
    "pending": ("gray", "Pending"),
}

_RESOLUTION_STYLE: dict[str, str] = {
    "resolved": "green",
    "unresolved": "amber",
    "escalated": "red",
}

_SEVERITY_STYLE: dict[str, str] = {
    "low": "blue",
    "medium": "amber",
    "high": "amber",
    "critical": "red",
}


def secs_to_mmss(seconds: float) -> str:
    """Convert float seconds to a zero-padded MM:SS string."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _badge(text: str, hue: str) -> str:
    color, bg = PALETTE[hue]
    return (
        f'<span style="display:inline-block;padding:3px 12px;border-radius:999px;'
        f"background:{bg};color:{color};font-weight:600;font-size:0.82em;"
        f'letter-spacing:0.01em;white-space:nowrap;">{text}</span>'
    )


def format_status_badge(status: str) -> str:
    hue, label = _STATUS_STYLE.get(status, ("gray", status.replace("_", " ").title()))
    return _badge(label, hue)


def score_hue(score: float) -> str:
    if score < 3.0:
        return "red"
    if score < 4.0:
        return "amber"
    return "green"


def _score_bar(score: float, max_score: float = 5.0) -> str:
    pct = max(0.0, min(1.0, score / max_score)) * 100
    color, _bg = PALETTE[score_hue(score)]
    return (
        '<div style="background:#e5e5ea;border-radius:6px;height:7px;width:100%;'
        f'overflow:hidden;margin-top:5px;"><div style="background:{color};height:100%;'
        f'width:{pct:.0f}%;border-radius:6px;"></div></div>'
    )


def format_summary(summary: SummaryResult) -> str:
    resolution = summary.resolution_status.value
    resolution_hue = _RESOLUTION_STYLE.get(resolution, "gray")
    resolution_badge = _badge(resolution.replace("_", " ").title(), resolution_hue)

    lines = [
        f'<div style="font-size:1.05em;font-weight:600;margin-bottom:6px;">{summary.call_purpose}</div>',
        f'<div style="margin-bottom:14px;">{resolution_badge}</div>',
        "**Key discussion points**",
    ]
    if summary.key_discussion_points:
        lines += [f"- {p}" for p in summary.key_discussion_points]
    else:
        lines.append("- None recorded.")

    lines.append("")
    lines.append("**Action items**")
    if summary.action_items:
        for item in summary.action_items:
            deadline = f" (due {item.deadline})" if item.deadline else ""
            lines.append(f"- {item.description} — owner: {item.owner}{deadline}")
    else:
        lines.append("- None.")

    lines += [
        "",
        f"**Sentiment trajectory:** {summary.sentiment_trajectory}",
        "",
        "**Entities**",
    ]
    if summary.entities:
        lines += [f"- {e.name} ({e.type})" for e in summary.entities]
    else:
        lines.append("- None detected.")

    return "\n".join(lines)


def _dimension_block(label: str, dim) -> str:
    color, _bg = PALETTE[score_hue(dim.score)]
    return (
        '<div style="margin-bottom:14px;">'
        '<div style="display:flex;justify-content:space-between;font-size:0.92em;">'
        f'<span style="font-weight:600;">{label}</span>'
        f'<span style="font-weight:700;color:{color};">{dim.score}/5</span>'
        "</div>"
        f"{_score_bar(dim.score)}"
        f'<div style="color:#6e6e73;font-size:0.85em;margin-top:5px;">{dim.justification}</div>'
        "</div>"
    )


def format_qa(qa: QAScoreResult) -> str:
    overall_color, _bg = PALETTE[score_hue(qa.overall_score)]
    header = (
        '<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:18px;">'
        f'<span style="font-size:2.1em;font-weight:700;color:{overall_color};">{qa.overall_score:.2f}</span>'
        '<span style="color:#6e6e73;">/ 5.00 overall</span>'
        "</div>"
    )

    body = header + "".join(
        [
            _dimension_block("Professionalism", qa.professionalism),
            _dimension_block("Empathy", qa.empathy),
            _dimension_block("Problem Resolution", qa.problem_resolution),
            _dimension_block("Compliance", qa.compliance),
            _dimension_block("Communication Clarity", qa.communication_clarity),
        ]
    )

    flags_html = ['<div style="font-weight:600;margin:6px 0 8px;">Compliance Flags</div>']
    if qa.compliance_flags:
        for flag in qa.compliance_flags:
            hue = _SEVERITY_STYLE.get(flag.severity, "gray")
            ref = f" ({flag.timestamp_reference})" if flag.timestamp_reference else ""
            badge = _badge(flag.severity.upper(), hue)
            flags_html.append(
                f'<div style="margin-bottom:6px;">{badge} '
                f'<span style="font-size:0.92em;">{flag.description}{ref}</span></div>'
            )
    else:
        flags_html.append(_badge("No compliance issues detected.", "green"))

    return body + "".join(flags_html)
