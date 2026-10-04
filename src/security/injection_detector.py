"""Prompt injection detection for raw call transcripts.

Runs on the full transcript text before it is ever shown to an LLM. Any
match routes the pipeline straight to the error node so a hostile audio
file can never influence the summarization or QA-scoring prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (compiled_pattern, name) tuples. At least 22 patterns covering the common
# injection tactics adversarial audio might carry.
INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I), "ignore_previous"),
    (re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.I), "ignore_prior"),
    (re.compile(r"disregard\s+(all\s+)?(the\s+)?prior\s+instructions", re.I), "disregard_prior"),
    (re.compile(r"forget\s+(all\s+)?(the\s+)?previous\s+instructions", re.I), "forget_previous"),
    (re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I), "prompt_leak"),
    (
        re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?instructions", re.I),
        "prompt_leak_question",
    ),
    (re.compile(r"<\|?system\|?>", re.I), "system_prompt_inject"),
    (re.compile(r"<<sys>>", re.I), "llama_system_tag"),
    (re.compile(r"\[inst\]", re.I), "llama_inst_tag"),
    (re.compile(r"\[/inst\]", re.I), "llama_inst_close_tag"),
    (re.compile(r"you\s+are\s+now\s+(a\s|an\s|acting\s+as|[A-Za-z]+\b)", re.I), "role_switch"),
    (re.compile(r"new\s+instructions\s*:", re.I), "new_instructions"),
    (re.compile(r"\bdan\s+mode\b", re.I), "dan_mode"),
    (re.compile(r"\bjailbreak\b", re.I), "jailbreak"),
    (re.compile(r"override\s+(your\s+)?safety", re.I), "override_safety"),
    (re.compile(r"ignore\s+(the\s+)?transcript", re.I), "ignore_transcript"),
    (re.compile(r"\bassistant\s*:\s*", re.I), "conversation_inject"),
    (re.compile(r"pretend\s+(to\s+be|you\s+are)", re.I), "social_engineering"),
    (re.compile(r"translate\s+the\s+above\s+and\s+then\s+execute", re.I), "translate_attack"),
    (re.compile(r"ignore\s+(all\s+)?safety\s+(rules|guidelines)", re.I), "ignore_safety"),
    (re.compile(r"system\s*:\s*override", re.I), "system_override"),
    (re.compile(r"reveal\s+(your\s+)?(training|configuration|api\s+key)", re.I), "reveal_attack"),
]


@dataclass
class InjectionScanResult:
    injection_detected: bool
    matched_patterns: list[str] = field(default_factory=list)


def detect_injection(text: str) -> InjectionScanResult:
    """Scan text against every known injection pattern; return all matches."""
    matched = [name for pattern, name in INJECTION_PATTERNS if pattern.search(text)]
    return InjectionScanResult(injection_detected=bool(matched), matched_patterns=matched)
