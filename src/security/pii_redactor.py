"""PII redaction applied to transcript text before any LLM sees it.

Uses Microsoft Presidio (spaCy NER + pattern recognizers) instead of
hand-written regex, so redaction covers structured identifiers (SSN, credit
card, email, phone) as well as named entities (person names, locations,
dates). The analyzer/anonymizer pair is loaded once as a module-level
singleton, since spinning up the spaCy pipeline per call is expensive.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import UsSsnRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Presidio's built-in UsSsnRecognizer blocklists well-known placeholder SSNs
# (e.g. 123-45-6789) as "canonical sample" numbers from its own docs. A call
# center transcript is real customer data, never a doc example, so that
# blocklist is dropped here - every other structural check (mismatched
# delimiters, all-zero groups, unissued area numbers) is kept.
class _LenientUsSsnRecognizer(UsSsnRecognizer):
    def invalidate_result(self, pattern_text: str) -> bool:
        delimiter_counts: dict[str, int] = defaultdict(int)
        for char in pattern_text:
            if char in (".", "-", " "):
                delimiter_counts[char] += 1
        if len(delimiter_counts) > 1:
            return True

        only_digits = "".join(c for c in pattern_text if c.isdigit())
        if all(only_digits[0] == c for c in only_digits):
            return True
        if only_digits[3:5] == "00" or only_digits[5:] == "0000":
            return True
        return only_digits[:3] in ("000", "666")


# Entity types considered, mapped to the short label used in [REDACTED_*]
# tags. Order here also sets priority for overlap resolution below -
# structured pattern matches (SSN/card/email/phone) always outrank generic
# NER guesses (e.g. spaCy's date model treating a bare 10-digit phone number
# as a DATE_TIME).
_ENTITY_LABELS: dict[str, str] = {
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE_TIME",
}
_ENTITY_PRIORITY = {entity: rank for rank, entity in enumerate(_ENTITY_LABELS)}

# Filters out Presidio's "very weak" (0.05-0.3) unconfirmed guesses - e.g. a
# bare 9-digit account number matching the loosest SSN pattern - while
# keeping every format this project's test suite exercises (phone at 0.4,
# SSN/card/email/NER entities all score >= 0.5).
_SCORE_THRESHOLD = 0.4

_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Build the analyzer/anonymizer once, lazily, as module singletons."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        ).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        analyzer.registry.remove_recognizer("UsSsnRecognizer")
        analyzer.registry.add_recognizer(_LenientUsSsnRecognizer())
        _analyzer = analyzer
        _anonymizer = AnonymizerEngine()
    return _analyzer, _anonymizer


def _resolve_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the highest-priority entity when two matches cover the same span."""
    ordered = sorted(
        results, key=lambda r: (_ENTITY_PRIORITY[r.entity_type], r.start, -(r.end - r.start))
    )
    kept: list[RecognizerResult] = []
    for result in ordered:
        if not any(result.start < k.end and k.start < result.end for k in kept):
            kept.append(result)
    return kept


@dataclass
class RedactionResult:
    redacted_text: str
    pii_found: bool = False
    matched_types: list[str] = field(default_factory=list)


def redact_pii(text: str) -> RedactionResult:
    """Detect and replace SSN, credit card, email, phone, person, location,
    and date/time entities using Presidio's NER + pattern recognizers."""
    if not text:
        return RedactionResult(redacted_text=text, pii_found=False, matched_types=[])

    analyzer, anonymizer = _get_engines()
    results = analyzer.analyze(
        text=text,
        entities=list(_ENTITY_LABELS),
        language="en",
        score_threshold=_SCORE_THRESHOLD,
    )
    if not results:
        return RedactionResult(redacted_text=text, pii_found=False, matched_types=[])

    resolved = _resolve_overlaps(results)
    operators = {
        entity: OperatorConfig("replace", {"new_value": f"[REDACTED_{label}]"})
        for entity, label in _ENTITY_LABELS.items()
    }
    anonymized = anonymizer.anonymize(text=text, analyzer_results=resolved, operators=operators)

    matched_types = sorted({_ENTITY_LABELS[r.entity_type] for r in resolved})
    return RedactionResult(
        redacted_text=anonymized.text, pii_found=True, matched_types=matched_types
    )
