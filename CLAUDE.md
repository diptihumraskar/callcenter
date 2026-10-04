# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable + dev extras: pytest, pytest-cov, ruff, pre-commit)
pip install -e ".[dev]"          # or: pip install -r requirements-dev.txt

# Run the app -> http://localhost:7860
make run                          # or: python app.py

# Lint / format
make lint                         # ruff check .
make format                       # ruff check . --fix && ruff format .

# Tests
make test                         # tests/unit + tests/security (fast, mocked LLM/Whisper)
make test-integration             # tests/integration (end-to-end pipeline + DB)
make test-all                     # everything

# Single test
pytest tests/security/test_pii_detection.py -v
pytest tests/unit/test_qa_scoring.py -k test_name -v

make clean                        # clears __pycache__, .pytest_cache, .ruff_cache, data/calls.db
```

Python 3.11 or 3.12 only (not 3.13). See `README.md` for `.env` setup, Docker, GPU deployment, and sample usage.

## Architecture

`app.py` wires: config → DB engine/init → Whisper singleton preload → compiled LangGraph workflow → Gradio app, then launches.

### Pipeline (`src/graph/workflow.py`)

A `StateGraph` over a shared `PipelineState` TypedDict (`src/graph/state.py`) — nodes return partial dicts that accumulate into state (no custom reducers, plain last-write-wins per key). Seven stages, three terminal nodes, all leading to `END`:

```
intake_step -> transcribe_step -> injection_check_step -> pii_redact_step -> summarize_and_qa_step -> {report_step | supervisor_step | error_step}
```

- **Routing** lives in `src/graph/edges.py`, called from `add_conditional_edges` lambdas in `workflow.py`. `route_after_intake` sends invalid audio to `error_step`. `route_after_qa` sends any `critical`-severity compliance flag to `supervisor_step`, otherwise `report_step`.
- **Order is security-load-bearing, don't rearrange it**: prompt-injection scanning (`injection_check_step`, raw transcript) always runs *before* PII redaction, which always runs *before* the transcript reaches an LLM in `summarize_and_qa_step`. A hostile or PII-laden transcript must never reach an LLM call.
- Every terminal node (`report_step`, `error_step`, `supervisor_step`) sets both `status` and `error`/`report` before reaching `END` — if you add a new terminal path, it must do the same, since `src/services/pipeline.py` trusts `result_state["status"]` to decide which fields are populated.
- `WorkflowDeps` (dataclass in `workflow.py`) is the single place runtime dependencies (DB engine, LLM provider, Whisper model size, audit logger, max retries) get threaded into node closures.

### Security stack (`src/security/`)

- `injection_detector.py` — ~22 regex patterns (jailbreak phrases, `[INST]` tags, role-switch attempts) scanned against the *raw* transcript before anything else touches it.
- `pii_redactor.py` — PII/NER redaction via **Microsoft Presidio** (spaCy `en_core_web_sm` + pattern recognizers), not regex. Two non-obvious pieces:
  - `_LenientUsSsnRecognizer` overrides Presidio's stock `UsSsnRecognizer` to drop its "canonical example SSN" denylist (e.g. `123-45-6789`) — real transcripts aren't doc examples, so that blocklist is wrong here. All other SSN validity checks (mismatched delimiters, all-zero groups, unissued area codes) are kept.
  - `_resolve_overlaps` enforces a fixed entity-type priority (structured patterns — SSN/credit card/email/phone — always outrank generic NER guesses like a bare phone number getting mis-tagged `DATE_TIME`) before anonymizing, since Presidio's anonymizer expects non-overlapping spans. The `AnalyzerEngine`/`AnonymizerEngine` pair is a lazy module-level singleton (`_get_engines`) — building it per-call would reload the spaCy pipeline every time.
  - Redacts 7 entity types: SSN, credit card, email, phone, person names, locations, dates — broader than pattern-only PII, so it will also redact things QA scoring might otherwise want (e.g. an agent's name in a greeting).
- `audit.py` — append-only; there is no update or delete path by design.

### LLM layer

- `src/utils/llm_factory.py::get_llm(provider)` is the only place that branches on provider (`openai` | `gemini` | `groq`) — switching providers is purely a `LLM_PROVIDER` env var change, never a code change. Don't add provider-specific logic anywhere else.
- `summarization.py` / `qa_scoring.py` both use LangChain's `with_structured_output(PydanticModel)` against `SummaryResult`/`QAScoreResult`, with exponential-backoff retry (`max_retries`, default 3) collapsing all transient failures into a single typed exception (`SummarizationError`/`QAScoringError`) that the graph routes on.
- `qa_scoring.py::_recompute_overall_score` always **discards** the LLM's own `overall_score` and recomputes it deterministically from `DIMENSION_WEIGHTS` — a guardrail against LLM arithmetic drift. Don't let an LLM-proposed overall score survive into a report.

### Transcription (`src/agents/transcription.py`)

- `faster-whisper` model is a process-wide singleton (`_get_whisper_model`), loaded once at startup in `app.py` and reused for every call. Device auto-detects CUDA, otherwise falls back to CPU/int8 (Apple Silicon/MPS also falls back to CPU — faster-whisper has no MPS backend).
- Results are cached by SHA-256 of the audio file (`transcription_cache` table) — a repeat upload of the same file skips Whisper entirely and returns with `from_cache=True`.
- Speaker labeling (`SpeakerDiarizer`) is a stateful heuristic (content-pattern match, then question→answer turn-taking, then silence-gap, then short-affirmation flip), not a real diarization model. `label()` takes `(start, end, text)` — the silence-gap rule measures `next_start - previous_end`, so passing only a start (as an earlier version of this code did) silently breaks gap detection by conflating "silence since the last turn" with "the last segment's own duration." Any new call site must pass both timestamps.

### Config gotcha (`src/utils/config.py`)

Four `Config` fields are parsed from `.env` but **not currently wired into any behavior** — `confidence_threshold`, `low_confidence_halt_ratio`, `llm_timeout_seconds`, and `max_file_retention` are read into the frozen `Config` dataclass but never consumed elsewhere in `src/` or `app.py`. The actual low-confidence marker in `transcription.py` hardcodes `0.55`, the temp-file cap in `pipeline.py` hardcodes `50`, and `llm_factory.get_llm` uses its own default timeout. Don't assume changing these env vars has any effect until they're actually threaded through — wire them into the relevant call site if you need them to work.

### Persistence (`src/database/`)

Three SQLAlchemy tables (`models.py`): `call_records`, `audit_log`, `transcription_cache`. All session lifecycles go through `connection.py::session_scope` (commit on success, rollback on exception, always closes) — use it rather than opening a raw `Session`. Optional SQLCipher-style encryption via `PRAGMA key` if `DB_ENCRYPTION_KEY` is set.

### Git state gotcha

`git rev-parse --show-toplevel` resolves to the user's home directory, not this project folder, and `master` has zero commits. `git log`/`git diff`/`git blame` will not give you real project history here — don't rely on them to determine when a change was introduced.
