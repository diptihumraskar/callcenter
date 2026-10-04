---
title: Call Center Intelligence System
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Call Center Intelligence System

A production-grade, multi-agent AI pipeline that turns raw call center audio
into structured transcripts, quality scores, compliance reports, and
downloadable artifacts — built as a capstone project.

## Architecture

A LangGraph state machine coordinates eight sequential pipeline stages plus
three terminal outcomes (report, supervisor review, error):

```
intake_step ──► transcribe_step ──► injection_check_step ──► pii_redact_step
   │ (invalid)         │                    │ (injection)         │
   ▼                   ▼                    ▼                     ▼
error_step        (always continues)     error_step      speaker_labeling_step
                                              │                    │ (failed)
                                              │                    ▼
                                              │               error_step
                                              │                    │
                          speaker_labeling_step ──────► summarize_and_qa_step
                                                                │
                                    ┌───────────────────────────┼───────────────┐
                                    ▼                            ▼               ▼
                              report_step              supervisor_step     error_step
                                    │                            │               │
                                    └────────────────────────────┴───────►      END
```

| # | Stage | What it does |
|---|-------|---------------|
| 1 | Intake | Magic-byte format validation (not extension), 50 MB / 60-minute limits, metadata PII scan |
| 2 | Transcription | faster-whisper (int8, greedy decode, VAD), turn-boundary diarization, SHA-256 caching |
| 3 | Injection Detection | 22 regex patterns block adversarial audio before any LLM call |
| 4 | PII Redaction | Strips SSN, credit card, email, phone from full text and every segment |
| 5 | Speaker Labeling | One LLM call labels every segment Agent/Customer directly from full-conversation context |
| 6 | Summarization | Structured Pydantic output: purpose, action items, sentiment, entities |
| 7 | QA Scoring | 5 weighted dimensions; overall score always recomputed deterministically in Python |
| 8 | Report | PDF (ReportLab) + JSON reports, SQLite persistence, append-only audit log |

Transcription's turn-boundary heuristic (silence gaps, question/answer
alternation, a short reply after a long turn) is only a rough first pass -
in practice a single person's monologue often gets sliced into several
short segments by natural mid-sentence pauses, which can desync a
two-bucket heuristic partway through a call. So role assignment is not
trusted to that heuristic at all: stage 5 makes one LLM call with the whole
redacted transcript and labels every segment individually as Agent or
Customer, using full conversational context (who greets, who verifies
identity, who pitches a price, who raises an affordability concern). This
is far more reliable than a per-line trigger-phrase heuristic and self-
corrects mid-call instead of silently inheriting an earlier misdetection.

Five-layer source layout:

```
src/
  agents/     intake, transcription, speaker_labeling, summarization, qa_scoring, report
  graph/      state (14 Pydantic models + PipelineState), edges, workflow
  security/   injection_detector, pii_redactor, audit
  services/   pipeline (UI ↔ workflow bridge), observability (dashboard)
  ui/         Gradio Blocks app + tabs (Analyze Call, Call History, Observability),
              custom theme/CSS (src/ui/theme.py)
  database/   SQLAlchemy models + connection/session helpers
  utils/      config, audio, llm_factory, formatters, observability_providers
```

`app.py` is the ~40-line entrypoint that wires config → database → the
Whisper singleton → the compiled workflow → the Gradio app.

## Tech Stack

Python 3.11/3.12, LangGraph 0.4+, faster-whisper (CTranslate2, int8), Pydantic
v2, SQLite + SQLAlchemy, Gradio 5.x, ReportLab, mutagen, pytest, ruff. LLM
providers: OpenAI GPT-4o, Google Gemini 2.0 Flash, Groq Llama 3.3 70B —
switchable with a single `LLM_PROVIDER` environment variable, no code changes.

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.11 or 3.12 — not 3.13)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# then edit .env and set at least one of:
#   OPENAI_API_KEY   (LLM_PROVIDER=openai, default)
#   GOOGLE_API_KEY    (LLM_PROVIDER=gemini)
#   GROQ_API_KEY      (LLM_PROVIDER=groq)
```

On a CPU-only laptop, keep `WHISPER_MODEL_SIZE=tiny` (the `.env.example`
default) — `large-v3` takes 25+ minutes per call on CPU and is meant for GPU
deployments.

## Run

```bash
make run
# or: python app.py
```

Open **http://localhost:7860**. On the **🎧 Analyze Call** tab, upload a
recording, pick one of the built-in sample calls with the one-click example
picker, or record from your microphone. The first run downloads the
faster-whisper model weights automatically (not committed to the repo).

## Test

```bash
make test            # unit + security suites
make test-integration # end-to-end pipeline + database tests (mocked LLM/Whisper)
make test-all         # everything — 100+ tests, 0 failures expected
```

```
pytest tests/ -v
```

## Docker

```bash
docker build . -t callcenter
docker run -p 7860:7860 --env-file .env callcenter
```

## GPU Deployment

faster-whisper auto-detects CUDA via `torch.cuda.is_available()` and falls
back to CPU with int8 quantization (Apple Silicon/MPS also falls back to
CPU — faster-whisper has no MPS backend). For GPU deployments, set
`WHISPER_MODEL_SIZE=large-v3` for the highest transcription quality.

## Sample Usage

1. On the **🎧 Analyze Call** tab, click one of the sample calls under
   **"Or try a sample call"** (or upload your own) and click **🔍 Analyze
   Call**. A colored status banner shows progress and, if the call is
   flagged for supervisor review or fails outright, a clear explanation
   instead of a generic error.
2. Once processing finishes, the **Transcript**, **Summary**, and **QA
   Scorecard** sub-tabs fill in: a speaker-labeled transcript (`Agent` /
   `Customer`, with `[LOW CONF]` markers on low-confidence segments), a
   formatted summary (purpose, action items, sentiment), and a
   five-dimension QA scorecard with an overall score and any compliance
   flags.
3. Download the PDF or JSON report.
4. Switch to **📁 Call History** to search and browse every call ever
   analyzed (filter by Call ID or filename, then inspect one in detail), or
   **📊 Observability** to see pipeline health as stat cards: total calls,
   success rate, average QA score, flags, failures, compliance flags, audit
   events, the 20 most recent audit log events, and LangSmith/Langfuse
   tracing status (both tabs auto-refresh when you open them).

## Observability / Tracing (optional)

Two LLM tracing providers can be enabled independently, both off by default:

- **LangSmith** — set `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and
  `LANGCHAIN_PROJECT` in `.env`. No extra install needed (`langsmith` ships
  as a core dependency); every `@traceable`-decorated pipeline node
  (`src/graph/workflow.py`) and each LLM call inside it is reported
  automatically.
- **Langfuse** — set `LANGFUSE_ENABLED=true`, `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`, and optionally `LANGFUSE_HOST` (defaults to
  Langfuse Cloud) in `.env`, then install the extra dependencies:
  ```bash
  pip install -e ".[observability]"
  # or: pip install "langfuse>=3.0" langchain
  ```
  Langfuse's LangChain integration needs the full `langchain` metapackage,
  not just `langchain-core` — that's why it's a separate optional install
  rather than a core dependency.

Both can be on at the same time; the Observability tab shows each one's
status and a link to its dashboard. Neither ever blocks call processing —
a missing package, missing keys, or a provider outage just means tracing is
silently off for that provider, not a failed call.

## Security Notes

- PII (SSN, credit card, email, phone) is redacted from the full transcript
  and every individual segment **before** any LLM call.
- A 22-pattern prompt injection scan runs on the raw transcript before
  summarization or QA scoring; a match routes straight to the error path and
  the LLM is never invoked.
- The audit log is append-only — no update or delete path exists.
- The QA `overall_score` returned by the LLM is always discarded and
  recomputed deterministically from the weighted rubric
  (Professionalism 15%, Empathy 20%, Problem Resolution 30%, Compliance 20%,
  Communication Clarity 15%) as a guardrail against scoring drift.

## What's Excluded From Version Control

`.env`, `data/audio/` (runtime uploads), `data/calls.db` (created at
runtime), and faster-whisper's downloaded model weights — see `.gitignore`.
