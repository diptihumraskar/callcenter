"""Optional third-party tracing providers: LangSmith and Langfuse.

LangSmith needs no runtime wiring beyond environment variables
(``LANGCHAIN_TRACING_V2`` / ``LANGCHAIN_API_KEY`` / ``LANGCHAIN_PROJECT``)
plus the ``@traceable`` decorators already on every workflow node in
``src.graph.workflow`` - the LangChain/LangGraph SDKs pick those up
automatically and report each node (and every LLM call inside it) as a run.

Langfuse has no equivalent env-var auto-instrumentation for LangGraph, so it
is wired in explicitly as a LangChain callback handler passed into the
*top-level* ``workflow.invoke(..., config={"callbacks": [...]})`` call.
LangChain propagates callbacks down through every nested chain/LLM call
automatically, so passing it once at the top is enough to trace the whole
pipeline - individual nodes never need to know about it.

Both providers are strictly optional: a missing package, missing keys, or a
provider-side connection problem must never break call processing, only
disable tracing.
"""

from __future__ import annotations

from typing import Any

from src.utils.config import Config

_langfuse_handler: Any | None = None
_langfuse_resolved = False


def get_langfuse_handler(config: Config) -> Any | None:
    """Return a cached Langfuse CallbackHandler, or None if disabled/unavailable.

    Resolved once and cached for the process lifetime (mirrors the
    faster-whisper model singleton pattern) so repeated calls don't re-import
    or re-authenticate on every request.

    Modern Langfuse SDKs (v3+) authenticate their global client from the
    standard LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    environment variables rather than constructor arguments - src.utils.config
    already loads those into the environment via python-dotenv, so
    CallbackHandler() picks them up with no explicit wiring here. The
    "langfuse" package's LangChain integration additionally requires the
    full "langchain" metapackage (not just langchain-core) to be installed -
    see pyproject.toml's "observability" extra.
    """
    global _langfuse_handler, _langfuse_resolved
    if _langfuse_resolved:
        return _langfuse_handler

    _langfuse_resolved = True
    if not config.langfuse_enabled:
        return None
    if not (config.langfuse_public_key and config.langfuse_secret_key):
        return None

    try:
        from langfuse.langchain import CallbackHandler  # noqa: PLC0415
    except ImportError:
        return None

    try:
        _langfuse_handler = CallbackHandler()
    except Exception:  # noqa: BLE001 - tracing setup must never break the pipeline
        _langfuse_handler = None

    return _langfuse_handler


def reset_langfuse_handler_cache() -> None:
    """Test-only hook to force re-resolution of the cached handler."""
    global _langfuse_handler, _langfuse_resolved
    _langfuse_handler = None
    _langfuse_resolved = False
