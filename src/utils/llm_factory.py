"""Multi-provider LLM factory - switch providers via LLM_PROVIDER only."""

from __future__ import annotations

from typing import Any


def get_llm(provider: str, model: str | None = None, timeout: int = 60, **kwargs: Any):
    """Return a LangChain chat model for the requested provider.

    provider: "openai" | "gemini" | "groq". No code changes are ever needed
    to switch - only the LLM_PROVIDER environment variable.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        return ChatOpenAI(model=model or "gpt-4o", timeout=timeout, **kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash", timeout=timeout, **kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq  # noqa: PLC0415

        return ChatGroq(model=model or "llama-3.3-70b-versatile", timeout=timeout, **kwargs)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Expected openai, gemini, or groq.")
