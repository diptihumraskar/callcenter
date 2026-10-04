import pytest

from src.utils.llm_factory import get_llm

# These tests must pass with no real API keys. A .env created from
# .env.example (the documented `cp .env.example .env` setup step) loads
# blank-string values for OPENAI_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY via
# src.utils.config's load_dotenv() at import time, which makes
# os.environ.setdefault(...) a no-op (the keys already "exist", just empty).
# monkeypatch.setenv forces the value regardless of what's already in the
# environment, so these tests are immune to whatever .env happens to exist.


@pytest.fixture(autouse=True)
def _fake_provider_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")


def test_get_llm_openai():
    from langchain_openai import ChatOpenAI

    llm = get_llm("openai")
    assert isinstance(llm, ChatOpenAI)


def test_get_llm_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = get_llm("gemini")
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_get_llm_groq():
    from langchain_groq import ChatGroq

    llm = get_llm("groq")
    assert isinstance(llm, ChatGroq)
