import os

from src.utils.llm_factory import get_llm

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")


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
