from src.utils.config import load_config


def test_load_config_reads_without_crashing():
    config = load_config()
    assert config.llm_provider in {"openai", "gemini", "groq"}
    assert config.whisper_model_size


def test_load_config_respects_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "base")
    monkeypatch.setenv("MAX_RETRIES_PER_NODE", "5")
    config = load_config()
    assert config.llm_provider == "groq"
    assert config.whisper_model_size == "base"
    assert config.max_retries_per_node == 5


def test_load_config_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)
    config = load_config()
    assert 0.0 <= config.confidence_threshold <= 1.0


def test_load_config_db_path_default():
    config = load_config()
    assert config.db_path
