import sys
import types

import pytest

from src.utils import observability_providers as providers
from src.utils.config import load_config


@pytest.fixture(autouse=True)
def _reset_cache():
    providers.reset_langfuse_handler_cache()
    yield
    providers.reset_langfuse_handler_cache()


def _config(monkeypatch, **overrides):
    monkeypatch.setenv("LANGFUSE_ENABLED", str(overrides.get("langfuse_enabled", False)).lower())
    if "langfuse_public_key" in overrides:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", overrides["langfuse_public_key"])
    else:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    if "langfuse_secret_key" in overrides:
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", overrides["langfuse_secret_key"])
    else:
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    return load_config()


def test_returns_none_when_disabled(monkeypatch):
    config = _config(monkeypatch, langfuse_enabled=False)
    assert providers.get_langfuse_handler(config) is None


def test_returns_none_when_enabled_but_missing_keys(monkeypatch):
    config = _config(monkeypatch, langfuse_enabled=True)
    assert providers.get_langfuse_handler(config) is None


def test_returns_none_when_package_not_installed(monkeypatch):
    config = _config(
        monkeypatch, langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk"
    )
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)
    monkeypatch.delitem(sys.modules, "langfuse.langchain", raising=False)

    # Simulate the package genuinely being absent by making the import fail.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("no langfuse installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    assert providers.get_langfuse_handler(config) is None


def test_builds_handler_when_package_available(monkeypatch):
    config = _config(
        monkeypatch, langfuse_enabled=True, langfuse_public_key="pk", langfuse_secret_key="sk"
    )

    created = {}

    class FakeCallbackHandler:
        # Modern Langfuse SDKs (v3+) authenticate from LANGFUSE_* env vars,
        # not constructor arguments - the handler takes no credentials.
        def __init__(self):
            created["constructed"] = True

    fake_module = types.ModuleType("langfuse.langchain")
    fake_module.CallbackHandler = FakeCallbackHandler
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_module)
    monkeypatch.setitem(sys.modules, "langfuse", types.ModuleType("langfuse"))

    handler = providers.get_langfuse_handler(config)

    assert isinstance(handler, FakeCallbackHandler)
    assert created == {"constructed": True}


def test_result_is_cached_across_calls(monkeypatch):
    config = _config(monkeypatch, langfuse_enabled=False)

    first = providers.get_langfuse_handler(config)
    # Even if config would now resolve differently, the cached value wins.
    second_config = _config(monkeypatch, langfuse_enabled=True)
    second = providers.get_langfuse_handler(second_config)

    assert first is None
    assert second is None
