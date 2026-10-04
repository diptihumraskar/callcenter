from unittest.mock import MagicMock

from src.database.connection import get_engine, init_db
from src.ui.app import build_app


def test_build_app_does_not_raise():
    engine = get_engine(":memory:")
    init_db(engine)
    workflow = MagicMock()

    demo = build_app(workflow, engine, langsmith_enabled=False, langsmith_project=None)

    assert demo is not None


def test_build_app_with_langfuse_enabled_does_not_raise():
    engine = get_engine(":memory:")
    init_db(engine)
    workflow = MagicMock()

    demo = build_app(
        workflow,
        engine,
        langfuse_enabled=True,
        langfuse_host="https://cloud.langfuse.com",
        langfuse_handler=MagicMock(),
    )

    assert demo is not None
