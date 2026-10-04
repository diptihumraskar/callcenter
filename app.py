"""Entrypoint. Wires config, database, the Whisper singleton, the compiled
LangGraph workflow, and the Gradio app, then launches on http://localhost:7860.
"""

import logging
import os

from src.agents.transcription import _get_whisper_model
from src.database.connection import get_engine, init_db
from src.graph.workflow import WorkflowDeps, compile_workflow
from src.security.audit import AuditLogger
from src.services.pipeline import set_max_temp_files
from src.ui.app import build_app
from src.utils.config import load_config
from src.utils.observability_providers import get_langfuse_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call_center_intelligence_system")


def main() -> None:
    config = load_config()

    engine = get_engine(config.db_path, config.db_encryption_key)
    init_db(engine)
    set_max_temp_files(config.max_file_retention)

    logger.info("Loading faster-whisper model (%s)...", config.whisper_model_size)
    _get_whisper_model(config.whisper_model_size)

    audit_logger = AuditLogger(engine)
    workflow = compile_workflow(
        WorkflowDeps(
            engine=engine,
            llm_provider=config.llm_provider,
            whisper_model_size=config.whisper_model_size,
            audit_logger=audit_logger,
            max_retries=config.max_retries_per_node,
        )
    )

    langfuse_handler = get_langfuse_handler(config)
    if config.langfuse_enabled and langfuse_handler is None:
        logger.warning(
            "LANGFUSE_ENABLED is set but tracing did not start (missing keys, "
            "the langfuse package, or a connection problem) - continuing without it."
        )

    demo = build_app(
        workflow,
        engine,
        langsmith_enabled=config.langsmith_tracing,
        langsmith_project=config.langsmith_project,
        langfuse_enabled=langfuse_handler is not None,
        langfuse_host=config.langfuse_host,
        langfuse_handler=langfuse_handler,
    )

    # Hosted platforms (HF Spaces sets SPACE_ID, Render/Railway set PORT) need
    # the server reachable from outside the container, on the port they assign.
    hosted = bool(os.getenv("SPACE_ID") or os.getenv("PORT"))
    server_name = "0.0.0.0" if hosted else "127.0.0.1"
    server_port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()
