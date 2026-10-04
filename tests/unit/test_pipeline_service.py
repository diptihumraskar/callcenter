from unittest.mock import MagicMock

import numpy as np

from src.services import pipeline
from src.services.pipeline import _audio_to_wav_bytes, process_call, set_max_temp_files
from src.utils.audio import detect_audio_format, make_wav_bytes


def test_audio_to_wav_bytes_from_numpy_tuple():
    sample_rate = 16000
    array = np.zeros(sample_rate, dtype=np.float32)
    wav_bytes = _audio_to_wav_bytes((sample_rate, array))
    assert detect_audio_format(wav_bytes) == "wav"


def test_set_max_temp_files_updates_module_level_cap():
    original = pipeline._MAX_TEMP_FILES
    try:
        set_max_temp_files(5)
        assert pipeline._MAX_TEMP_FILES == 5
    finally:
        set_max_temp_files(original)


def test_track_temp_file_evicts_oldest_past_configured_cap(tmp_path):
    original = pipeline._MAX_TEMP_FILES
    original_list = list(pipeline._temp_files)
    try:
        set_max_temp_files(2)
        pipeline._temp_files.clear()
        paths = [tmp_path / f"f{i}.tmp" for i in range(3)]
        for p in paths:
            p.write_bytes(b"x")
            pipeline._track_temp_file(str(p))

        # Oldest file should have been evicted and removed from disk; the cap
        # keeps exactly the configured number of most-recent files.
        assert not paths[0].exists()
        assert paths[1].exists() and paths[2].exists()
        assert len(pipeline._temp_files) == 2
    finally:
        set_max_temp_files(original)
        pipeline._temp_files.clear()
        pipeline._temp_files.extend(original_list)


def _workflow_returning_failed():
    workflow = MagicMock()
    workflow.invoke.return_value = {"status": "failed", "error": "boom"}
    return workflow


def test_process_call_invokes_without_callbacks_by_default(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(make_wav_bytes(1.0))
    workflow = _workflow_returning_failed()

    process_call(workflow, str(audio_path), None, None)

    _, kwargs = workflow.invoke.call_args
    assert kwargs["config"] is None


def test_process_call_passes_langfuse_handler_as_callback(tmp_path):
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(make_wav_bytes(1.0))
    workflow = _workflow_returning_failed()
    fake_handler = object()

    process_call(workflow, str(audio_path), None, None, langfuse_handler=fake_handler)

    _, kwargs = workflow.invoke.call_args
    assert kwargs["config"] == {"callbacks": [fake_handler]}
