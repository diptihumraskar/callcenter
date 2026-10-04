from unittest.mock import MagicMock

import numpy as np

from src.services.pipeline import _audio_to_wav_bytes, process_call
from src.utils.audio import detect_audio_format, make_wav_bytes


def test_audio_to_wav_bytes_from_numpy_tuple():
    sample_rate = 16000
    array = np.zeros(sample_rate, dtype=np.float32)
    wav_bytes = _audio_to_wav_bytes((sample_rate, array))
    assert detect_audio_format(wav_bytes) == "wav"


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
