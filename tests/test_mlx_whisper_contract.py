"""mlx-whisper internals Sotto relies on.

LocalWhisper (sotto.py) reuses one encoder pass for language detection and
transcription, which needs three private pieces of mlx-whisper. The exact pin
in constraints-alpha.txt keeps them; this test fails loudly if an upgrade
removes one, instead of dictation breaking at runtime. It checks in a child
interpreter so this test process never imports mlx-whisper (other tests assert
that some paths do not).
"""
import importlib.util
import subprocess
import sys
import unittest

HAVE_MLX_WHISPER = importlib.util.find_spec("mlx_whisper") is not None

CHECK = """
import inspect
from mlx_whisper.decoding import DecodingTask
from mlx_whisper.transcribe import ModelHolder
from mlx_whisper.whisper import AudioEncoder
assert callable(getattr(ModelHolder, "get_model", None)), "ModelHolder.get_model"
assert callable(getattr(DecodingTask, "_get_audio_features", None)), "DecodingTask._get_audio_features"
assert "_positional_embedding" in inspect.getsource(AudioEncoder.__init__), "AudioEncoder._positional_embedding"
print("contract holds")
"""


@unittest.skipUnless(HAVE_MLX_WHISPER, "mlx-whisper is not installed (macOS alpha environment)")
class MlxWhisperContractTests(unittest.TestCase):
    def test_the_private_pieces_sotto_uses_still_exist(self):
        result = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-600:])
        self.assertIn("contract holds", result.stdout)


if __name__ == "__main__":
    unittest.main()
