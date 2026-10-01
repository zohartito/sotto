import tempfile
import unittest
import wave
from pathlib import Path

from audio_codec import write_pcm16_wav


class WritePcm16WavTests(unittest.TestCase):
    def test_writes_durable_wav_on_every_platform(self):
        # Regression: fsync on a read-only handle raised EBADF on Windows,
        # which failed every live dictation before it could paste.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "audio" / "clip.wav"
            pcm = b"\x01\x00" * 1600
            write_pcm16_wav(target, pcm, 16000)
            with wave.open(str(target), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.readframes(wav.getnframes()), pcm)
            self.assertEqual(list(target.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
