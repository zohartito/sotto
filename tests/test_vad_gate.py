"""Output-side no-speech verdict (reads_as_no_speech).

The VAD is advisory: it never blocks transcription, because Silero scores
real whispered dictation at 0% speech frames. Instead every deliberate
capture reaches the ASR and the transcript is judged afterwards. Reference
data from this machine's capture history: silence and dead-mic captures
transcribe to short stock phrases ("you", "Thank you.", <=15 chars) at
VAD ~0%, while real whispered dictations produced 240+ chars at VAD 0%.
"""

import unittest
import unittest.mock

from sotto import reads_as_no_speech


class ReadsAsNoSpeechTest(unittest.TestCase):
    def test_empty_transcript_is_no_speech(self):
        self.assertEqual(reads_as_no_speech("", 0.0), "empty transcript")
        self.assertEqual(reads_as_no_speech("   ", 0.5), "empty transcript")

    def test_stock_hallucinations_on_speechless_audio_are_no_speech(self):
        # the exact outputs whisper produced on this machine's genuine
        # no-speech captures
        for text in ["you", "Thank you.", "!TTTT", "Girl, you don't"]:
            self.assertIsNotNone(reads_as_no_speech(text, 0.0), text)

    def test_long_transcript_wins_over_zero_vad(self):
        # whispered dictation: VAD 0% but a real transcript — must paste
        whispered = ("So because they're not recording all the screen, "
                     "they're just recording a lot of room and up.")
        self.assertIsNone(reads_as_no_speech(whispered, 0.0))

    def test_short_transcript_with_real_vad_score_pastes(self):
        # a normal short dictation ("Hello.") with a voiced VAD score
        self.assertIsNone(reads_as_no_speech("Hello.", 0.15))

    def test_boundary_is_calibrated_between_observed_classes(self):
        # worst observed silence output was 15 chars; shortest rescued
        # whisper was 240+ — the 20-char boundary sits in that margin
        self.assertIsNotNone(reads_as_no_speech("x" * 20, 0.0))
        self.assertIsNone(reads_as_no_speech("x" * 21, 0.0))




class SileroInstallTests(unittest.TestCase):
    """The advisory VAD model is pinned: nothing unverified is kept or loaded."""

    def test_install_keeps_only_the_pinned_bytes(self):
        import hashlib, tempfile
        from pathlib import Path
        import vad
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "models" / "silero_vad.onnx"
            with self.assertRaises(RuntimeError):
                vad.install(target, fetch=lambda: b"not the model")
            self.assertFalse(target.exists())
            good = b"pinned model bytes"
            with unittest.mock.patch.object(vad, "MODEL_SHA256", hashlib.sha256(good).hexdigest()):
                self.assertEqual(vad.install(target, fetch=lambda: good).read_bytes(), good)
                calls = []
                vad.install(target, fetch=lambda: calls.append(1) or good)  # present: no download
                self.assertEqual(calls, [])
                target.write_bytes(b"corrupt")
                with self.assertRaises(FileNotFoundError):              # offline never fetches
                    vad.install(target, fetch=lambda: calls.append(1) or good, allow_download=False)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
