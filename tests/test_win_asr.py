"""win_asr: pinned cache-first model resolution and CUDA -> CPU fallback.

No real model loads and no network: model factories and downloads are fakes,
and the fake model asserts the exact kwargs mapping, because
``sotto.transcribe_canonical_samples`` and ``sotto.warm_speech_runtime``
forward their kwargs verbatim into ``transcribe``.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import win_asr

TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
TURBO_SHA = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8"


class FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModel:
    def __init__(self, fail: Exception | None = None, probabilities=None) -> None:
        self.calls: list[tuple] = []
        self.detections: list = []
        self.fail = fail
        # faster-whisper's detect_language: (top, its probability, all sorted)
        self.probabilities = probabilities or [("en", 0.6), ("he", 0.3), ("fr", 0.1)]

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if self.fail is not None:
            raise self.fail
        return [FakeSegment("hello "), FakeSegment("world")], object()

    def detect_language(self, audio=None, **_kwargs):
        self.detections.append(audio)
        return self.probabilities[0][0], self.probabilities[0][1], list(self.probabilities)


def write_snapshot(directory: Path, files=win_asr.MODEL_FILES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in files:
        (directory / name).write_bytes(b"x")
    return directory


class WinAsrProfilesTest(unittest.TestCase):
    def test_profile_ids_are_the_verified_ct2_repos(self) -> None:
        # Verified against the HuggingFace API on 2026-09-20 (ungated, CT2).
        self.assertEqual(win_asr.PROFILES, {
            "auto": TURBO,
            "hebrew-turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",
            "hebrew-quality": "ivrit-ai/whisper-large-v3-ct2",
        })

    def test_every_profile_is_pinned_to_a_full_commit(self) -> None:
        for repo in win_asr.PROFILES.values():
            self.assertRegex(win_asr.REVISIONS[repo], r"^[0-9a-f]{40}$")
            self.assertEqual(win_asr.pinned(repo), (repo, win_asr.REVISIONS[repo]))

    def test_explicit_models_must_carry_a_commit(self) -> None:
        sha = "a" * 40
        self.assertEqual(win_asr.pinned(f"org/model@{sha}"), ("org/model", sha))
        for unpinned in ("org/model", "org/model@main", f"model@{sha}", f"org/model@{sha[:12]}"):
            with self.assertRaises(ValueError, msg=unpinned):
                win_asr.pinned(unpinned)

    def test_fast_speed_model_is_pinned_multilingual_small(self) -> None:
        # Chosen by the 2026-09-29 benchmark (docs/windows-alpha.md).
        self.assertEqual(win_asr.FAST_REPO, "Systran/faster-whisper-small")
        self.assertEqual(win_asr.pinned(win_asr.FAST_REPO),
                         (win_asr.FAST_REPO, "536b0662742c02347bc0e980a01041f333bce120"))
        # 80-mel port: no preprocessor config, a text vocabulary.
        self.assertEqual(win_asr.required_files(win_asr.FAST_REPO),
                         ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"))
        self.assertEqual(win_asr.required_files(TURBO), win_asr.MODEL_FILES)

    def test_repo_for_profile(self) -> None:
        self.assertEqual(win_asr.repo_for_profile("hebrew-turbo"),
                         "ivrit-ai/whisper-large-v3-turbo-ct2")

    def test_unknown_profile_raises_with_choices(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            win_asr.repo_for_profile("nope")
        self.assertIn("auto", str(ctx.exception))
        self.assertIn("hebrew-quality", str(ctx.exception))


class ResolveModelDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "hf"
        self.snapshot = self.cache / "hub" / "models--deepdml--faster-whisper-large-v3-turbo-ct2" / "snapshots" / TURBO_SHA
        self.logs: list[str] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _resolve(self, *, offline: bool, download=None, model: str = TURBO) -> Path:
        return win_asr.resolve_model_dir(model, cache_dir=self.cache, offline=offline,
                                         log=self.logs.append, download=download)

    def test_snapshot_path_is_the_hub_cache_layout(self) -> None:
        self.assertEqual(win_asr.snapshot_dir(self.cache, TURBO, TURBO_SHA), self.snapshot)

    def test_cached_snapshot_resolves_without_download_or_network(self) -> None:
        write_snapshot(self.snapshot)
        download = mock.Mock(side_effect=AssertionError("download attempted"))
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network")):
            for offline in (False, True):
                self.assertEqual(self._resolve(offline=offline, download=download), self.snapshot)
        download.assert_not_called()
        self.assertEqual(self.logs, [])

    def test_first_use_downloads_exactly_the_pinned_commit_once(self) -> None:
        def download(**kwargs):
            write_snapshot(self.snapshot)
            return str(self.snapshot)

        fake = mock.Mock(side_effect=download)
        self.assertEqual(self._resolve(offline=False, download=fake), self.snapshot)
        fake.assert_called_once_with(repo_id=TURBO, revision=TURBO_SHA,
                                     cache_dir=str(self.cache / "hub"),
                                     allow_patterns=list(win_asr.MODEL_FILES))
        self.assertEqual(self._resolve(offline=False, download=fake), self.snapshot)
        self.assertEqual(fake.call_count, 1)

    def test_offline_missing_model_is_a_plain_actionable_error(self) -> None:
        write_snapshot(self.snapshot, files=("config.json", "model.bin"))  # interrupted download
        download = mock.Mock()
        with self.assertRaises(win_asr.ModelUnavailable) as ctx:
            self._resolve(offline=True, download=download)
        download.assert_not_called()
        message = str(ctx.exception)
        self.assertIsInstance(ctx.exception, win_asr.SetupError)
        for fragment in (TURBO, TURBO_SHA[:12], str(self.cache), "SOTTO_OFFLINE", "without those variables"):
            self.assertIn(fragment, message)

    def test_failed_or_incomplete_download_is_a_setup_error(self) -> None:
        with self.assertRaisesRegex(win_asr.SetupError, "could not download"):
            self._resolve(offline=False, download=mock.Mock(side_effect=OSError("no route")))
        with self.assertRaisesRegex(win_asr.SetupError, "incomplete"):
            self._resolve(offline=False, download=mock.Mock(return_value=str(self.snapshot)))

    def test_local_directory_is_used_as_is(self) -> None:
        local = write_snapshot(Path(self._tmp.name) / "my-model")
        download = mock.Mock()
        self.assertEqual(self._resolve(offline=True, download=download, model=str(local)), local.absolute())
        download.assert_not_called()
        # Without tokenizer.json faster-whisper would fetch one from the Hub.
        partial = write_snapshot(Path(self._tmp.name) / "partial",
                                 files=("config.json", "model.bin", "vocabulary.json"))
        with self.assertRaisesRegex(win_asr.SetupError, "tokenizer.json"):
            self._resolve(offline=True, download=download, model=str(partial))
        download.assert_not_called()

    def test_fast_model_downloads_and_resolves_its_own_file_set(self) -> None:
        sha = win_asr.REVISIONS[win_asr.FAST_REPO]
        snapshot = win_asr.snapshot_dir(self.cache, win_asr.FAST_REPO, sha)
        files = win_asr.required_files(win_asr.FAST_REPO)

        def download(**kwargs):
            write_snapshot(snapshot, files=files)
            return str(snapshot)

        fake = mock.Mock(side_effect=download)
        self.assertEqual(self._resolve(offline=False, download=fake, model=win_asr.FAST_REPO), snapshot)
        fake.assert_called_once_with(repo_id=win_asr.FAST_REPO, revision=sha,
                                     cache_dir=str(self.cache / "hub"), allow_patterns=list(files))
        # Cached: offline resolves with no download; a local copy is accepted too.
        self.assertEqual(self._resolve(offline=True, download=mock.Mock(side_effect=AssertionError),
                                       model=win_asr.FAST_REPO), snapshot)
        self.assertEqual(self._resolve(offline=True, model=str(snapshot)), snapshot.absolute())

    def test_unpinned_hub_id_is_rejected_before_any_download(self) -> None:
        download = mock.Mock()
        with self.assertRaises(ValueError):
            self._resolve(offline=False, download=download, model="org/unpinned")
        download.assert_not_called()


class DeviceSelectionTest(unittest.TestCase):
    def test_cli_beats_environment_and_blank_means_auto(self) -> None:
        self.assertEqual(win_asr.requested_device(None, {}), "auto")
        self.assertEqual(win_asr.requested_device(None, {"SOTTO_DEVICE": "  "}), "auto")
        self.assertEqual(win_asr.requested_device(None, {"SOTTO_DEVICE": "CPU"}), "cpu")
        self.assertEqual(win_asr.requested_device("cuda", {"SOTTO_DEVICE": "cpu"}), "cuda")
        with self.assertRaises(ValueError):
            win_asr.requested_device(None, {"SOTTO_DEVICE": "gpu"})

    def test_device_plans(self) -> None:
        self.assertEqual(win_asr.device_plan("auto", 1), [win_asr.CUDA, win_asr.CPU])
        self.assertEqual(win_asr.device_plan("auto", 0), [win_asr.CPU])
        self.assertEqual(win_asr.device_plan("cpu", 2), [("cpu", "int8")])
        self.assertEqual(win_asr.device_plan("cuda", 0), [("cuda", "float16")])


class LocalCT2WhisperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logs: list[str] = []
        self.built: list[tuple[str, str, str]] = []
        self.models: dict[str, FakeModel] = {"cuda": FakeModel(), "cpu": FakeModel()}
        self.construct_fail: dict[str, Exception] = {}

    def factory(self, path, *, device, compute_type):
        self.built.append((path, device, compute_type))
        if device in self.construct_fail:
            raise self.construct_fail[device]
        return self.models[device]

    def whisper(self, device="auto", cuda_devices=1) -> win_asr.LocalCT2Whisper:
        return win_asr.LocalCT2Whisper(Path("C:/models/turbo"), device=device, model_factory=self.factory,
                                       cuda_devices=lambda: cuda_devices, log=self.logs.append)

    def _samples(self) -> np.ndarray:
        return np.zeros(16_000, dtype=np.float32)

    def test_kwargs_mapping_text_join_and_local_path_only(self) -> None:
        whisper = self.whisper()
        samples = self._samples()
        result = whisper.transcribe(samples, path_or_hf_repo=TURBO, condition_on_previous_text=False,
                                    word_timestamps=False, language="he",
                                    initial_prompt="Speech context: foo.", some_future_flag=True)
        self.assertEqual(result, {"text": "hello world"})
        self.assertEqual(self.built, [(str(Path("C:/models/turbo")), "cuda", "float16")])
        audio, kwargs = self.models["cuda"].calls[0]
        self.assertIs(audio, samples)
        self.assertEqual(kwargs, {"language": "he", "initial_prompt": "Speech context: foo.",
                                  "condition_on_previous_text": False, "word_timestamps": False})
        self.assertNotIn(TURBO, repr(self.built))

    def test_auto_language_passes_none_and_model_stays_resident(self) -> None:
        whisper = self.whisper()
        whisper.transcribe(self._samples(), path_or_hf_repo="label")
        whisper.transcribe(self._samples(), path_or_hf_repo="label")
        self.assertIsNone(self.models["cuda"].calls[0][1]["language"])
        self.assertEqual(len(self.built), 1)
        self.assertEqual((whisper.device, whisper.compute_type), ("cuda", "float16"))
        self.assertEqual(self.logs, ["✓ ASR device: cuda (float16)"])

    def test_cuda_construction_failure_falls_back_to_cpu_int8(self) -> None:
        self.construct_fail["cuda"] = RuntimeError("CUDA driver version is insufficient")
        whisper = self.whisper()
        self.assertEqual(whisper.transcribe(self._samples())["text"], "hello world")
        self.assertEqual([b[1:] for b in self.built], [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual((whisper.device, whisper.compute_type), ("cpu", "int8"))
        self.assertIn("falling back to CPU", self.logs[0])
        self.assertEqual(self.logs[-1], "✓ ASR device: cpu (int8)")

    def test_lazy_cublas_failure_at_first_transcription_falls_back(self) -> None:
        # CTranslate2 loads cuBLAS at the first matmul, not in the constructor.
        self.models["cuda"] = FakeModel(fail=RuntimeError("Library cublas64_12.dll is not found"))
        whisper = self.whisper()
        self.assertEqual(whisper.transcribe(self._samples())["text"], "hello world")
        self.assertEqual(whisper.device, "cpu")
        whisper.transcribe(self._samples())
        self.assertEqual(len(self.models["cuda"].calls), 1)
        self.assertEqual(len(self.models["cpu"].calls), 2)

    def test_forced_cpu_and_no_cuda_device_never_touch_cuda(self) -> None:
        for whisper in (self.whisper(device="cpu", cuda_devices=2), self.whisper(cuda_devices=0)):
            whisper.transcribe(self._samples())
            self.assertEqual(whisper.device, "cpu")
        self.assertEqual({b[1] for b in self.built}, {"cpu"})

    def test_forced_cuda_failure_does_not_fall_back(self) -> None:
        self.construct_fail["cuda"] = RuntimeError("no CUDA-capable device")
        with self.assertRaises(win_asr.SetupError) as ctx:
            self.whisper(device="cuda").transcribe(self._samples())
        self.assertIn("--device cpu", str(ctx.exception))
        self.assertEqual([b[1] for b in self.built], ["cuda"])

    def test_invalid_device_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.whisper(device="tpu")

    def test_automatic_chooses_only_among_the_allowed_languages(self) -> None:
        # French is the model's own top guess, but the user speaks en/he only.
        self.models["cuda"].probabilities = [("fr", 0.5), ("he", 0.3), ("en", 0.2)]
        whisper = self.whisper()
        result = whisper.transcribe(self._samples(), allowed_languages=("en", "he"))
        self.assertEqual(self.models["cuda"].calls[-1][1]["language"], "he")
        self.assertEqual(result, {"text": "hello world", "language": "he"})
        self.assertEqual(len(self.models["cuda"].detections), 1)

    def test_one_allowed_language_or_a_fixed_language_skips_detection(self) -> None:
        whisper = self.whisper()
        result = whisper.transcribe(self._samples(), allowed_languages=("he",))
        self.assertEqual(self.models["cuda"].calls[-1][1]["language"], "he")
        self.assertEqual(result["language"], "he")
        result = whisper.transcribe(self._samples(), language="en", allowed_languages=("he", "fr"))
        self.assertEqual(self.models["cuda"].calls[-1][1]["language"], "en")
        self.assertNotIn("language", result)
        self.assertEqual(self.models["cuda"].detections, [])

    def test_language_restricted_view_passes_the_set_and_reports_the_choice(self) -> None:
        whisper = self.whisper()
        view = win_asr.LanguageRestricted(whisper, ("en", "he"))
        self.assertEqual(view.transcribe(self._samples(), path_or_hf_repo="label")["text"], "hello world")
        self.assertEqual(view.last_language, "en")  # read by sotto.transcribe_canonical_samples
        self.assertEqual(self.models["cuda"].calls[-1][1]["language"], "en")
        view.transcribe(self._samples(), language="he")
        self.assertIsNone(view.last_language)


class OnePassModel:
    """faster-whisper's surface for Automatic: features, encode, detect, decode.

    ``transcribe`` encodes its first window exactly as generate_segments does,
    so a reused encoder output shows up as one encode call instead of two.
    """

    class Extractor:
        nb_max_frames = 3000

        def __call__(self, samples):
            frames = len(samples) // 160 + 1
            return np.tile(np.linspace(0, 1, frames, dtype=np.float32), (80, 1))

    def __init__(self) -> None:
        self.feature_extractor = self.Extractor()
        self.model = self
        self.is_multilingual = True
        self.encodes: list = []
        self.languages: list = []

    def encode(self, features):
        self.encodes.append(features.shape)
        return ("encoder-output", len(self.encodes))

    def detect_language(self, encoder_output):  # ctranslate2.models.Whisper.detect_language
        return [[("<|fr|>", 0.6), ("<|he|>", 0.3), ("<|en|>", 0.1)]]

    def transcribe(self, samples, **options):
        from faster_whisper.audio import pad_or_trim
        features = self.feature_extractor(samples)
        window = pad_or_trim(features[:, :features.shape[-1] - 1])
        output = self.encode(window)
        self.languages.append((options["language"], output))
        return [FakeSegment("one pass")], object()


class OnePassTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import faster_whisper.audio  # noqa: F401
        except ImportError:
            self.skipTest("faster-whisper is not installed (Windows requirements)")

    def test_short_clip_reuses_the_detection_encoder_pass(self) -> None:
        model = OnePassModel()
        whisper = win_asr.LocalCT2Whisper(Path("C:/m"), device="cpu", model_factory=lambda *a, **k: model,
                                          cuda_devices=lambda: 0, log=lambda message: None)
        result = whisper.transcribe(np.zeros(16_000 * 5, dtype=np.float32), allowed_languages=("en", "he"))
        self.assertEqual(result, {"text": "one pass", "language": "he"})
        self.assertEqual(len(model.encodes), 1, "one encoder pass for detection and decode")
        self.assertEqual(model.languages, [("he", ("encoder-output", 1))])
        self.assertNotIn("encode", model.__dict__, "the model's own encode is restored")
        # A second clip reuses nothing from the first.
        whisper.transcribe(np.ones(16_000 * 3, dtype=np.float32), allowed_languages=("en", "he"))
        self.assertEqual(len(model.encodes), 2)

    def test_long_clip_uses_the_normal_detection(self) -> None:
        model = OnePassModel()
        self.assertIsNone(win_asr._first_window(model, np.zeros(16_000 * 31, dtype=np.float32)))
        self.assertIsNotNone(win_asr._first_window(model, np.zeros(16_000 * 29, dtype=np.float32)))


class LanguageChoiceTest(unittest.TestCase):
    def test_choose_language_is_argmax_within_the_set_with_user_order_ties(self) -> None:
        probabilities = {"fr": 0.4, "en": 0.3, "he": 0.3}
        self.assertEqual(win_asr.choose_language(probabilities, ("he", "en")), "he")
        self.assertEqual(win_asr.choose_language(probabilities, ("en", "he")), "en")
        self.assertEqual(win_asr.choose_language({"en": 0.7, "ar": 0.2}, ("ar", "en")), "en")
        self.assertEqual(win_asr.choose_language({}, ("es", "de")), "es")
        with self.assertRaises(ValueError):
            win_asr.choose_language(probabilities, ())

    def test_clearly_other_speech_is_written_in_its_own_language_not_translated(self) -> None:
        # The Mac's rule: allowed all < 0.25 while another language >= 0.5.
        self.assertEqual(win_asr.choose_language({"es": 0.86, "en": 0.11, "he": 0.01}, ("en", "he")), "es")
        self.assertEqual(win_asr.choose_language({"es": 0.49, "en": 0.11}, ("en", "he")), "en")
        self.assertEqual(win_asr.choose_language({"es": 0.6, "en": 0.25}, ("en", "he")), "en")


def _cuda_extras_installed() -> bool:
    try:
        import nvidia.cublas  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(sys.platform == "win32", "cuBLAS DLL lookup is Windows-only")
class WinAsrCublasTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = os.environ.get("PATH", "")

    def tearDown(self) -> None:
        os.environ["PATH"] = self._old_path

    def test_noop_when_dll_already_resolvable(self) -> None:
        with mock.patch("ctypes.util.find_library", return_value="cublas64_12.dll"):
            win_asr._ensure_cublas()
        self.assertEqual(os.environ.get("PATH", ""), self._old_path)

    def test_cpu_only_install_leaves_the_dll_path_alone(self) -> None:
        with mock.patch("ctypes.util.find_library", return_value=None), \
                mock.patch.dict(sys.modules, {"nvidia.cublas": None}):
            win_asr._ensure_cublas()
        self.assertEqual(os.environ.get("PATH", ""), self._old_path)

    @unittest.skipUnless(_cuda_extras_installed(), "optional CUDA extras (nvidia-cublas-cu12) not installed")
    def test_adds_pip_package_bin_dir_when_missing(self) -> None:
        # The ctranslate2 wheel links CUDA 12 but does not bundle cuBLAS;
        # the pip nvidia-cublas-cu12 package must be put on the DLL path.
        with mock.patch("ctypes.util.find_library", return_value=None):
            win_asr._ensure_cublas()
        first = os.environ.get("PATH", "").split(os.pathsep)[0]
        self.assertIn("cublas", first)
        self.assertTrue(first.endswith("bin"))


if __name__ == "__main__":
    unittest.main()
