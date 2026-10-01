# English adaptive speech research

Findings reviewed 2026-08-06. These are implementation inputs, not evidence
that a model is better on an individual Sotto corpus.

- [OpenAI Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo)
  documents the pruned/fine-tuned Whisper variant used as Sotto's immutable
  local English baseline; it is not the separately named Distil-Whisper line.
- [NVIDIA Parakeet TDT 0.6B v2 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
  documents the first independent challenger family.
- [senstella/parakeet-mlx](https://github.com/senstella/parakeet-mlx) provides
  the local Python/MLX adapter API (`from_pretrained`, then
  `model.transcribe(path).text`) used here.
- [FluidAudio README](https://github.com/FluidInference/FluidAudio),
  [models](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Models.md),
  and [benchmarks](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md)
  describe a current local Core ML v2 path, including benchmark and keyword-
  boosting work. They make FluidAudio a promising future receipt-backed
  adapter, but generic benchmarks do not establish personal Sotto quality.
  Parakeet-MLX remains first because its Python integration is smaller and
  easier to freeze under the current evaluator.
- [Distil-Whisper v3.5 model card](https://huggingface.co/distil-whisper/distil-large-v3.5)
  is a later candidate, not an assumed replacement for Sotto's baseline.
- Apple documents [AnalysisContext/contextual strings](https://developer.apple.com/documentation/speech/analysiscontext)
  and [DictationTranscriber customized language models](https://developer.apple.com/documentation/speech/dictationtranscriber/customizedlanguagemodel).
  Those APIs are future local adapters, not a reason to mix an Apple language
  model into the present frozen candidate family.
- Hugging Face's [Whisper fine-tuning guide](https://huggingface.co/docs/transformers/tasks/asr)
  describes the training workflow. Sotto deliberately does **not** auto-fine-
  tune a tiny personal set: 20 development samples and one 33-item pool are
  sufficient only for a conservative model-selection gate, not for trustworthy
  weight adaptation.

Sotto therefore starts with Python `parakeet-mlx` as the first local challenger
and keeps FluidAudio/CoreML and Apple custom language models behind future
receipt-backed adapters. Human-verified gold, exact offline snapshots, and a
fresh one-look evaluation pool are required before any deployment claim.
