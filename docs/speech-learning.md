# Local speech-quality learning

## Adaptive English policy

Adaptive mode is enabled only with `sotto.py --adaptive --language en`. It is
not weight training: installed or enabled models do not self-train. The system
collects only explicit human gold. Development needs **20 non-empty verified
examples and 200 words**. It then freezes a new **33-item** promotion pool;
promotion requires every disposition, at least **30 verified outcomes, 300
words, and 3 explicit no-speech outcomes** (at most three skips). It gets one
look only. A promotion is rollback/revocation-aware, including automatic
rollback after three consecutive deployed-champion runtime failures.

The operational commands are `learning-preflight`, `learning-status`,
`learning-review-list`, `learning-evaluate`, and `learning-rollback`.
Preflight is the only command that deliberately warms models and records their
exact local snapshot receipts. Model files and private data stay local, and an
offline launch with a missing receipt/cache fails rather than downloading or
changing the frozen identity. With zero human gold the state is collecting;
there is no automatic fine-tuning.

Sotto is local dictation, not a cloud transcription service. This release does
not upload speech, transcripts, corrections, or model data; it does not train
model weights automatically; and it does not add TTS, diarization, meeting
capture, or agent features.

## Choose a candidate before assuming an improvement

Sotto resolves a named profile first, then accepts an explicit repository or
language override when one is supplied.

| Profile | Local MLX repository | Default language |
| --- | --- | --- |
| `auto` | `mlx-community/whisper-large-v3-turbo` | auto |
| `hebrew-turbo` | `mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` | `he` |
| `hebrew-quality` | `mlx-community/ivrit-ai-whisper-large-v3-mlx` | `he` |

The ivrit.ai repositories are Hebrew-model candidates selected from external
Hebrew evidence. They have **not** been shown superior on Sotto data before a
personal-corpus A/B comparison exists. `hebrew-turbo` and `hebrew-quality`
force Hebrew unless `--language auto|he|en` explicitly changes that choice.
`--model REPOSITORY` explicitly replaces the repository selected by the
profile.

```bash
venv/bin/python sotto.py --profile auto
venv/bin/python sotto.py --profile hebrew-turbo
venv/bin/python sotto.py --profile hebrew-quality --language auto
venv/bin/python sotto.py --model mlx-community/ivrit-ai-whisper-large-v3-mlx
```

The selected model is downloaded to local storage only when it is actually run.
`learning-status`, `learning-list`, `learning-remove`, `export-learning`, and
`benchmark.py --dry-run` do not load a model. There is no speech upload or
newly added cloud provider.

## Glossary prompting

Pass a local glossary with `--glossary PATH`. Its format is either UTF-8 text,
one term per line, or a JSON array of strings. Empty lines, `#` comments in
text files, duplicates, and overlong entries are ignored; the accepted terms
and their combined prompt length are bounded so a glossary cannot become an
unbounded instruction.

```text
# sotto-glossary.txt
Sotto
PyObjC
זוהר
```

```json
["Sotto", "PyObjC", "זוהר"]
```

The glossary is supplied as an initial prompt only for captures of **30
seconds** or less. The installed `mlx-whisper` resets the initial prompt
between long-form windows when previous-text conditioning is disabled, so it
cannot honestly provide consistent glossary coverage for longer captures.

## Human review authorizes local enrollment

In adaptive English, saving **Correct Transcript**, choosing **Transcript is
correct**, or choosing **No speech / should be blank** is an explicit human
action. It automatically copies the paired canonical audio and resulting
reference into Sotto's private local learning corpus. There is no contradictory
second consent popup in adaptive mode. Ordinary non-adaptive correction keeps
its established separate manual enrollment behavior. Choose **Remove from
Learning Set**, Delete, or Clear history to revoke local copies.

If a correction is changed later, the earlier enrolled version is revoked
before the replacement is enrolled. Retry transcription is
revision-safe: it works from a snapshot and cannot overwrite a newer revision
or a human correction. The original model hypothesis is retained immutably in
history; retry results are recorded as later attempts.

Adaptive reviews use a revision-keyed, content-free outbox beside the reviewed
history row. It records only disposition metadata (and an allowed skip reason),
never the reference text. Recovery revalidates the row's canonical audio and
revision, then reloads the reference from history to finish adaptive state and
automatic enrollment. Until every such operation is complete, adaptive
generation publication/advancement fails closed. A frozen
incumbent/challenger/evaluator/development-input identity is a one-look
comparison: a consumed identity cannot be re-opened just because a new pool is
available; a legitimate changed frozen input is required. Incomplete outbox
rows pin their canonical history/audio beyond normal retention. The same
recovery-and-zero-pending gate protects direct evaluation, and identity ignores
local snapshot/cache paths so moving an identical model cannot create a second
look. Rollback uses that same semantic deployment identity, invalidating all
matching prior copies before selecting the next independent fallback.

Only schema-v2 canonical PCM16 rows captured in forced-English adaptive mode
are eligible for adaptive review, gold, or cohort membership. Legacy and
ordinary history rows remain available for their normal correction/manual
learning workflows but cannot create adaptive review recovery records.

Skips still count as resolved cohort dispositions (subject to the skip limit),
but are excluded from all model inference, paired statistics, no-speech
evidence, and promotion dependency digests. They are never interpreted as an
empty/no-speech reference.

For paired evaluation, both model arms consume the same canonical PCM16 input.
Path-based Parakeet inference receives the already identity-validated retained
WAV; live Parakeet requests use a private temporary WAV written from the exact
canonical bytes, never a lossy float re-encoding.

Private history, learning, adaptive state, receipts, locks, and temporary audio
are tightened by the application to owner-only permissions (`0700` directories,
`0600` files), including retained history WAVs and recoverable learning WAVs
created by earlier releases. Startup only repairs regular, direct children of
Sotto's private audio roots; it does not follow metadata paths or symlinks
outside the store. This does not rely on the shell's umask.

**Delete** first revokes every enrolled sample linked to that history item,
then removes the item. **Clear history** revokes the **entire active learning
corpus** - including samples whose history rows were already pruned - before
clearing the retained history. Conversely, ordinary history pruning does not
revoke independently consented learning data: the copied artifacts remain
available until you remove them explicitly, Clear history, or integrity
recovery finds the copied artifacts missing or corrupt.

## What stays on this Mac

Sotto stores its application data under:

```text
~/Library/Application Support/sotto/
```

History keeps an immutable initial hypothesis, revisions/attempts, and audio
needed for dictation history. On a human review action, Sotto creates separate
copies under its learning area:

- canonical inference audio: PCM16LE, mono, 16 kHz WAV - exactly the bytes
  prepared for inference;
- native-rate raw audio: separately copied when available, retaining its
  original capture rate;
- corrected transcript and provenance metadata.

Both copies have recorded content digests and audio identity metadata. File
publication is staged and fsynced; startup recovery removes unpublished or
orphaned artifacts and completes interrupted revocations. A revocation first
writes a durable revoking record, removes both copied audio artifacts, and
then leaves a minimal revoked tombstone. That makes deletion recoverable after
an interruption without retaining the correction or audio in the tombstone.

There is no automatic training, no automatic export, and no automatic upload.
Inspect the active learning count locally:

```bash
venv/bin/python sotto.py learning-status
```

To manage local samples even after ordinary history pruning, list the
active corpus and explicitly revoke an opaque sample ID:

```bash
venv/bin/python sotto.py learning-list
venv/bin/python sotto.py learning-remove --sample-id SAMPLE_ID
```

`learning-list` prints only opaque `sample_id` and `history_id` values; it
never prints transcript or reference text. `learning-remove` is an explicit,
local, cross-process-locked revocation of that one active sample. It does not
load a model.

Export is intentional and requires a directory that does not already contain
files:

```bash
venv/bin/python sotto.py export-learning --output <empty-dir>
```

Treat an export as a new sensitive copy that you control. It is the supported
way to prepare data for personal analysis or another local workflow; export
does not begin training.

## Offline A/B benchmark

`benchmark.py` evaluates audio and references on this Mac. It accepts a JSONL
manifest: each nonblank line has `audio` and `reference`; `id`, `language`,
`tags`, and `required_terms` are optional. Audio paths are relative to the
manifest location.

```json
{"id":"he-001","audio":"audio/he-001.wav","reference":"הטקסט המתוקן","language":"he","tags":["dictation","proper-name"],"required_terms":["Sotto"]}
{"id":"en-001","audio":"audio/en-001.wav","reference":"Open Sotto","language":"en","tags":["dictation"],"required_terms":["Sotto"]}
```

First validate a manifest and candidate names without importing MLX or
transcribing:

```bash
venv/bin/python benchmark.py manifests/held-out.jsonl --profile hebrew-turbo --dry-run
```

For a real, Hebrew-language stock-versus-candidate comparison, give each
candidate a durable label and write reports outside the manifest's audio
directory:

```bash
venv/bin/python benchmark.py manifests/held-out-hebrew.jsonl \
  --profile hebrew-turbo \
  --model stock=mlx-community/whisper-large-v3-turbo \
  --model hebrew-turbo=mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx \
  --output-json results/stock-vs-hebrew-turbo.json \
  --output-markdown results/stock-vs-hebrew-turbo.md
```

`--profile hebrew-turbo` deliberately supplies forced `he` to both candidates
in that comparison. For a different explicit policy, pass
`--language auto|he|en`; use `--glossary PATH` if that is part of the frozen
evaluation condition. A real benchmark loads the selected local models; a
dry-run does not.

The report includes these measures:

| Measure | What it answers |
| --- | --- |
| WER | How many word insertions, deletions, and substitutions differ from the reference. Lower is better. |
| CER | Character-level error rate; especially useful where word segmentation is less informative. Lower is better. |
| Entity-term accuracy | Whether each declared `required_terms` item is retained in the hypothesis. Higher is better. |
| Number accuracy | Whether reference numbers appear correctly in the hypothesis. Higher is better. |
| Hallucination proxy | A warning signal for excess, unsupported output; it is not a substitute for reviewing errors. Lower is better. |
| Language/tag slices | Whether results differ by each manifest `language` or `tags` group. |
| Latency | Per-sample transcription elapsed time, considered alongside accuracy. |

Small slices are marked directional. Do not treat a low-count WER, entity, or
latency difference as proof of a winner. Keep a held-out evaluation set frozen:
do not select a model, tune a glossary, or make a training decision based on
examples that later become its claimed independent evaluation. Review the raw
hypotheses as well as aggregates, especially for names, numbers, and language
switches.

## Training gate

Build a **separate, opt-in corrected corpus** if you may eventually consider
fine-tuning. **5–10 hours** of material is a decision checkpoint, not proof
that the corpus is sufficient. First compare routing, language policy,
glossary prompting, and stock/Hebrew model swaps on a frozen held-out set.

Fine-tuning becomes worth investigating only when those comparisons show
systematic residual errors that a trained adaptation could plausibly address,
and only with held-out data to test it. This release trains no model weights.
