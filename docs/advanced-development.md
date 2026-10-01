# Advanced development (outside the source alpha)

The public source alpha uses baseline Whisper or manually selected Nemotron.
Do not enable `--adaptive`, run learning preflight, provision teachers, or
install a background learning worker for alpha testing. These notes describe
experimental development workflows; they are not onboarding steps.

Sealed deployment remains a separate maintainer workflow through
`scripts/rollout.sh`. Templates contain placeholders and must be rendered by
`launchd_templates.py`; copying a template directly into LaunchAgents is not
a supported installation. Source changes invalidate source-bound receipts.

## Experimental learning and evaluation

### Adaptive English (human-grounded, local only)

`--adaptive` is an English-only runtime (`--language en` is forced). It does
not make installed model weights self-train. Genuine improvement requires
human-verified gold: 20 development examples and 200 words, followed by a
fresh fixed 33-item review pool with at least 30 verified outcomes, 300 words,
and 3 explicit no-speech outcomes. The pool is evaluated once; a promoted
challenger can be rolled back explicitly or automatically after three runtime
failures, and revoking dependent gold removes its promotion eligibility.
Each human review is first written as a revision-keyed, content-free recovery
record; on restart Sotto verifies the retained canonical audio and completes
the matching adaptive/enrollment operation before it can open or advance a
generation. A fixed incumbent/challenger/configuration comparison is consumed
once: repeating it requires a changed frozen human input, not merely a new
cohort. Generation publication itself is one locked transition, so concurrent
reconciliation cannot open competing pools.
Incomplete review records pin their history/audio past ordinary retention, and
the same zero-pending recovery gate runs before both reconciliation and direct
evaluation. Comparison identity uses semantic model/configuration fields—not
local cache or snapshot paths.
Rollback also treats semantically identical prior deployments as one identity:
after a fallback is selected, another copy of that same model/configuration is
never redeployed in a later manual or runtime-failure rollback.
Only new schema-v2, canonical PCM16, forced-English adaptive captures can enter
the adaptive review queue; retained legacy/non-adaptive history keeps its
ordinary correction and manual-learning behavior.
Skipped review items resolve the fixed cohort administratively but are not sent
to either model or used in scoring, hallucination checks, or promotion evidence.

### Autonomous silver lane (shadow-only)

The optional autonomous lane is separate from `LearningStore` and never calls
machine-generated text gold.  It retains the single canonical 16 kHz WAV while
a content-free history marker is pending, then runs two **disjoint frozen
teacher families** offline (not statistically independent): Qwen3-ASR 1.7B
English and Granite 4.0 1B Speech English.  V1 accepts only exact equality of
the frozen normalized token sequence; blanks/no-speech, errors, mismatches,
numbers, names, acronyms, negations, and command-like text abstain. Teacher
surface hypotheses are memory-only.  Accepted evidence is labelled SILVER.

`adaptive_worker.py` is headless: it imports neither AppKit nor Quartz and can
run without the dictation application. It validates history revision/audio
digest, clear epoch, receipt, lineage, and policy immediately before accepting
a label. Human correction/delete/clear revokes prior silver first. The worker
opens a prospective post-boundary horizon before teacher work, freezes the
receipt-backed Whisper champion and disjoint available candidate arms, and
evaluates the identical accepted cohort synchronously under evaluator leases.
Only persisted calibration/cohort/receipt/code evidence can authorize a 5%
session-sticky provisional route; durable canary observations advance it or
roll it back. Human-reviewed gold keeps higher precedence, and no fine-tuning
exists in V1.

Do not install launchd or run the autonomous worker before public Calibration
V2 and strict sealed readiness pass. After that it may run shadow-only to
collect the personal horizon; candidate routing remains impossible until
personal evaluation authorizes at least 500 captures over at least 7 days.
Teacher packages are intentionally isolated from the live app requirements in
`teachers/requirements-qwen.txt` and `teachers/requirements-granite.txt`.
Provision computes (rather than trusts) absolute interpreter/package, adapter,
canonicalizer, and full snapshot identities. Runtime is offline and must never
resolve Hub ids:

```bash
venv/bin/python adaptive_worker.py --base-dir "$HOME/Library/Application Support/sotto"
venv/bin/python adaptive_worker.py --base-dir "$HOME/Library/Application Support/sotto" --status
venv/bin/python teacher_provision.py --base-dir "$HOME/Library/Application Support/sotto" \
  --qwen-python "$HOME/Library/Application Support/sotto/teacher-runtimes/qwen3-asr-1.7b/venv/bin/python" \
  --qwen-snapshot "$HOME/Library/Application Support/sotto/huggingface/models--Qwen--Qwen3-ASR-1.7B/snapshots/<pinned-revision>" \
  --granite-python "$HOME/Library/Application Support/sotto/teacher-runtimes/granite-4.0-1b-speech/venv/bin/python" \
  --granite-snapshot "$HOME/Library/Application Support/sotto/huggingface/models--ibm-granite--granite-4.0-1b-speech/snapshots/<pinned-revision>"
venv/bin/python calibration_worker.py --base-dir "$HOME/Library/Application Support/sotto" --manifest /absolute/frozen-calibration.json \
  --test-parquet /absolute/en-test-00000-of-00001.parquet \
  --validation-parquet /absolute/en-validation-00000-of-00001.parquet \
  --pyarrow-python /absolute/pyenv/versions/3.11.7/bin/python3.11
plutil -lint launchd/com.zohartito.sotto.adaptive-silver.plist
```

Provisioning writes the private `adaptive-learning/silver/teacher-receipts.json`
map keyed by `qwen3-asr-1.7b-en` and `granite-4.0-1b-speech-en`. Every entry
must bind `family`, absolute `interpreter`, absolute `adapter`, `snapshot_path`,
`snapshot_digest`, `package_versions`, `adapter_hash`, `decode`, and
`canonicalizer_hash`; invalid/missing records stop the worker before audio is
read. The worker, not caller-provided metrics, derives provisional
authorization from frozen evidence; there is no CLI command that bypasses
those gates. Parakeet requires FFmpeg under launchd; the template carries
`__SOTTO_HOME__`/`__SOTTO_FFMPEG__` placeholders plus the required safe PATH.
Never install a raw template: render it first with
`/usr/bin/python3 -I -S -B launchd_templates.py --template <template> --output <LaunchAgents plist>`
(`scripts/rollout.sh` does this for the app agent; `SOTTO_FFMPEG` selects the
FFmpeg path, default `/opt/homebrew/bin/ffmpeg`).

Calibration V2 validates both pinned parquet bytes and independently derives
the frozen union before any teacher call. Give it regular local files: copy a
cached/symlinked Hub artifact to a private regular file first. It emits no
teacher output until this exact freeze succeeds.

The live app venv deliberately does not import `pyarrow`.  The required
`--pyarrow-python` is a pinned reader/canonicalizer Python 3.11.7 / Unicode 14 /
pyarrow 23.0.1 / NumPy 2.0.2 helper: it may only stream the two local parquet
files into a private PCM/reference projection.  Its runtime identity is bound
into the app protocol; the app independently verifies the compiled source,
membership, PCM/reference projection, and ledger commitments, and remains the
policy, receipt, and teacher authority. Before the frozen public Calibration
V2 build and validation succeed, do not run teacher inference or install the
service. After public pass, provisioning, calibration, status, and service
actions use verified sealed entrypoints; the worker remains `shadow_only` with
no routing until the personal gates authorize it.

Do not install the launchd template until public Calibration V2 and strict
sealed readiness pass. It may then collect shadow-only evidence; routing stays
disabled until personal evaluation authorizes at least 500 captures over at
least 7 days. `SilverStore.status()` and
`DeploymentController.status()` are content-free inspection interfaces. Clear
uses SQLite secure delete/checkpoint/vacuum but cannot erase SSD snapshots,
backups, or filesystem journal remnants.

The repository launchd templates are sealed-release templates, not commands to
install during development. `scripts/rollout.sh` backs up the installed app
plist, installs the template from the activated sealed bundle, and reloads the
job because `kickstart` alone does not reread changed arguments. Their bootstrap verifies the immutable source bundle
and its pinned virtualenv launcher, package closure, and FFmpeg identity before
re-executing the configured venv Python. It runs with bytecode writes disabled;
any bundle, launcher-chain, package, or runtime drift refuses to start rather
than falling back to the mutable checkout or a base Homebrew interpreter.

Once the public Calibration V2 gate has passed, the headless worker may run in
`shadow_only` mode to collect the first prospective personal horizon. This is
evidence collection only: no candidate routing or authorization is possible
until the separate frozen personal-horizon evaluation has passed. Use
`adaptive_worker.py --base-dir <private-state-root> --status` for the strict,
content-free sealed readiness JSON; it exits zero only when that shadow
collection policy is ready and otherwise fails closed without constructing or
repairing state.

```bash
venv/bin/python sotto.py learning-preflight   # the only deliberate model warmup
venv/bin/python sotto.py --adaptive --language en
venv/bin/python sotto.py learning-status
venv/bin/python sotto.py learning-review-list
venv/bin/python sotto.py learning-evaluate
venv/bin/python sotto.py learning-rollback
```

All models, snapshots, receipts, audio, and evaluation remain on this Mac;
offline launch fails clearly if the preflighted local snapshot is absent. The
initial candidate is `mlx-community/whisper-large-v3-turbo`; the first
challenger is local `mlx-community/parakeet-tdt-0.6b-v2`. At zero gold Sotto is
simply collecting—there is no auto-fine-tuning. See [adaptive research](docs/english-adaptive-research.md).

The production LaunchAgent keeps the Whisper model resident but not the
microphone. With `--idle-release 0`, key-down starts `AVAudioEngine` and
installs its input tap; key-up immediately stops the engine and removes the
tap. Sotto therefore does not capture or retain audio while idle, and the
macOS microphone indicator is lit only for an active gesture. Production uses
the receipt-independent Whisper baseline plus the sealed local
`config/sotto-glossary.txt`; the adaptive candidate route remains an explicit,
separately gated mode until its calibration and personal-evidence requirements
pass.
After four idle minutes, the next press queues one serialized model refresh
while speech is being captured, hiding Metal/page-in warmup behind the user's
utterance instead of adding it after release.

The default `--profile auto` selects the stock multilingual candidate
`mlx-community/whisper-large-v3-turbo`. The menu-bar `◦` menu defaults to
`Language: Automatic (English + Hebrew)`: each push-to-talk recording is
detected independently. Use its Language submenu to force English or Hebrew
for short or ambiguous phrases; the choice is local, persistent, and applies
from the next recording without loading another model. For Hebrew, Sotto also offers two
locally-run candidates selected from external Hebrew evidence (not yet proven
better on your personal corpus):

| Profile | Repository | Language default |
| --- | --- | --- |
| `auto` | `mlx-community/whisper-large-v3-turbo` | auto-detect |
| `hebrew-turbo` | `mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` | forced `he` |
| `hebrew-quality` | `mlx-community/ivrit-ai-whisper-large-v3-mlx` | forced `he` |

Use `--profile auto|hebrew-turbo|hebrew-quality`; use
`--language auto|he|en` to explicitly override its language decision; and use
`--model REPOSITORY` only when an explicit repository override is wanted.

```bash
# Hebrew, fast candidate; the profile forces Hebrew unless --language overrides it.
venv/bin/python sotto.py --profile hebrew-turbo

# Explicit language and model overrides remain local.
venv/bin/python sotto.py --profile auto --language he \
  --model mlx-community/ivrit-ai-whisper-large-v3-mlx

# Offer bounded proper names and terms to short captures.
venv/bin/python sotto.py --glossary ~/Documents/sotto-glossary.txt

# Inspect, explicitly remove, or deliberately export the separately enrolled corpus.
venv/bin/python sotto.py learning-status
venv/bin/python sotto.py learning-list
venv/bin/python sotto.py learning-remove --sample-id SAMPLE_ID
venv/bin/python sotto.py export-learning --output <empty-dir>
```

Model files download locally only when their profile or repository is selected
and run. Corpus administration (`learning-status`, `learning-list`, and
`learning-remove`), export, and benchmark `--dry-run` do not load models.
Sotto added no speech upload or cloud provider.

A glossary is a local UTF-8 file with one term or short phrase per line (blank lines and `#`
comments are ignored), or a JSON file containing a string list. Sotto bounds
the number and size of accepted terms to keep the prompt small. Prompting is
applied only to captures of **30 seconds** or less: installed `mlx-whisper`
resets the initial prompt between long-form windows when previous-text
conditioning is disabled, so applying it to longer recordings would give
misleading coverage.

For the full consent, storage, export, evaluation, and training guidance, see
[Speech-quality learning](docs/speech-learning.md).

## Offline backup and restore

Stop both the Sotto app and adaptive worker before invoking this tool; use it
only against that explicitly quiesced, copied application-state root.  It
detects in-window source/SQLite drift but cannot prove an absent external
owner, and never downloads models or starts Sotto.  Backups
contain private mutable state only (not model/Hugging Face caches, releases,
logs, locks, or scheduler debris), are content-addressed, and verify before a
restore is staged.  Restore requires a fresh empty destination and forces its
deployment ledger to `shadow_only`; a restored candidate needs fresh evidence
and authorization before it can route traffic.

```bash
venv/bin/python offline_backup.py backup --source /private/state-copy --bundle /private/backups/new
venv/bin/python offline_backup.py verify --bundle /private/backups/<digest>
venv/bin/python offline_backup.py restore --bundle /private/backups/<digest> --target /private/restore-fixture
```
