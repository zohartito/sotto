"""Offline public calibration runner for a pre-frozen manifest.

The manifest may contain public references while running, but no reference or
teacher text is written to the personal silver database or process output.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Iterator

from silver_store import SilverStore
from inference_scheduler import InferenceScheduler
from teacher_backends import OfflineTeacherRunner, REQUIRED_TEACHERS, load_receipts
from teacher_consensus import exact_unanimous, normalized_tokens, policy_hash
from calibration_manifest_builder import derive_plan_with_reader, _protocol_hash, SOURCE_HASH, MEMBERSHIP_HASH, LEDGER_HASH, SPLIT_COUNTS, protocol_bundle_identity, PROTOCOL_FILES
from audio_codec import read_canonical_wav


def _hash_file(path: Path) -> str:
    if stat.S_ISLNK(os.lstat(path).st_mode) or not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError("unsafe calibration audio")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def _verify_observed_wav(path: Path, expected_wav_sha: str, expected_pcm_sha: str,
                         expected_samples: int) -> None:
    hex64=lambda value: isinstance(value,str) and len(value)==64 and all(char in "0123456789abcdef" for char in value)
    if not hex64(expected_wav_sha) or not hex64(expected_pcm_sha) or isinstance(expected_samples,bool) or not isinstance(expected_samples,int) or expected_samples <= 0:
        raise RuntimeError("invalid immutable calibration identity")
    try:
        metadata=os.lstat(path)
    except OSError as exc:
        raise RuntimeError("unsafe observed calibration audio") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode)!=0o400:
        raise RuntimeError("unsafe observed calibration audio")
    if _hash_file(path)!=expected_wav_sha:
        raise RuntimeError("calibration audio changed")
    _samples,identity=read_canonical_wav(path)
    if identity.sha256!=expected_pcm_sha or identity.sample_count!=expected_samples:
        raise RuntimeError("calibration canonical PCM identity mismatch")


def _transcribe_observed_teacher(runner: Any, scheduler: InferenceScheduler, teacher: Any,
                                 observed: Path, item_hash: str, receipt_hash: str,
                                 spec: dict[str, Any]) -> str:
    """Transcribe one immutable observation under the evaluator lease."""
    if not isinstance(spec,dict): raise RuntimeError("invalid immutable calibration identity")
    expected_wav=spec.get("audio_sha256"); expected_pcm=spec.get("pcm_sha256"); expected_samples=spec.get("sample_count")
    def receipt_current() -> None:
        try: current=runner.validate()
        except Exception as exc: raise RuntimeError("teacher receipt unavailable") from exc
        if current!=receipt_hash: raise RuntimeError("teacher receipt changed during calibration")
    deadline=time.monotonic()+30
    while True:
        receipt_current()
        _verify_observed_wav(observed,expected_wav,expected_pcm,expected_samples)
        with scheduler.evaluator_lease() as granted:
            if granted:
                receipt_current()
                _verify_observed_wav(observed,expected_wav,expected_pcm,expected_samples)
                try: transcript=runner.transcribe(teacher,observed,item_hash)
                except Exception as exc: raise RuntimeError("teacher transcription failed") from exc
                _verify_observed_wav(observed,expected_wav,expected_pcm,expected_samples)
                receipt_current()
                return transcript
        if time.monotonic()>=deadline: raise RuntimeError("scheduler_timeout")
        time.sleep(.1)


def _collect_observed_answers(runner: Any, scheduler: InferenceScheduler, source: Path,
                              item_hash: str, receipt_hash: str,
                              spec: dict[str, Any]) -> dict[str, str | None]:
    """Collect both teacher answers from one private immutable WAV copy."""
    if not isinstance(spec,dict): raise RuntimeError("invalid immutable calibration identity")
    try:
        expected_wav=spec["audio_sha256"]; expected_pcm=spec["pcm_sha256"]; expected_samples=spec["sample_count"]
    except (KeyError,TypeError) as exc:
        raise RuntimeError("invalid immutable calibration identity") from exc
    answers: dict[str,str|None]={}
    with _immutable_observed_wav(source,expected_wav,expected_pcm,expected_samples) as observed:
        for teacher in REQUIRED_TEACHERS:
            answers[teacher.stable_id]=_transcribe_observed_teacher(runner,scheduler,teacher,observed,item_hash,receipt_hash,spec)
            _verify_observed_wav(observed,expected_wav,expected_pcm,expected_samples)
    return answers


@contextlib.contextmanager
def _immutable_observed_wav(source: Path, expected_wav_sha: str, expected_pcm_sha: str,
                            expected_samples: int) -> Iterator[Path]:
    """Stage one verified WAV outside the manifest tree for one observation."""
    hex64=lambda value: isinstance(value,str) and len(value)==64 and all(char in "0123456789abcdef" for char in value)
    if not hex64(expected_wav_sha) or not hex64(expected_pcm_sha) or isinstance(expected_samples,bool) or not isinstance(expected_samples,int) or expected_samples <= 0:
        raise RuntimeError("invalid immutable calibration identity")
    source_fd: int | None=None; held_fd: int | None=None; root: Path | None=None
    try:
        try:
            before=os.lstat(source)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise RuntimeError("unsafe calibration audio")
            source_fd=os.open(source,os.O_RDONLY | getattr(os,"O_NOFOLLOW",0))
            opened=os.fstat(source_fd)
            if not stat.S_ISREG(opened.st_mode) or (before.st_dev,before.st_ino)!=(opened.st_dev,opened.st_ino):
                raise RuntimeError("unsafe calibration audio")
        except OSError as exc:
            raise RuntimeError("unsafe calibration audio") from exc
        root=Path(tempfile.mkdtemp(prefix="sotto-calibration-observed-"))
        os.chmod(root,0o700)
        destination=root/"observed.wav"
        output=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        digest=hashlib.sha256()
        try:
            while True:
                block=os.read(source_fd,1 << 20)
                if not block: break
                digest.update(block)
                view=memoryview(block)
                while view:
                    written=os.write(output,view)
                    if written <= 0: raise RuntimeError("immutable calibration copy failed")
                    view=view[written:]
            os.fsync(output)
        finally:
            os.close(output); output=None
            os.close(source_fd); source_fd=None
        if digest.hexdigest()!=expected_wav_sha:
            raise RuntimeError("calibration audio changed")
        os.chmod(destination,0o400)
        held_fd=os.open(destination,os.O_RDONLY | getattr(os,"O_NOFOLLOW",0))
        staged=os.fstat(held_fd)
        if not stat.S_ISREG(staged.st_mode):
            raise RuntimeError("calibration audio changed")
        _verify_observed_wav(destination,expected_wav_sha,expected_pcm_sha,expected_samples)
        yield destination
    finally:
        if source_fd is not None: os.close(source_fd)
        if held_fd is not None: os.close(held_fd)
        if root is not None:
            shutil.rmtree(root,ignore_errors=True)


def _canonical_manifest(value: dict[str, Any]) -> str:
    copy = dict(value); supplied = copy.pop("manifest_hash", None)
    if isinstance(copy.get("entries"),list):
        copy["entries"]=[{k:v for k,v in entry.items() if k != "audio_path"} if isinstance(entry,dict) else entry for entry in copy["entries"]]
    digest = hashlib.sha256(json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if supplied != digest: raise RuntimeError("calibration manifest digest mismatch")
    return digest

def _load_protocol_bundle(manifest_path: Path, value: Any) -> tuple[Path,dict]:
    if not isinstance(value,dict) or set(value) != {"schema","path","digest"} or value.get("schema") != 1 or value.get("path") != "protocol/metadata.json":
        raise RuntimeError("invalid protocol bundle identity")
    digest=value.get("digest")
    if not isinstance(digest,str) or len(digest)!=64 or any(char not in "0123456789abcdef" for char in digest) or digest != _protocol_hash():
        raise RuntimeError("protocol bundle identity changed")
    root=manifest_path.parent.resolve(); directory=root/"protocol"; metadata=directory/"metadata.json"
    if directory.is_symlink() or not directory.is_dir() or stat.S_IMODE(directory.stat().st_mode)!=0o700 or metadata.is_symlink() or not metadata.is_file() or stat.S_IMODE(metadata.stat().st_mode)!=0o600:
        raise RuntimeError("unsafe protocol bundle")
    try: identity=json.loads(metadata.read_text("utf-8"))
    except (OSError,ValueError,json.JSONDecodeError) as exc: raise RuntimeError("invalid protocol bundle") from exc
    if identity != protocol_bundle_identity() or identity.get("digest") != digest:
        raise RuntimeError("protocol bundle authority changed")
    files=identity.get("files")
    if not isinstance(files,list) or [entry.get("path") if isinstance(entry,dict) else None for entry in files] != list(PROTOCOL_FILES):
        raise RuntimeError("invalid protocol bundle files")
    for entry in files:
        relative=Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts: raise RuntimeError("unsafe protocol bundle file")
        raw=directory/relative
        if raw.is_symlink(): raise RuntimeError("unsafe protocol bundle file")
        copied=raw.resolve()
        if directory not in copied.parents or not copied.is_file() or stat.S_IMODE(copied.stat().st_mode)!=0o600:
            raise RuntimeError("unsafe protocol bundle file")
        data=copied.read_bytes()
        if len(data)!=entry.get("size") or hashlib.sha256(data).hexdigest()!=entry.get("sha256"):
            raise RuntimeError("protocol bundle file changed")
    return directory,identity


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if stat.S_ISLNK(os.lstat(path).st_mode) or not stat.S_ISREG(os.lstat(path).st_mode):
        raise RuntimeError("unsafe calibration manifest")
    value = json.loads(path.read_text("utf-8")); digest = _canonical_manifest(value)
    source = value.get("source")
    entries = value.get("entries")
    allowed_top={"schema","policy_hash","protocol_hash","selection_algorithm","source","N","G","split_counts","membership_hash","protocol_bundle","entries","manifest_hash"}
    allowed_source={"repo","revision","language","splits"}
    allowed_entry={"id","cluster_id","split","audio_path","audio_sha256","pcm_sha256","sample_count","reference"}
    if set(value) != allowed_top or not isinstance(source,dict) or set(source) != allowed_source:
        raise RuntimeError("calibration manifest has unknown identity fields")
    if not isinstance(entries,list) or any(not isinstance(item,dict) or set(item) != allowed_entry for item in entries):
        raise RuntimeError("calibration manifest item schema invalid")
    if (value.get("schema") != 2 or value.get("policy_hash") != policy_hash() or not isinstance(source, dict) or
            source.get("repo") != "facebook/voxpopuli" or source.get("revision") != "42f01879c780b4a2e90ec0b4f616c2ece526e4f1" or
            source.get("language") != "en" or source.get("splits") != [{"name":"test","sha256":"02cc7290425ddb95beeabda1d1e81e5e068cd7a25380bcf1fb69071b62610ffa"},{"name":"validation","sha256":"bc8bbe5fe23aba60c3e02e6f6239943bc92abc0aa327c926bd3a2d61e44d646b"}] or
            value.get("selection_algorithm") != "voxpopuli-en-test-validation-round-robin-max2-speaker-v3" or value.get("protocol_hash") != _protocol_hash() or not isinstance(entries, list)):
        raise RuntimeError("calibration manifest identity invalid")
    if value.get("N") != 500 or value.get("G") != 339 or value.get("split_counts") != SPLIT_COUNTS or len(entries) != 500 or len({str(e.get("id")) for e in entries if isinstance(e, dict)}) != len(entries) or len({str(e.get("cluster_id")) for e in entries if isinstance(e, dict)}) != 339:
        raise RuntimeError("calibration requires 149 distinct frozen clusters")
    if not isinstance(value.get("membership_hash"),str): raise RuntimeError("calibration membership identity missing")
    membership=hashlib.sha256(json.dumps([(i,e.get("id"),e.get("cluster_id"),e.get("split"),e.get("audio_sha256"),e.get("pcm_sha256"),e.get("sample_count"),hashlib.sha256(str(e.get("reference","")).encode()).hexdigest()) for i,e in enumerate(entries)],separators=(",",":")).encode()).hexdigest()
    if membership != value["membership_hash"] or membership != MEMBERSHIP_HASH:
        raise RuntimeError("calibration membership mismatch")
    ledger=[]
    for ordinal,entry in enumerate(entries):
        if (not isinstance(entry.get("sample_count"),int) or entry["sample_count"] <= 0 or
                entry.get("split") not in SPLIT_COUNTS or any(not isinstance(entry.get(field),str) or not entry[field]
                for field in ("id","cluster_id","audio_sha256","pcm_sha256","reference"))):
            raise RuntimeError("calibration manifest entry identity invalid")
        item_hash=hashlib.sha256(json.dumps({"id":entry["id"],"cluster":entry["cluster_id"],"split":entry["split"],
            "pcm":entry["pcm_sha256"],"samples":entry["sample_count"],"reference":hashlib.sha256(entry["reference"].encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()).hexdigest()
        ledger.append((ordinal,item_hash,hashlib.sha256(entry["cluster_id"].encode()).hexdigest()))
    if hashlib.sha256(json.dumps(ledger,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() != LEDGER_HASH:
        raise RuntimeError("calibration compiled ledger mismatch")
    _load_protocol_bundle(path,value.get("protocol_bundle"))
    return value, digest


def validate_item(item: dict[str, Any], *, manifest_dir: Path | None = None) -> tuple[Path, str, str, str]:
    required = ("id", "cluster_id", "audio_path", "audio_sha256", "pcm_sha256", "reference")
    if any(not isinstance(item.get(k), str) or not item[k] for k in required): raise RuntimeError("invalid calibration item")
    raw=Path(item["audio_path"])
    if manifest_dir is None: manifest_dir=Path.cwd()
    if raw.is_absolute() or raw.parts != ("audio",raw.name) or len(raw.name) != 68 or not raw.name.endswith(".wav") or any(char not in "0123456789abcdef" for char in raw.name[:-4]):
        raise RuntimeError("unsafe calibration audio path")
    root=manifest_dir.resolve(); audio_dir=root / "audio"; path=(root/raw).resolve()
    if audio_dir.is_symlink() or path.parent != audio_dir.resolve() or path.is_symlink() or _hash_file(path) != item["audio_sha256"]: raise RuntimeError("calibration audio digest mismatch")
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16_000 or wav.getcomptype() != "NONE":
            raise RuntimeError("calibration canonical wav required")
    _samples,identity=read_canonical_wav(path)
    if item.get("pcm_sha256") != identity.sha256 or item.get("sample_count") != identity.sample_count:
        raise RuntimeError("calibration canonical PCM identity mismatch")
    # Do not collapse Unicode differences to an ASCII prefix.  The consensus
    # normalizer will abstain on normalization loss; the frozen reference is
    # otherwise compared exactly under the same NFKC/whitespace form.
    import unicodedata
    reference=" ".join(unicodedata.normalize("NFKC",item["reference"]).split())
    if not reference: raise RuntimeError("empty frozen calibration reference")
    # The durable expected-item key commits every semantic member field, not
    # merely a caller-chosen utterance ID.  This prevents a forged manifest
    # from reusing a favorable id/cluster with different audio or gold.
    item_hash=hashlib.sha256(json.dumps({"id":item["id"],"cluster":item["cluster_id"],"split":item["split"],
        "pcm":item["pcm_sha256"],"samples":item["sample_count"],"reference":hashlib.sha256(reference.encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return path,item_hash,hashlib.sha256(item["cluster_id"].encode()).hexdigest(),reference


def run(base_dir: Path, manifest_path: Path, *, test_parquet: Path, validation_parquet: Path, pyarrow_python: Path | str | None = None) -> dict[str, Any]:
    manifest, manifest_hash = load_manifest(manifest_path)
    derived=derive_plan_with_reader(test_parquet,validation_parquet,pyarrow_python)
    projected=[(audio_id if len(audio_id)==64 else hashlib.sha256(audio_id.encode()).hexdigest(),speaker if len(speaker)==64 else hashlib.sha256(speaker.encode()).hexdigest(),split,hashlib.sha256(pcm).hexdigest(),len(pcm)//2,reference) for audio_id,speaker,split,pcm,reference in derived]
    supplied=[(str(item.get("id")),str(item.get("cluster_id")),str(item.get("split")),str(item.get("pcm_sha256")),item.get("sample_count"),str(item.get("reference"))) for item in manifest["entries"]]
    if projected != supplied: raise RuntimeError("calibration source-derived plan mismatch")
    store = SilverStore(base_dir)
    # The public one-shot tombstone covers the canonical two-artifact source,
    # independently of manifest location, membership, receipt, or policy.
    # Source fields were already matched literally above, so no caller JSON
    # alias can mint a second overlapping look.
    source_hash=SOURCE_HASH
    holdout_key=SOURCE_HASH
    # Validate every immutable item before touching a receipt or teacher.
    entries: dict[str, tuple[Path,str,str,str]]={}
    entry_specs: dict[str,dict[str,Any]]={}
    expected=[]
    for item in manifest["entries"]:
        path,item_hash,cluster_hash,reference=validate_item(item,manifest_dir=manifest_path.parent)
        if item_hash in entries: raise RuntimeError("duplicate calibration item")
        entries[item_hash]=(path,item_hash,cluster_hash,reference); entry_specs[item_hash]=item
        expected.append({"item_hash":item_hash,"cluster_hash":cluster_hash})
    if len(expected) != 500 or len({value["cluster_hash"] for value in expected}) != 339:
        raise RuntimeError("calibration immutable plan mismatch")
    runner = OfflineTeacherRunner(load_receipts(base_dir)); receipt_hash = runner.validate()
    store.begin_calibration_v2(holdout_key=holdout_key,manifest_hash=manifest_hash,source_hash=source_hash,
                               protocol_hash=str(manifest["protocol_hash"]),policy_hash=manifest["policy_hash"],
                               receipt_hash=receipt_hash,expected=expected)
    scheduler = InferenceScheduler(base_dir)
    while True:
        claimed=store.claim_calibration_v2(holdout_key,scheduler.owner_id)
        if claimed is None: break
        path,item_hash,cluster_hash,reference=entries[claimed["item_hash"]]
        try:
            if runner.validate() != receipt_hash:
                raise RuntimeError("teacher receipt changed during calibration")
            checked_path,checked_hash,checked_cluster,checked_reference=validate_item(entry_specs[item_hash],manifest_dir=manifest_path.parent)
            if (checked_path,checked_hash,checked_cluster,checked_reference) != (path,item_hash,cluster_hash,reference):
                raise RuntimeError("calibration PCM changed before inference")
            answers=_collect_observed_answers(runner,scheduler,path,item_hash,receipt_hash,entry_specs[item_hash])
            checked_path,checked_hash,checked_cluster,checked_reference=validate_item(entry_specs[item_hash],manifest_dir=manifest_path.parent)
            if (checked_path,checked_hash,checked_cluster,checked_reference) != (path,item_hash,cluster_hash,reference):
                raise RuntimeError("calibration PCM changed during inference")
            if runner.validate() != receipt_hash:
                raise RuntimeError("teacher receipt changed during calibration")
            decision = exact_unanimous(answers, [t.stable_id for t in REQUIRED_TEACHERS])
            if runner.validate() != receipt_hash:
                raise RuntimeError("teacher receipt changed during calibration")
            match = decision.state == "accepted" and decision.reference == reference
            if not store.finish_calibration_v2(claimed,outcome=decision.state,reference_match=match if decision.state == "accepted" else None,code=decision.reason):
                raise RuntimeError("calibration terminal lease lost")
        except (RuntimeError, OSError, TimeoutError, __import__("subprocess").TimeoutExpired) as exc:
            # A split is one-shot evidence, not a way to turn a broken
            # teacher/receipt/adapter into conveniently low coverage.  Leave
            # this item absent so the exact frozen run resumes later.  Release
            # the owned lease now rather than stranding it until expiry.
            # Persist a fixed infrastructure family only; subclass names can
            # be caller/environment-specific and are not part of the
            # content-free durable protocol.
            if isinstance(exc, __import__("subprocess").TimeoutExpired): code="TimeoutExpired"
            elif isinstance(exc, TimeoutError): code="TimeoutError"
            elif isinstance(exc, OSError): code="OSError"
            else: code="RuntimeError"
            store.release_calibration_v2(claimed,code=code)
            raise
    # A receipt/code change invalidates a partial run rather than mixing arms.
    if runner.validate() != receipt_hash: raise RuntimeError("teacher receipt changed during calibration")
    result=store.finalize_calibration_v2(holdout_key)
    if result is None: raise RuntimeError("calibration incomplete")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run frozen offline Sotto public calibration")
    p.add_argument("--base-dir", required=True); p.add_argument("--manifest", required=True); p.add_argument("--test-parquet",required=True); p.add_argument("--validation-parquet",required=True); p.add_argument("--pyarrow-python",required=True)
    args = p.parse_args(argv)
    # One content-free machine result for automation.  Neither reference text
    # nor model output is ever emitted, including on a rejected calibration.
    try:
        result=run(Path(args.base_dir), Path(args.manifest),test_parquet=Path(args.test_parquet),validation_parquet=Path(args.validation_parquet),pyarrow_python=Path(args.pyarrow_python))
        if not isinstance(result,dict) or not isinstance(result.get("state"),str):
            raise RuntimeError("calibration result malformed")
        output={key:result[key] for key in ("state","accepted","clusters","reference_errors","total") if key in result}
    except Exception as exc:
        output={"state":"blocked","code":type(exc).__name__}
    print(json.dumps(output,sort_keys=True,separators=(",",":")))
    return 0 if output.get("state") == "passed" else 1


if __name__ == "__main__": raise SystemExit(main())
