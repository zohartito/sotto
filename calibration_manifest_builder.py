"""Deterministically extract a private frozen VoxPopuli calibration manifest."""
from __future__ import annotations

import argparse, hashlib, json, os, stat, struct, subprocess, sys, uuid, wave, unicodedata, tempfile, shutil, io
from pathlib import Path
from typing import Any

from storage_lock import ensure_private_directory, advisory_lock
from audio_codec import encode_canonical, write_canonical_wav
from teacher_consensus import normalized_tokens, normalization_preserves_words, policy_hash

VOX_REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
TEST_SHA256 = "02cc7290425ddb95beeabda1d1e81e5e068cd7a25380bcf1fb69071b62610ffa"
VALIDATION_SHA256 = "bc8bbe5fe23aba60c3e02e6f6239943bc92abc0aa327c926bd3a2d61e44d646b"
SELECTION = "voxpopuli-en-test-validation-round-robin-max2-speaker-v3"
N = 500
G = 339
SOURCE_HASH = "c39879f752138ca96a782376bb8c5491f2026bd7783cc85dd8f27f6914e4a1c0"
MEMBERSHIP_HASH = "889de30c36dbce19e2c8a9679cd78bd4eb7c03bc9497cc06930641a4f26be8ef"
LEDGER_HASH = "7e9a3d8e83052cbb4290df47d2f76646500aec3e0256bebf93240a41010dfa3a"
SPLIT_COUNTS = {"test": 401, "validation": 99}
READER_IDENTITY = {"python":"3.11.7","unicode":"14.0.0","pyarrow":"23.0.1","numpy":"2.0.2"}
PROTOCOL_FILES=("calibration_manifest_builder.py","calibration_worker.py","teacher_consensus.py","audio_codec.py","teacher_backends.py","silver_store.py")

def reader_runtime_identity() -> dict[str,str]:
    """Content-free identity for the pinned parquet-reader semantics."""
    try:
        import pyarrow, numpy
        arrow=pyarrow.__version__; numpy_version=numpy.__version__
    except ImportError as exc:
        raise RuntimeError("pinned parquet reader requires pyarrow and numpy") from exc
    return {"python":sys.version.split()[0],"unicode":unicodedata.unidata_version,
            "pyarrow":arrow,"numpy":numpy_version}

def _sha(path: Path) -> str:
    if stat.S_ISLNK(os.lstat(path).st_mode) or not stat.S_ISREG(os.lstat(path).st_mode): raise RuntimeError("unsafe source")
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def _canonicalize_wav(data: bytes) -> bytes | None:
    """Decode only RIFF mono 16k PCM16/IEEE-float and canonicalize locally.

    VoxPopuli's parquet stores format-3 float WAVs, which :mod:`wave` rejects.
    This intentionally avoids ffmpeg/resampling variation; clipping and
    float->PCM16 conversion are exactly Sotto's canonical encoder.
    """
    import numpy as np
    if len(data)<44 or data[:4]!=b"RIFF" or data[8:12]!=b"WAVE": return None
    offset=12; fmt=None; payload=None
    while offset+8<=len(data):
        name=data[offset:offset+4]; size=struct.unpack_from("<I",data,offset+4)[0]; body=offset+8
        if body+size>len(data): return None
        if name==b"fmt ": fmt=data[body:body+size]
        elif name==b"data": payload=data[body:body+size]
        offset=body+size+(size&1)
    if fmt is None or payload is None or len(fmt)<16: return None
    form,channels,rate,_byte_rate,align,bits=struct.unpack_from("<HHIIHH",fmt)
    if channels!=1 or rate!=16000 or (form,bits) not in {(1,16),(3,32)}: return None
    if form==1:
        if len(payload)%2:return None
        samples=np.frombuffer(payload,dtype="<i2").astype(np.float32)/32768.0
    else:
        if len(payload)%4:return None
        samples=np.frombuffer(payload,dtype="<f4")
        if not np.isfinite(samples).all():return None
    if not .5<=len(samples)/16000<=30:return None
    return encode_canonical(samples)[0]

def _canonical_wav_bytes(pcm: bytes) -> bytes:
    """Exact private WAV serialization used for frozen output identities."""
    if len(pcm) % 2: raise ValueError("invalid PCM16")
    target=io.BytesIO()
    with wave.open(target,"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000); handle.setcomptype("NONE","not compressed"); handle.writeframes(pcm)
    return target.getvalue()

def _canonical_wav_sha256(pcm: bytes) -> str:
    return hashlib.sha256(_canonical_wav_bytes(pcm)).hexdigest()

def _new_output_stage(output: Path) -> Path:
    output=output.resolve()
    if output.exists() or output.is_symlink(): raise RuntimeError("calibration output already exists")
    parent=output.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir(): raise RuntimeError("unsafe calibration output parent")
    stage=parent / ("."+output.name+"."+uuid.uuid4().hex+".tmp")
    stage.mkdir(mode=0o700,exist_ok=False); os.chmod(stage,0o700)
    return stage

def _publish_output_stage(stage: Path, output: Path) -> None:
    output=output.resolve(); stage=stage.resolve(); parent=output.parent
    if stage.parent != parent or stage.is_symlink() or not stage.is_dir() or output.exists() or output.is_symlink(): raise RuntimeError("unsafe calibration publication")
    fd=os.open(stage,os.O_RDONLY); os.fsync(fd); os.close(fd)
    with advisory_lock(parent):
        if stage.parent != parent or stage.is_symlink() or not stage.is_dir() or output.exists() or output.is_symlink(): raise RuntimeError("unsafe calibration publication")
        os.rename(stage,output)
        fd=os.open(parent,os.O_RDONLY); os.fsync(fd); os.close(fd)

def _bytes(row: Any) -> bytes | None:
    audio=row.get("audio") if isinstance(row,dict) else None
    if isinstance(audio,dict): audio=audio.get("bytes")
    return bytes(audio) if isinstance(audio,(bytes,bytearray,memoryview)) else None

def _protocol_hash() -> str:
    # V2 calibration is tied to the code that chooses rows, canonicalizes
    # audio, and decides consensus—not merely a label string.
    return protocol_bundle_identity()["digest"]

def protocol_bundle_identity() -> dict[str,Any]:
    root=Path(__file__).parent; seen=set(); entries=[]
    for name in PROTOCOL_FILES:
        path=Path(name)
        if path.is_absolute() or ".." in path.parts or name in seen: raise RuntimeError("unsafe protocol source")
        seen.add(name); target=root/path
        mode=os.lstat(target).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise RuntimeError("unsafe protocol source")
        data=target.read_bytes(); entries.append({"path":name,"sha256":hashlib.sha256(data).hexdigest(),"size":len(data)})
    try:
        import numpy
        numpy_version=numpy.__version__
    except ImportError: numpy_version="missing"
    body={"schema":1,"selection":SELECTION,"policy_hash":policy_hash(),"audio":"canonical-mono16k-v2","reader":READER_IDENTITY,
          "app":{"python":sys.version.split()[0],"unicode":unicodedata.unidata_version,"numpy":numpy_version},"files":entries}
    body["digest"]=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    return body

def _write_protocol_bundle(bundle_dir: Path, identity: dict | None = None) -> dict:
    identity=identity or protocol_bundle_identity(); parent=bundle_dir.parent
    if bundle_dir.exists() or bundle_dir.is_symlink() or not parent.is_dir() or parent.is_symlink(): raise RuntimeError("unsafe protocol bundle target")
    expected=protocol_bundle_identity()
    if identity != expected or not isinstance(identity.get("files"),list) or [entry.get("path") for entry in identity["files"]] != list(PROTOCOL_FILES):
        raise RuntimeError("protocol bundle identity mismatch")
    try:
        bundle_dir.mkdir(mode=0o700); os.chmod(bundle_dir,0o700)
        root=Path(__file__).parent
        for entry in identity["files"]:
            name=entry["path"]; relative=Path(name)
            if relative.is_absolute() or ".." in relative.parts: raise RuntimeError("unsafe protocol source")
            source=root/relative; mode=os.lstat(source).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise RuntimeError("unsafe protocol source")
            data=source.read_bytes()
            if len(data)!=entry["size"] or hashlib.sha256(data).hexdigest()!=entry["sha256"]: raise RuntimeError("protocol source changed")
            target=bundle_dir/relative; target.parent.mkdir(mode=0o700,parents=True,exist_ok=True); os.chmod(target.parent,0o700)
            fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,"wb") as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
        raw=json.dumps(identity,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode(); fd=os.open(bundle_dir/"metadata.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        directory_fd=os.open(bundle_dir,os.O_RDONLY); os.fsync(directory_fd); os.close(directory_fd)
        return identity
    except Exception:
        shutil.rmtree(bundle_dir,ignore_errors=True); raise

def _eligible(rows: list[dict[str, Any]], split: str) -> list[tuple[str,str,str,bytes,str]]:
    values=[]
    for row in rows:
        # Membership is technical only.  In particular, do not filter risky
        # gold: unsafe teacher omissions must remain observable outcomes.
        if not row.get("is_gold_transcript",False): continue
        # ``normalized_text`` in the converted parquet was produced by an
        # ASCII-biased normalizer.  Prefer the dataset's raw Unicode gold and
        # use normalized_text only when raw gold is genuinely absent.
        ref=row.get("raw_text") or row.get("normalized_text"); speaker=row.get("speaker_id"); audio_id=row.get("audio_id")
        data=_bytes(row); canonical=_canonicalize_wav(data) if data is not None else None
        if not all(isinstance(x,str) and x for x in (ref,speaker,audio_id)) or speaker.casefold() in {"none","unknown"} or canonical is None: continue
        # Preserve the canonical Unicode reference verbatim (apart from
        # deterministic Unicode/whitespace normalization).  The old ASCII
        # tokenizer could turn ``hello שלום`` into ``hello``, allowing both
        # teachers to omit the risky word and still pass reference matching.
        unicode_ref=" ".join(unicodedata.normalize("NFKC",ref).split())
        # Match the consensus canonicalizer for ordinary ASCII, while keeping
        # meaningful Unicode intact so an ASCII-only teacher omission is a
        # visible accepted-reference mismatch rather than a hidden agreement.
        reference=(" ".join(normalized_tokens(unicode_ref)) if normalization_preserves_words(unicode_ref) else unicode_ref)
        if not reference or len(reference.split())>80: continue
        values.append((str(audio_id),str(speaker),split,canonical,reference))
    return values

def derive_plan(test_parquet: Path, validation_parquet: Path) -> list[tuple[str,str,str,bytes,str]]:
    """Pure, path-independent frozen V2 membership derivation."""
    if _sha(test_parquet)!=TEST_SHA256 or _sha(validation_parquet)!=VALIDATION_SHA256:
        raise RuntimeError("pinned VoxPopuli source mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc: raise RuntimeError("run with an isolated pyarrow interpreter") from exc
    # Stream row groups instead of materializing two ~0.9GB audio Parquets in
    # one table.  Selection remains deterministic because every record is
    # subsequently ranked by its domain-separated hashes.
    # Keep only the two deterministic clip candidates needed for each speaker.
    # The input Parquets are multi-gigabyte once decoded; retaining every
    # canonical PCM buffer defeats row-group streaming and is unnecessary for
    # the predeclared max-two selection.
    by_speaker: dict[str,list[tuple[str,str,str,bytes,str]]]={}
    seen: dict[str,tuple[str,str,str]]={}
    canonical_seen: dict[str,str]={}
    rank=lambda domain, value: hashlib.sha256((domain+"\0"+value).encode()).hexdigest()
    for source,split in ((test_parquet,"test"),(validation_parquet,"validation")):
        reader=pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=32):
            for value in _eligible(batch.to_pylist(),split):
                audio_digest=hashlib.sha256(value[3]).hexdigest()
                old=seen.get(value[0])
                if old and old != (value[1],value[4],audio_digest): raise RuntimeError("conflicting duplicate utterance")
                if old: continue
                seen[value[0]]=(value[1],value[4],audio_digest)
                if audio_digest in canonical_seen and canonical_seen[audio_digest] != value[0]: raise RuntimeError("duplicate canonical audio")
                canonical_seen[audio_digest]=value[0]
                candidates=by_speaker.setdefault(value[1],[])
                candidates.append(value)
                candidates.sort(key=lambda x:(rank("clip-v1",x[0]),x[0]))
                del candidates[2:]
    if len(by_speaker) != G: raise RuntimeError("unexpected frozen speaker universe")
    speakers=sorted(by_speaker,key=lambda speaker:(rank("speaker-v1",speaker),speaker))
    records=[by_speaker[speaker][0] for speaker in speakers]
    for speaker in speakers:
        if len(records)>=N: break
        if len(by_speaker[speaker])>1: records.append(by_speaker[speaker][1])
    if len(records)!=N: raise RuntimeError("insufficient pinned round-robin eligibility")
    return records

def _write_reader_bundle(test_parquet: Path, validation_parquet: Path, bundle: Path) -> None:
    """Helper-only: parse pinned parquet and emit a private PCM plan bundle."""
    if bundle.exists() or bundle.is_symlink(): raise RuntimeError("unsafe reader bundle target")
    bundle.mkdir(mode=0o700,parents=False); os.chmod(bundle,0o700)
    entries=[]
    for ordinal,(audio_id,speaker,split,pcm,reference) in enumerate(derive_plan(test_parquet,validation_parquet)):
        name=f"{ordinal:03d}.pcm"; target=bundle/name
        fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as handle: handle.write(pcm); handle.flush(); os.fsync(handle.fileno())
        entries.append({"ordinal":ordinal,"id":hashlib.sha256(audio_id.encode()).hexdigest(),"cluster":hashlib.sha256(speaker.encode()).hexdigest(),"split":split,
                        "pcm":hashlib.sha256(pcm).hexdigest(),"samples":len(pcm)//2,"reference":reference,"file":name})
    payload={"schema":1,"runtime":reader_runtime_identity(),"entries":entries}
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    payload["digest"]=hashlib.sha256(raw).hexdigest()
    target=bundle/"plan.json"; target.write_text(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False),"utf-8"); os.chmod(target,0o600)

def derive_plan_with_reader(test_parquet: Path, validation_parquet: Path, reader_python: Path | str | None) -> list[tuple[str,str,str,bytes,str]]:
    """Use only a pinned parser subprocess; app runtime owns all authority."""
    if reader_python is None:
        if reader_runtime_identity() != READER_IDENTITY:
            raise RuntimeError("inline parquet reader identity is not pinned")
        return derive_plan(test_parquet,validation_parquet)
    # The parent validates the same pinned input bytes independently.  The
    # helper path is deliberately excluded from source identity.
    if _sha(test_parquet) != TEST_SHA256 or _sha(validation_parquet) != VALIDATION_SHA256:
        raise RuntimeError("pinned VoxPopuli source mismatch")
    reader=Path(reader_python).resolve()
    if not reader.is_absolute() or reader.is_symlink() or not reader.is_file() or not os.access(reader,os.X_OK): raise RuntimeError("unsafe parquet reader interpreter")
    root=Path(tempfile.mkdtemp(prefix="sotto-calibration-reader-")); os.chmod(root,0o700); bundle=root/"bundle"
    try:
        command=[str(reader),str(Path(__file__).resolve()),"--reader-helper","--test-parquet",str(test_parquet),"--validation-parquet",str(validation_parquet),"--bundle",str(bundle)]
        done=subprocess.run(command,shell=False,check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120)
        if done.returncode != 0: raise RuntimeError("pinned parquet reader failed")
        plan=bundle/"plan.json"
        if plan.is_symlink() or not plan.is_file(): raise RuntimeError("reader plan missing")
        value=json.loads(plan.read_text("utf-8")); digest=value.pop("digest",None)
        if set(value) != {"schema","runtime","entries"} or value.get("schema") != 1 or value.get("runtime") != READER_IDENTITY or not isinstance(value.get("entries"),list) or digest != hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest():
            raise RuntimeError("reader bundle identity invalid")
        records=[]
        hex64=lambda item: isinstance(item,str) and len(item)==64 and all(char in "0123456789abcdef" for char in item)
        for index,item in enumerate(value["entries"]):
            if not isinstance(item,dict) or item.get("ordinal") != index or set(item) != {"ordinal","id","cluster","split","pcm","samples","reference","file"}: raise RuntimeError("reader entry invalid")
            if (not hex64(item.get("id")) or not hex64(item.get("cluster")) or not hex64(item.get("pcm")) or
                    item.get("split") not in SPLIT_COUNTS or not isinstance(item.get("samples"),int) or item["samples"] <= 0 or
                    not isinstance(item.get("reference"),str) or not item["reference"] or item.get("file") != f"{index:03d}.pcm"):
                raise RuntimeError("reader entry identity invalid")
            raw=bundle/str(item["file"])
            if raw.is_symlink() or not raw.is_file() or raw.parent != bundle: raise RuntimeError("reader PCM unsafe")
            pcm=raw.read_bytes()
            if hashlib.sha256(pcm).hexdigest()!=item["pcm"] or len(pcm)//2!=item["samples"]: raise RuntimeError("reader PCM identity invalid")
            records.append((str(item["id"]),str(item["cluster"]),str(item["split"]),pcm,str(item["reference"])))
        # IDs in the bundle are already semantic hashes; callers only use this
        # projection for equality, never as raw source identifiers.
        if (len(records) != N or len({row[1] for row in records}) != G or
                {split:sum(row[2] == split for row in records) for split in SPLIT_COUNTS} != SPLIT_COUNTS or
                any(sum(row[1] == item[1] for row in records)>2 for item in records)):
            raise RuntimeError("reader projection shape changed")
        return records
    finally:
        shutil.rmtree(root,ignore_errors=True)

def build(test_parquet: Path, validation_parquet: Path, output: Path, *, pyarrow_python: Path | str | None = None) -> Path:
    output=output.resolve()
    records=derive_plan_with_reader(test_parquet,validation_parquet,pyarrow_python)
    entries=[]
    for audio_id, speaker, split, data, reference in records:
        ident=audio_id if len(audio_id)==64 else hashlib.sha256(audio_id.encode()).hexdigest()
        if len(ident) != 64 or any(char not in "0123456789abcdef" for char in ident):
            raise RuntimeError("unsafe calibration item identity")
        wav_sha=_canonical_wav_sha256(data)
        entries.append({"id":ident,"cluster_id":speaker if len(speaker)==64 else hashlib.sha256(speaker.encode()).hexdigest(),"split":split,"audio_path":"audio/"+ident+".wav","audio_sha256":wav_sha,
                        "pcm_sha256":hashlib.sha256(data).hexdigest(),"sample_count":len(data)//2,"reference":reference})
    source={"repo":"facebook/voxpopuli","revision":VOX_REVISION,"language":"en","splits":[{"name":"test","sha256":TEST_SHA256},{"name":"validation","sha256":VALIDATION_SHA256}]}
    membership_hash=hashlib.sha256(json.dumps([(i,e["id"],e["cluster_id"],e["split"],e["audio_sha256"],e["pcm_sha256"],e["sample_count"],hashlib.sha256(e["reference"].encode()).hexdigest()) for i,e in enumerate(entries)],separators=(",",":")).encode()).hexdigest()
    counts={split:sum(e["split"]==split for e in entries) for split in ("test","validation")}
    if counts != SPLIT_COUNTS or membership_hash != MEMBERSHIP_HASH:
        raise RuntimeError("compiled calibration projection changed")
    ledger=[]
    for ordinal,entry in enumerate(entries):
        item_hash=hashlib.sha256(json.dumps({"id":entry["id"],"cluster":entry["cluster_id"],"split":entry["split"],
            "pcm":entry["pcm_sha256"],"samples":entry["sample_count"],"reference":hashlib.sha256(entry["reference"].encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()).hexdigest()
        ledger.append((ordinal,item_hash,hashlib.sha256(entry["cluster_id"].encode()).hexdigest()))
    if hashlib.sha256(json.dumps(ledger,separators=(",",":"),ensure_ascii=False).encode()).hexdigest() != LEDGER_HASH:
        raise RuntimeError("compiled calibration ledger changed")
    bundle_identity=protocol_bundle_identity()
    stage=_new_output_stage(output); audio_dir=stage/"audio"
    try:
        ensure_private_directory(audio_dir)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    for entry,record in zip(entries,records):
        target=stage/entry["audio_path"]
        if target.parent != audio_dir: raise RuntimeError("unsafe calibration output path")
        try:
            write_canonical_wav(target,record[3])
            if _sha(target) != entry["audio_sha256"]: raise RuntimeError("canonical WAV write mismatch")
        except BaseException:
            shutil.rmtree(stage,ignore_errors=True)
            raise
    try:
        _write_protocol_bundle(stage/"protocol",bundle_identity)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    body={"schema":2,"policy_hash":policy_hash(),"protocol_hash":bundle_identity["digest"],"selection_algorithm":SELECTION,
          "source":source,"N":N,"G":G,"split_counts":counts,"membership_hash":membership_hash,"protocol_bundle":{"schema":1,"path":"protocol/metadata.json","digest":bundle_identity["digest"]},"entries":entries}
    semantic=dict(body); semantic["entries"]=[{k:v for k,v in entry.items() if k != "audio_path"} for entry in entries]
    body["manifest_hash"]=hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    target=stage/"manifest.json"; temp=target.with_name(".manifest."+uuid.uuid4().hex+".tmp")
    try:
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(body,f,sort_keys=True,separators=(",",":")); f.flush(); os.fsync(f.fileno())
        os.chmod(temp,0o600);os.replace(temp,target);os.chmod(target,0o600)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    try:
        _publish_output_stage(stage,output)
    except BaseException:
        if stage.exists() and stage.parent == output.parent:
            shutil.rmtree(stage,ignore_errors=True)
        raise
    return output/"manifest.json"

def main(argv=None)->int:
    p=argparse.ArgumentParser();p.add_argument("--test-parquet",required=True);p.add_argument("--validation-parquet",required=True);p.add_argument("--output");p.add_argument("--pyarrow-python");p.add_argument("--reader-helper",action="store_true");p.add_argument("--bundle")
    a=p.parse_args(argv)
    if a.reader_helper:
        if not a.bundle: raise SystemExit("--bundle required")
        _write_reader_bundle(Path(a.test_parquet),Path(a.validation_parquet),Path(a.bundle)); return 0
    if not a.output: raise SystemExit("--output required")
    build(Path(a.test_parquet),Path(a.validation_parquet),Path(a.output),pyarrow_python=a.pyarrow_python);return 0
if __name__=="__main__":raise SystemExit(main())
