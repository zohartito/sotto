"""Pinned, offline teacher family declarations and safe subprocess adapter."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from offline_runtime import offline_subprocess_env
from storage_lock import ensure_private_directory, ensure_private_file
import teacher_consensus
from teacher_consensus import policy_hash


@dataclass(frozen=True)
class TeacherFamily:
    stable_id: str
    repo: str
    revision: str
    package: str
    decode: dict[str, Any]
    critical_packages: tuple[str, ...]


QWEN = TeacherFamily("qwen3-asr-1.7b-en", "Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5",
                     "mlx-qwen3-asr==0.3.5", {"language": "English", "decode": "greedy", "context": "", "hotword": False, "max_new_tokens": 4096},
                     ("mlx-qwen3-asr", "mlx", "mlx-metal", "numpy", "huggingface-hub", "regex"))
GRANITE = TeacherFamily("granite-4.0-1b-speech-en", "ibm-granite/granite-4.0-1b-speech", "bd87ab862416353633ea431fe49b1614003623c5",
                        "mlx-audio[stt]==0.4.8", {"prompt": "default_transcription", "temperature": 0, "top_p": 1.0, "top_k": 0, "max_tokens": 4096, "prefill_step_size": 2048},
                        ("mlx-audio", "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "safetensors", "sentencepiece", "miniaudio", "scipy", "numpy", "huggingface-hub"))
REQUIRED_TEACHERS = (QWEN, GRANITE)


class TeacherPreempted(RuntimeError):
    """A foreground request cancelled background teacher work."""


@dataclass(frozen=True)
class TeacherSnapshot:
    """One checked Hugging Face snapshot layout for a pinned teacher family."""
    path: Path
    model_root: Path
    blobs: Path


def receipt_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / "adaptive-learning" / "silver" / "teacher-receipts.json"


def load_receipts(base_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Read the provisioned private receipt file without attempting downloads."""
    path = receipt_path(base_dir)
    ensure_private_directory(path.parent)
    ensure_private_file(path)
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("teacher receipts unavailable") from exc
    if not isinstance(value, dict) or not all(isinstance(v, dict) for v in value.values()):
        raise RuntimeError("teacher receipts malformed")
    return value


def sha256_file(path: Path) -> str:
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError("unsafe receipt file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_directory(path: Path) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise RuntimeError("unsafe teacher snapshot") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError("unsafe teacher snapshot")


def teacher_snapshot(family: TeacherFamily, snapshot: Path) -> TeacherSnapshot:
    """Validate the sole accepted local HF layout for one pinned teacher.

    A snapshot path is receipt authority, not a general local-model path.  It
    therefore has one accepted lexical hierarchy and all cache directories
    involved in that hierarchy must be real directories.  Snapshot symlinks
    are checked when enumerated below, because HF uses them for blob dedupe.
    """
    if not isinstance(snapshot, Path) or not snapshot.is_absolute():
        raise RuntimeError("unsafe teacher snapshot")
    expected_root = "models--" + family.repo.replace("/", "--")
    snapshots = snapshot.parent
    model_root = snapshots.parent
    if snapshot.name != family.revision or snapshots.name != "snapshots" or model_root.name != expected_root:
        raise RuntimeError("teacher snapshot does not match pinned family")
    blobs = model_root / "blobs"
    for directory in (model_root, snapshots, snapshot, blobs):
        _strict_directory(directory)
    return TeacherSnapshot(snapshot, model_root, blobs)


def _snapshot_entries(family: TeacherFamily, snapshot: Path) -> tuple[TeacherSnapshot, list[tuple[Path, Path | None, str | None]]]:
    """Enumerate checked entries without traversing a snapshot-controlled link."""
    checked = teacher_snapshot(family, snapshot)
    blobs_resolved = checked.blobs.resolve(strict=True)
    entries: list[tuple[Path, Path | None, str | None]] = []
    pending = [checked.path]
    while pending:
        directory = pending.pop()
        _strict_directory(directory)
        for path in sorted(directory.iterdir(), key=lambda value: value.name, reverse=True):
            try:
                mode = os.lstat(path).st_mode
            except OSError as exc:
                raise RuntimeError("unsafe teacher snapshot entry") from exc
            if stat.S_ISDIR(mode):
                pending.append(path)
                continue
            if stat.S_ISREG(mode):
                entries.append((path, None, None))
                continue
            if not stat.S_ISLNK(mode):
                raise RuntimeError("unsafe teacher snapshot entry")
            try:
                link = os.readlink(path)
                target = (path.parent / link).resolve(strict=True) if not os.path.isabs(link) else Path(link).resolve(strict=True)
                target.relative_to(blobs_resolved)
                target_info = os.lstat(target)
            except (OSError, ValueError) as exc:
                raise RuntimeError("teacher snapshot link escapes model blobs") from exc
            if stat.S_ISLNK(target_info.st_mode) or not stat.S_ISREG(target_info.st_mode):
                raise RuntimeError("teacher snapshot link escapes model blobs")
            # A resolved target under blobs is not enough: no component below
            # blobs may itself be a symlink that changes later.
            cursor = blobs_resolved
            for component in target.relative_to(blobs_resolved).parts:
                cursor = cursor / component
                try:
                    component_mode = os.lstat(cursor).st_mode
                except OSError as exc:
                    raise RuntimeError("teacher snapshot link escapes model blobs") from exc
                if stat.S_ISLNK(component_mode):
                    raise RuntimeError("teacher snapshot link escapes model blobs")
            entries.append((path, target, link))
    return checked, entries


def snapshot_digest(family: TeacherFamily, snapshot: Path) -> str:
    """Hash a family-pinned snapshot without allowing a cross-model blob link."""
    checked, entries = _snapshot_entries(family, snapshot)
    rows: list[str] = []
    for path, target, link in sorted(entries, key=lambda value: value[0].relative_to(checked.path).as_posix()):
        rel = path.relative_to(checked.path).as_posix()
        if target is None:
            rows.append("F\0" + rel + "\0" + sha256_file(path))
        else:
            rows.append("L\0" + rel + "\0" + str(link) + "\0" + sha256_file(target))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def snapshot_stat_manifest(family: TeacherFamily, snapshot: Path) -> str:
    """Cheap change detector; a change forces the next full content hash."""
    rows = []
    checked, entries = _snapshot_entries(family, snapshot)
    for path, target, link in sorted(entries, key=lambda value: value[0].relative_to(checked.path).as_posix()):
        info = os.lstat(path)
        target_marker = ""
        if target is not None:
            resolved = os.lstat(target)
            target_marker = f"\0{target}\0{resolved.st_dev}\0{resolved.st_ino}\0{resolved.st_size}\0{resolved.st_mtime_ns}\0{resolved.st_ctime_ns}\0{link}"
        rows.append(f"{path.relative_to(checked.path)}\0{info.st_dev}\0{info.st_ino}\0{info.st_size}\0{info.st_mtime_ns}\0{info.st_ctime_ns}{target_marker}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _teacher_env() -> dict[str,str]:
    return {**offline_subprocess_env(),"PYTHONDONTWRITEBYTECODE":"1"}


def _marker(info: os.stat_result) -> list[int]:
    # ``st_dev`` is assigned by macOS at mount time and can change across a
    # reboot for unchanged APFS bytes. Persistent teacher receipts bind the
    # stable metadata below plus exact hashes instead.
    return [info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns]


def _launcher_identity(interpreter: Path) -> dict[str,Any]:
    if not interpreter.is_absolute(): raise RuntimeError("unsafe teacher interpreter")
    cursor=interpreter; chain=[]; seen=set()
    while True:
        if str(cursor) in seen: raise RuntimeError("teacher interpreter link loop")
        seen.add(str(cursor))
        try: info=os.lstat(cursor)
        except OSError as exc: raise RuntimeError("unsafe teacher interpreter") from exc
        if stat.S_ISLNK(info.st_mode):
            target=os.readlink(cursor); chain.append({"path":str(cursor),"target":target,"marker":_marker(info),"link_sha256":hashlib.sha256(target.encode()).hexdigest()})
            cursor=cursor.parent/target if not os.path.isabs(target) else Path(target); continue
        if not stat.S_ISREG(info.st_mode) or not os.access(cursor,os.X_OK): raise RuntimeError("unsafe teacher interpreter")
        return {"configured_path":str(interpreter),"chain":chain,"terminal_path":str(cursor),"terminal_marker":_marker(info),"terminal_sha256":sha256_file(cursor)}


def _closure_probe(interpreter: Path, expected_import_roots: list[dict[str,Any]] | None) -> dict[str,Any]:
    """Isolated child probe: discover, freeze and verify purelib/platlib first.

    The child runs under ``-I -S -B`` from ``/`` with a minimal offline
    environment, so no ``site.main``, ``.pth``, ``sitecustomize``,
    ``usercustomize`` or inherited ``PYTHONPATH`` artifact can execute.  It
    hashes the complete purelib/platlib trees with stdlib-only code and, when a
    prior identity is supplied, compares them to the frozen trees BEFORE
    inserting only those verified roots into ``sys.path``.
    """
    code="""import hashlib,json,os,stat,sys,sysconfig
import importlib.metadata as m
payload=json.load(sys.stdin)
def tree(root):
 info=os.lstat(root)
 if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError('teacher_import_root_unsafe')
 rows=[]
 for directory,dirs,files in os.walk(root,followlinks=False):
  dirs.sort(); files.sort(); base=os.path.relpath(directory,root)
  for name in dirs+files:
   path=os.path.join(directory,name); entry=os.lstat(path); rel=name if base=='.' else base+'/'+name
   if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)): raise RuntimeError('teacher_import_root_unsafe')
   row={'path':rel,'type':'dir' if stat.S_ISDIR(entry.st_mode) else 'file','marker':[entry.st_ino,entry.st_size,entry.st_mtime_ns,entry.st_ctime_ns]}
   if stat.S_ISREG(entry.st_mode):
    d=hashlib.sha256(); f=open(path,'rb')
    for b in iter(lambda:f.read(1048576),b''): d.update(b)
    f.close(); row['sha256']=d.hexdigest()
   rows.append(row)
 return {'root':root,'marker':[info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns],'entries':rows}
roots=[]
venv=os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
if os.path.isfile(os.path.join(venv,'pyvenv.cfg')):
 # Python 3.13 under -S does not apply pyvenv.cfg, so sysconfig would report
 # the base interpreter's site-packages.  Derive the venv root lexically.
 roots.append(os.path.join(venv,'lib','python%d.%d'%sys.version_info[:2],'site-packages'))
else:
 for name in ('purelib','platlib'):
  value=sysconfig.get_path(name)
  if not value or not os.path.isabs(value): raise RuntimeError('teacher_site_root_missing')
  if value not in roots: roots.append(value)
computed=[tree(value) for value in roots]
expected=payload.get('expected_import_roots')
if expected is not None and computed != expected: raise RuntimeError('teacher_import_root_changed')
for value in reversed(roots): sys.path.insert(0,value)
rows=[]
for d in m.distributions():
 n=d.metadata.get('Name')
 if n: rows.append({'name':n,'version':d.version,'root':str(d.locate_file('')),'files':[str(x) for x in (d.files or [])]})
print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'sys_path':sys.path,'import_roots':computed,'distributions':sorted(rows,key=lambda x:x['name'].lower())},sort_keys=True,separators=(',',':')))
"""
    run=subprocess.run([str(interpreter),"-I","-S","-B","-c",code],shell=False,check=False,text=True,capture_output=True,env=_teacher_env(),cwd="/",timeout=120,
                       input=json.dumps({"expected_import_roots":expected_import_roots},sort_keys=True,separators=(",",":")))
    if run.returncode or len(run.stdout)>64*1024*1024: raise RuntimeError("teacher runtime identity unavailable")
    try: value=json.loads(run.stdout)
    except (ValueError,json.JSONDecodeError) as exc: raise RuntimeError("teacher runtime identity malformed") from exc
    if (not isinstance(value,dict) or not all(isinstance(value.get(k),str) for k in ("python","executable","prefix","base_prefix")) or
            not isinstance(value.get("distributions"),list) or not isinstance(value.get("sys_path"),list) or
            not isinstance(value.get("import_roots"),list) or not value["import_roots"]): raise RuntimeError("teacher runtime identity malformed")
    return value


def _closure_identity(probe: dict[str,Any], package_names: tuple[str,...]) -> tuple[dict[str,str],list[dict[str,Any]]]:
    found={}; closure=[]
    for row in probe["distributions"]:
        if not isinstance(row,dict) or not all(isinstance(row.get(k),str) for k in ("name","version","root")) or not isinstance(row.get("files"),list): raise RuntimeError("teacher package closure malformed")
        root=Path(row["root"])
        if not root.is_absolute() or not root.is_dir(): raise RuntimeError("teacher package closure unsafe")
        files=[]
        for rel in row["files"]:
            # Distribution RECORD also lists console scripts outside the
            # interpreter's import root.  The sealed launcher is bound
            # separately; this closure deliberately covers importable files.
            if not isinstance(rel,str) or not rel: raise RuntimeError("teacher package closure unsafe")
            if Path(rel).is_absolute() or ".." in Path(rel).parts: continue
            path=root/rel
            try: info=os.lstat(path)
            except OSError as exc: raise RuntimeError("teacher package closure unavailable") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): raise RuntimeError("teacher package closure unsafe")
            files.append({"path":rel,"marker":_marker(info),"sha256":sha256_file(path)})
        closure.append({"name":row["name"],"version":row["version"],"root":str(root),"files":files}); found[row["name"].lower().replace("_","-")]=row["version"]
    roots={name.lower().replace("_","-") for name in package_names}
    if not roots <= set(found): raise RuntimeError("teacher runtime package missing")
    return {name:found[name.lower().replace("_","-")] for name in package_names},closure


def _import_surface(root: Path, *, with_hash: bool, _seen: set[tuple[int,int]] | None = None) -> dict[str,Any]:
    """Exact no-symlink import root identity; entries catch additions too."""
    try: info=os.lstat(root)
    except FileNotFoundError: return {"root":str(root),"kind":"absent"}
    except OSError as exc: raise RuntimeError("teacher import surface unavailable") from exc
    if stat.S_ISLNK(info.st_mode): raise RuntimeError("teacher import surface unsafe")
    seen=set() if _seen is None else _seen
    key=(info.st_dev,info.st_ino)
    if key in seen: raise RuntimeError("teacher import surface cycle")
    seen.add(key)
    if stat.S_ISREG(info.st_mode):
        return {"root":str(root),"kind":"file","marker":_marker(info),"sha256":sha256_file(root) if with_hash else None}
    if not stat.S_ISDIR(info.st_mode): raise RuntimeError("teacher import surface unsafe")
    entries=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs.sort(); files.sort(); relative=Path(directory).relative_to(root)
        for name in [*dirs,*files]:
            path=Path(directory)/name; rel=(relative/name).as_posix(); entry=os.lstat(path)
            # CPython framework installs expose a non-importable
            # ``site-packages`` compatibility symlink below stdlib while the
            # isolated venv site root is separately bound in sys.path.
            if stat.S_ISLNK(entry.st_mode):
                target=os.readlink(path); resolved=(path.parent/target if not os.path.isabs(target) else Path(target)).resolve(strict=True); target_info=os.lstat(resolved)
                if stat.S_ISLNK(target_info.st_mode) or not (stat.S_ISDIR(target_info.st_mode) or stat.S_ISREG(target_info.st_mode)): raise RuntimeError("teacher import surface unsafe")
                row={"path":rel,"kind":"link","marker":_marker(entry),"target":target,"target_marker":_marker(target_info)}
                if stat.S_ISREG(target_info.st_mode) and with_hash: row["target_sha256"]=sha256_file(resolved)
                if stat.S_ISDIR(target_info.st_mode): row["target_surface"]=_import_surface(resolved,with_hash=with_hash,_seen=seen)
                entries.append(row); continue
            if not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)): raise RuntimeError("teacher import surface unsafe")
            row={"path":rel,"kind":"dir" if stat.S_ISDIR(entry.st_mode) else "file","marker":_marker(entry)}
            if stat.S_ISREG(entry.st_mode) and with_hash: row["sha256"]=sha256_file(path)
            entries.append(row)
    return {"root":str(root),"kind":"dir","marker":_marker(info),"entries":entries}


def _import_surfaces(probe: dict[str,Any]) -> list[dict[str,Any]]:
    paths=probe["sys_path"]
    if not paths or any(not isinstance(value,str) or not Path(value).is_absolute() for value in paths): raise RuntimeError("teacher isolated sys.path malformed")
    return [_import_surface(Path(value),with_hash=True) for value in paths]


def runtime_identity(interpreter: Path, package_names: tuple[str, ...], expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Immutable configured-launcher/runtime/package closure, fully offline.

    With ``expected`` (a previously frozen identity), the isolated child
    rejects a changed purelib/platlib tree before inserting any import root.
    """
    launcher=_launcher_identity(interpreter)
    expected_roots=None
    if expected is not None:
        roots=expected.get("import_roots") if isinstance(expected,dict) else None
        if not isinstance(roots,list) or not roots: raise RuntimeError("teacher runtime identity malformed")
        expected_roots=roots
    probe=_closure_probe(interpreter,expected_roots); packages,closure=_closure_identity(probe,package_names)
    return {"schema":4,"launcher":launcher,"python":probe["python"],"executable":probe["executable"],"prefix":probe["prefix"],"base_prefix":probe["base_prefix"],"sys_path":probe["sys_path"],"import_roots":probe["import_roots"],"packages":packages,"closure":closure,"import_surface":_import_surfaces(probe)}


def _runtime_markers_current(identity: object) -> bool:
    """Cheap fail-closed marker check for a previously full-hashed closure."""
    try:
        if not isinstance(identity,dict) or identity.get("schema") != 4: return False
        launcher=identity["launcher"]
        if not isinstance(launcher,dict): return False
        for link in launcher["chain"]:
            info=os.lstat(link["path"])
            if _marker(info) != link["marker"] or not stat.S_ISLNK(info.st_mode) or hashlib.sha256(os.readlink(link["path"]).encode()).hexdigest() != link["link_sha256"]: return False
        terminal=os.lstat(launcher["terminal_path"])
        if not stat.S_ISREG(terminal.st_mode) or _marker(terminal) != launcher["terminal_marker"]: return False
        for distribution in identity["closure"]:
            root=Path(distribution["root"])
            for entry in distribution["files"]:
                info=os.lstat(root/entry["path"])
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or _marker(info) != entry["marker"]: return False
        for expected in identity["import_surface"]:
            current=_import_surface(Path(expected["root"]),with_hash=False)
            if current != {key:value for key,value in expected.items() if key != "sha256" and not (key == "entries" and any("sha256" in x for x in value))}:
                # Compare a marker-only projection without rereading bytes.
                def marker_only(value: Any) -> Any:
                    if isinstance(value,dict): return {key:marker_only(row) for key,row in value.items() if key not in {"sha256","target_sha256"}}
                    if isinstance(value,list): return [marker_only(row) for row in value]
                    return value
                projection=marker_only(expected)
                if current != projection: return False
        return True
    except (KeyError,TypeError,OSError,ValueError):
        return False


def family_hash(families: tuple[TeacherFamily, ...] = REQUIRED_TEACHERS) -> str:
    rows = [{"id": x.stable_id, "repo": x.repo, "revision": x.revision, "package": x.package, "decode": x.decode} for x in families]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_disjoint_lineage(candidate: dict[str, Any], teachers: tuple[TeacherFamily, ...] = REQUIRED_TEACHERS) -> bool:
    """Reject semantic code/model overlap rather than claiming independence."""
    candidate_values = {str(candidate.get(x, "")) for x in ("stable_id", "repo", "revision", "backend", "lineage_hash", "source_hash")}
    if any({t.stable_id, t.repo, t.revision} & candidate_values for t in teachers):
        return False
    lineage = candidate.get("lineage")
    if isinstance(lineage, (list, tuple, set)):
        teacher_values = {v for t in teachers for v in (t.stable_id, t.repo, t.revision)}
        return not bool(set(map(str, lineage)) & teacher_values)
    return True


def receipt_identity(receipt: dict[str, Any]) -> str:
    """Bind interpreter, runtime packages, snapshot files, adapter and decode."""
    required = ("family", "repo", "revision", "interpreter", "interpreter_identity", "snapshot_path", "snapshot_digest", "package_versions", "adapter", "adapter_hash", "decode", "canonicalizer_hash", "canonicalizer_source_hash")
    if any(key not in receipt for key in required):
        raise ValueError("teacher receipt lacks immutable identity")
    return hashlib.sha256(json.dumps({key: receipt[key] for key in required}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_teacher_receipts_read_only(receipts: dict[str, dict[str, Any]], *,
                                        snapshot_cache: dict[str, tuple[str,str]] | None = None,
                                        runtime_cache: dict[str,dict[str,Any]] | None = None,
                                        runtime_probe: Callable[[Path,tuple[str,...],dict[str,Any] | None],dict[str,Any]] | None = None) -> str:
    """Runtime execution's strict receipt check, factored for no-write status.

    This intentionally performs the same interpreter, adapter and snapshot
    checks as the runner.  It neither provisions nor creates private paths.
    """
    if not isinstance(receipts,dict): raise RuntimeError("teacher receipts malformed")
    runtime_probe=runtime_probe or runtime_identity
    identities=[]; cache=snapshot_cache if snapshot_cache is not None else {}; runtimes=runtime_cache if runtime_cache is not None else {}
    for teacher in REQUIRED_TEACHERS:
        receipt=receipts.get(teacher.stable_id)
        if not isinstance(receipt,dict) or receipt.get("family") != teacher.stable_id:
            raise RuntimeError("missing pinned teacher receipt")
        path=Path(str(receipt.get("snapshot_path", ""))); interpreter=Path(str(receipt.get("interpreter", ""))); adapter=Path(str(receipt.get("adapter", "")))
        if (receipt.get("repo") != teacher.repo or receipt.get("revision") != teacher.revision or receipt.get("decode") != teacher.decode or
                not path.is_absolute() or not interpreter.is_absolute() or not adapter.is_absolute() or not interpreter.is_file()):
            raise RuntimeError("teacher receipt path is invalid")
        teacher_snapshot(teacher,path)
        marker=snapshot_stat_manifest(teacher,path); cached=cache.get(str(path))
        if cached is None or cached[0] != marker: cache[str(path)]=(marker,snapshot_digest(teacher,path))
        if sha256_file(adapter) != receipt.get("adapter_hash") or cache[str(path)][1] != receipt.get("snapshot_digest"):
            raise RuntimeError("teacher receipt artifact changed")
        if (receipt.get("canonicalizer_hash") != policy_hash() or
                receipt.get("canonicalizer_source_hash") != sha256_file(Path(teacher_consensus.__file__))):
            raise RuntimeError("teacher canonicalizer changed")
        expected=receipt.get("interpreter_identity")
        cached_runtime=runtimes.get(teacher.stable_id)
        current=expected if cached_runtime == expected and _runtime_markers_current(expected) else runtime_probe(interpreter,teacher.critical_packages,expected)
        if current != expected:
            raise RuntimeError("teacher runtime changed")
        if receipt.get("package_versions") != expected.get("packages"):
            raise RuntimeError("teacher package receipt changed")
        runtimes[teacher.stable_id]=expected
        identities.append(receipt_identity(receipt))
    return hashlib.sha256((family_hash()+policy_hash()+"".join(sorted(identities))).encode()).hexdigest()


class OfflineTeacherRunner:
    """Runs receipt-declared interpreters without shell interpolation or Hub ids."""
    def __init__(self, receipts: dict[str, dict[str, Any]], *, invoke: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
                 popen: Callable[..., Any] = subprocess.Popen, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.receipts = receipts
        self.invoke = invoke
        self.popen = popen
        self.sleep = sleep
        self.monotonic = monotonic
        self._snapshot_cache: dict[str, tuple[str, str]] = {}
        self._runtime_cache: dict[str,dict[str,Any]] = {}

    def validate(self) -> str:
        return validate_teacher_receipts_read_only(self.receipts,snapshot_cache=self._snapshot_cache,runtime_cache=self._runtime_cache)

    def _terminate_cancellable(self, process: Any) -> None:
        """Stop a dedicated process group and drain pipes without content logs."""
        pid = getattr(process, "pid", None)
        try:
            if type(pid) is int and pid > 0:
                os.killpg(pid, signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, AttributeError):
            pass
        deadline = self.monotonic() + 1.0
        while process.poll() is None and self.monotonic() < deadline:
            self.sleep(.05)
        if process.poll() is None:
            try:
                if type(pid) is int and pid > 0:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
            except (OSError, AttributeError):
                pass
        try:
            process.communicate()
        except (OSError, subprocess.SubprocessError, AttributeError):
            pass

    def _invoke_cancellable(self, argv: list[str], env: dict[str, str], cancel: Callable[[], bool], payload: str = "") -> str:
        process = self.popen(argv, shell=False, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=env, cwd="/", start_new_session=True)
        try:
            process.stdin.write(payload); process.stdin.close()
            # communicate() otherwise tries to flush the already closed pipe
            # on Python 3.12, breaking cancellation before TeacherPreempted.
            process.stdin = None
        except (OSError, ValueError, AttributeError):
            # A dead or fixture child surfaces through returncode handling.
            pass
        deadline = self.monotonic() + 120.0
        while process.poll() is None:
            if cancel():
                self._terminate_cancellable(process)
                raise TeacherPreempted("teacher_preempted")
            if self.monotonic() >= deadline:
                self._terminate_cancellable(process)
                raise RuntimeError("teacher_timeout")
            self.sleep(.05)
        try:
            stdout, _stderr = process.communicate()
        except (OSError, subprocess.SubprocessError, AttributeError) as exc:
            raise RuntimeError("teacher_failed") from exc
        if process.returncode != 0:
            raise RuntimeError("teacher_failed")
        return str(stdout).strip()

    def transcribe(self, family: TeacherFamily, audio_path: Path, identity: str,
                   cancel: Callable[[], bool] | None = None) -> str:
        receipt = self.receipts[family.stable_id]
        # The worker command is supplied by provisioning and gets only opaque
        # identity, WAV path, snapshot path and JSON decode settings.
        # Revalidate the bound launcher closure before every executable use.
        self.validate()
        adapter = Path(str(receipt["adapter"]))
        # Close the validate-to-exec window on the adapter itself: bind its
        # exact bytes again immediately before the spawn.
        if sha256_file(adapter) != receipt.get("adapter_hash"):
            raise RuntimeError("teacher receipt artifact changed")
        # ``-I -S`` starts the adapter stdlib-only.  The receipt-frozen
        # import-root trees are piped on stdin; the adapter child re-verifies
        # every byte of them itself before inserting any root.
        runtime = receipt.get("interpreter_identity")
        payload = json.dumps({"expected_import_roots": (runtime or {}).get("import_roots")}, sort_keys=True, separators=(",", ":"))
        argv = [str(receipt["interpreter"]), "-I", "-S", "-B", str(adapter),
                "--audio", str(audio_path), "--identity", identity,
                "--snapshot", str(receipt["snapshot_path"]), "--decode", json.dumps(family.decode, sort_keys=True)]
        env = _teacher_env()
        if cancel is not None:
            return self._invoke_cancellable(argv, env, cancel, payload)
        completed = self.invoke(argv, shell=False, check=False, text=True, capture_output=True, env=env, cwd="/", timeout=120, input=payload)
        if completed.returncode != 0:
            raise RuntimeError("teacher_failed")
        # stdout is intentionally used only in memory. Do not include it in exceptions.
        return completed.stdout.strip()
