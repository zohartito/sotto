"""Offline construction and verification of private sealed Sotto releases.

This module only manipulates a caller-supplied release root.  It never installs
launchd jobs, opens the application store, or starts a process.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable

# Staging/verification is an administrative sealed-source path as well: do not
# create mutable bytecode beside any source it validates or imports.
sys.dont_write_bytecode = True

from history import _fsync_dir
from runtime_source_manifest import runtime_source_manifest


RELEASE_SCHEMA = 1
_HEX = set("0123456789abcdef")
_RUNTIME_POLICY_SCHEMA = 1
_RUNTIME_POLICY_FILE = "runtime_dependency_policy.json"


def _create_private_dir(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        path.mkdir(parents=True, mode=0o700)
        info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("unsafe release directory")
    os.chmod(path, 0o700)


def _private_dir(path: Path) -> None:
    """Read-only private-directory assertion for verification paths."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("unsafe release directory") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError("unsafe release directory")


def _regular(path: Path, *, mode: int | None = None) -> os.stat_result:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("release artifact is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or (mode is not None and stat.S_IMODE(info.st_mode) != mode):
        raise RuntimeError("release artifact is unsafe")
    return info


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compact(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("unsafe release path")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("unsafe release path")
    return path


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def _launcher_chain(configured: Path) -> tuple[list[dict[str, str]], Path]:
    """Bind the launcher path as well as its final executable target.

    A virtualenv launcher is normally a symlink chain.  Resolving it before a
    probe or exec silently drops virtualenv ``sys.prefix`` semantics, so record
    each lexical link and use the configured launcher for subprocesses.
    """
    if not configured.is_absolute():
        raise RuntimeError("release Python is not absolute")
    current = configured
    chain: list[dict[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(16):
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RuntimeError("release Python is unavailable") from exc
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            raise RuntimeError("release Python launcher loop")
        seen.add(identity)
        if stat.S_ISLNK(info.st_mode):
            try:
                link = os.readlink(current)
            except OSError as exc:
                raise RuntimeError("release Python is unavailable") from exc
            chain.append({"path": str(current), "link_target": link,
                          "sha256": _digest_bytes(link.encode("utf-8"))})
            current = Path(link) if os.path.isabs(link) else current.parent / link
            continue
        if not stat.S_ISREG(info.st_mode) or not os.access(current, os.X_OK):
            raise RuntimeError("release Python is unsafe")
        chain.append({"path": str(current), "sha256": _digest_bytes(current.read_bytes())})
        try:
            return chain, current.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("release Python is unavailable") from exc
    raise RuntimeError("release Python launcher chain is too deep")


def _runtime_policy() -> dict[str, Any]:
    """Load the checked-in, pinned executable-distribution policy.

    The policy is generated from requirements.txt and its active dependency
    graph.  Only the explicitly named build tooling is excluded; a release
    refuses a venv whose executable closure no longer exactly matches it.
    """
    root = Path(__file__).resolve().parent
    policy_path = root / _RUNTIME_POLICY_FILE
    requirements = root / "requirements.txt"
    try:
        policy_bytes = policy_path.read_bytes(); requirements_bytes = requirements.read_bytes()
        policy = json.loads(policy_bytes)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime dependency policy is unavailable") from exc
    required = {"schema", "requirements", "requirements_sha256", "excluded", "roots", "distributions"}
    if not isinstance(policy, dict) or set(policy) != required or policy.get("schema") != _RUNTIME_POLICY_SCHEMA:
        raise RuntimeError("runtime dependency policy is malformed")
    if policy.get("requirements") != "requirements.txt" or policy.get("requirements_sha256") != _digest_bytes(requirements_bytes):
        raise RuntimeError("runtime dependency policy does not bind requirements")
    if policy.get("excluded") != ["pip", "setuptools", "wheel"]:
        raise RuntimeError("runtime dependency policy exclusion is unsafe")
    roots = policy.get("roots"); distributions = policy.get("distributions")
    if (not isinstance(roots, list) or not roots or any(not isinstance(value, str) for value in roots) or
            not isinstance(distributions, list) or not distributions):
        raise RuntimeError("runtime dependency policy is malformed")
    parsed_roots = []
    for line in requirements_bytes.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        name = line.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0].strip()
        if not name: raise RuntimeError("runtime requirements are malformed")
        parsed_roots.append(_normalize_distribution(name))
    if roots != parsed_roots:
        raise RuntimeError("runtime dependency policy roots drifted")
    names: list[str] = []
    for item in distributions:
        if not isinstance(item, dict) or set(item) != {"name", "version"} or not isinstance(item["version"], str) or not item["version"]:
            raise RuntimeError("runtime dependency policy is malformed")
        name = _normalize_distribution(item.get("name"))
        if name != item["name"]: raise RuntimeError("runtime dependency policy is malformed")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)) or any(root_name not in names for root_name in roots):
        raise RuntimeError("runtime dependency policy is malformed")
    return {**policy, "digest": _digest_bytes(_compact(policy))}


def _normalize_distribution(value: Any) -> str:
    if not isinstance(value, str) or not value: raise RuntimeError("runtime distribution name is unsafe")
    result = []
    previous_separator = False
    for char in value.lower():
        if char.isalnum(): result.append(char); previous_separator = False
        elif char in "-_.":
            if not previous_separator: result.append("-")
            previous_separator = True
        else: raise RuntimeError("runtime distribution name is unsafe")
    name = "".join(result).strip("-")
    if not name: raise RuntimeError("runtime distribution name is unsafe")
    return name


def _stable_marker(info: os.stat_result, *, volatile: bool = False) -> list[int]:
    """Cross-boot file marker; content hashes remain the byte authority.

    macOS can renumber ``st_dev`` for the same unchanged APFS volume after a
    reboot.  Persisting it made a valid sealed runtime crash-loop even though
    every path, inode, timestamp and SHA-256 digest still matched.
    """
    return [info.st_ino, info.st_size,
            0 if volatile else info.st_mtime_ns,
            0 if volatile else info.st_ctime_ns]


def _sealed_import_root(root: Path) -> dict[str, Any]:
    """Freeze every importable byte below one post--S venv root."""
    try: info=os.lstat(root)
    except OSError as exc: raise RuntimeError("sealed import root unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError("sealed import root unsafe")
    rows=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs.sort(); files.sort(); base=Path(directory).relative_to(root)
        for name in [*dirs,*files]:
            path=Path(directory)/name; entry=os.lstat(path); rel=(base/name).as_posix()
            if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)):
                raise RuntimeError("sealed import root unsafe")
            # Interpreter machinery creates and removes temp files inside
            # ``__pycache__`` even under -B, permanently churning only that
            # directory's timestamps.  Every contained byte stays hashed and
            # any added/removed entry still changes the tree, so those two
            # volatile fields carry no authority for pycache directories.
            volatile=stat.S_ISDIR(entry.st_mode) and name == "__pycache__"
            row={"path":rel,"type":"dir" if stat.S_ISDIR(entry.st_mode) else "file","marker":_stable_marker(entry,volatile=volatile)}
            if stat.S_ISREG(entry.st_mode): row["sha256"]=_digest_bytes(path.read_bytes())
            rows.append(row)
    return {"root":str(root),"marker":_stable_marker(info),"entries":rows}


def _probe_python_layout(interpreter: Path) -> dict[str,Any]:
    """Stdlib-only isolated layout discovery; no site root is imported."""
    code="import json,sys,sysconfig; print(json.dumps({'python':{'version':sys.version,'executable':sys.executable,'implementation':sys.implementation.name,'unicode_version':__import__('unicodedata').unidata_version,'prefix':sys.prefix,'base_prefix':sys.base_prefix},'roots':[sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]}))"
    run=subprocess.run([str(interpreter),"-I","-S","-B","-c",code],shell=False,check=False,text=True,capture_output=True,env={"PATH":"/usr/bin:/bin","PYTHONDONTWRITEBYTECODE":"1"},timeout=30)
    if run.returncode: raise RuntimeError("release runtime layout probe failed")
    value=json.loads(run.stdout); roots=value.get("roots")
    if not isinstance(roots,list) or any(not isinstance(x,str) or not Path(x).is_absolute() for x in roots): raise RuntimeError("release runtime layout malformed")
    value["roots"]=list(dict.fromkeys(roots)); return value


def runtime_closure(*, python: Path | str | None = None, ffmpeg: Path | str | None = None) -> dict[str, Any]:
    """Capture the external interpreter/packages/FFmpeg closure used by release code."""
    configured = Path(python or sys.executable)
    try:
        chain, resolved = _launcher_chain(configured); info = os.lstat(resolved)
    except OSError as exc:
        raise RuntimeError("release Python is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise RuntimeError("release Python is unsafe")
    binary = _digest_bytes(resolved.read_bytes())
    layout=_probe_python_layout(configured); import_roots=[_sealed_import_root(Path(root)) for root in layout["roots"]]
    policy = _runtime_policy()
    packages = _probe_runtime(configured, policy=policy, import_roots=import_roots)
    raw_ffmpeg = Path(ffmpeg or os.environ.get("SOTTO_FFMPEG", "/opt/homebrew/bin/ffmpeg"))
    if not raw_ffmpeg.is_absolute(): raise RuntimeError("release ffmpeg is not absolute")
    try:
        resolved_ffmpeg = raw_ffmpeg.resolve(strict=True); ff_info = os.lstat(resolved_ffmpeg)
        completed = subprocess.run([str(resolved_ffmpeg), "-version"], shell=False, check=False, text=True, capture_output=True,
                                   env={"PATH":"/usr/bin:/bin", "SOTTO_OFFLINE":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("release ffmpeg is unavailable") from exc
    first = completed.stdout.splitlines()[0] if isinstance(completed.stdout, str) and completed.stdout.splitlines() else ""
    if (stat.S_ISLNK(ff_info.st_mode) or not stat.S_ISREG(ff_info.st_mode) or not os.access(resolved_ffmpeg, os.X_OK) or
            completed.returncode != 0 or not first.startswith("ffmpeg version ")):
        raise RuntimeError("release ffmpeg is unsafe")
    return {"schema": 1,
            "python": {"configured_path": str(configured), "resolved_path": str(resolved), "launcher_chain": chain, "sha256": binary,
                       **packages.pop("python")}, "packages": packages["packages"],
            "import_roots": import_roots,
            "distribution_policy": {"schema": policy["schema"], "digest": policy["digest"],
                                    "requirements_sha256": policy["requirements_sha256"], "roots": policy["roots"],
                                    "excluded": policy["excluded"], "distributions": policy["distributions"]},
            "ffmpeg": {"configured_path": str(raw_ffmpeg), "resolved_path": str(resolved_ffmpeg),
                       "sha256": _digest_bytes(resolved_ffmpeg.read_bytes()), "version_first_line": first,
                       "version_sha256": _digest_bytes(first.encode("utf-8"))}}


def _probe_runtime(interpreter: Path, *, policy: dict[str, Any], expected: dict[str, Any] | None = None,
                   import_roots: list[dict[str,Any]] | None = None) -> dict[str, Any]:
    """Ask the exact pinned interpreter for versions and installed-file bytes."""
    code = """import os,sys,json,hashlib,stat
payload=json.load(sys.stdin); expected_roots=payload['import_roots']
def tree(root):
 info=os.lstat(root)
 if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError('unsafe_import_root')
 rows=[]
 for directory,dirs,files in os.walk(root,followlinks=False):
  dirs.sort(); files.sort(); base=os.path.relpath(directory,root)
  for name in dirs+files:
   path=os.path.join(directory,name); entry=os.lstat(path); rel=name if base=='.' else base+'/'+name
   if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)): raise RuntimeError('unsafe_import_root')
   v=stat.S_ISDIR(entry.st_mode) and name=='__pycache__'
   row={'path':rel,'type':'dir' if stat.S_ISDIR(entry.st_mode) else 'file','marker':[entry.st_ino,entry.st_size,0 if v else entry.st_mtime_ns,0 if v else entry.st_ctime_ns]}
   if stat.S_ISREG(entry.st_mode):
    d=hashlib.sha256(); f=open(path,'rb')
    for b in iter(lambda:f.read(1048576),b''): d.update(b)
    f.close(); row['sha256']=d.hexdigest()
   rows.append(row)
 return {'root':root,'marker':[info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns],'entries':rows}
if [tree(row['root']) for row in expected_roots] != expected_roots: raise RuntimeError('import_root_changed')
sys.path[:]=[row['root'] for row in expected_roots]+[p for p in sys.path if p not in [row['root'] for row in expected_roots]]
import hashlib,importlib.metadata as m,json,re,stat,unicodedata
from packaging.markers import default_environment
from packaging.requirements import Requirement
def h(p):
 s=os.lstat(p)
 if stat.S_ISLNK(s.st_mode) or not stat.S_ISREG(s.st_mode): raise RuntimeError('unsafe_distribution_file')
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): d.update(b)
 return d.hexdigest()
def marker(s): return [s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
policy=payload['policy']; expected=payload.get('expected')
locked=policy['distributions']; names=[row['name'] for row in locked]
if names != sorted(names) or len(names) != len(set(names)): raise RuntimeError('runtime_policy_malformed')
if expected is not None and set(expected) != set(names): raise RuntimeError('runtime_package_set_changed')
def norm(value): return re.sub(r'[-_.]+','-',value).lower()
environment=default_environment(); environment['extra']=''
active=set(); todo=list(policy['roots']); excluded=set(policy['excluded'])
while todo:
 name=norm(todo.pop())
 if name in active or name in excluded: continue
 try: dist=m.distribution(name)
 except m.PackageNotFoundError: raise RuntimeError('runtime_distribution_missing')
 active.add(name)
 for raw in dist.requires or ():
  requirement=Requirement(raw)
  if requirement.marker is None or requirement.marker.evaluate(environment): todo.append(requirement.name)
if active != set(names): raise RuntimeError('runtime_dependency_graph_changed')
packages={}
for locked_row in locked:
 name=locked_row['name']
 try: dist=m.distribution(name)
 except m.PackageNotFoundError: raise RuntimeError('runtime_distribution_missing')
 if dist.version != locked_row['version']: raise RuntimeError('runtime_distribution_version_changed')
 discovered={}
 for rel in sorted(dist.files or (),key=lambda x:str(x)):
  p=dist.locate_file(rel)
  try: s=os.lstat(p)
  except FileNotFoundError: raise RuntimeError('runtime_distribution_file_missing')
  if stat.S_ISDIR(s.st_mode): continue
  if stat.S_ISLNK(s.st_mode) or not stat.S_ISREG(s.st_mode): raise RuntimeError('unsafe_distribution_file')
  discovered[str(rel)]=(p,s)
 if not discovered: raise RuntimeError('runtime_distribution_has_no_files')
 previous=None if expected is None else expected[name]
 if previous is not None and (not isinstance(previous,dict) or previous.get('version') != dist.version or not isinstance(previous.get('files'),list)):
  raise RuntimeError('runtime_distribution_identity_malformed')
 prior={} if previous is None else {row.get('path'):row for row in previous['files'] if isinstance(row,dict)}
 if previous is not None and (len(prior) != len(previous['files']) or set(prior) != set(discovered)):
  raise RuntimeError('runtime_distribution_file_set_changed')
 files=[]
 for rel,(p,s) in discovered.items():
  old=prior.get(rel)
  current=marker(s)
  if old is not None:
   if set(old) != {'path','sha256','marker'} or not isinstance(old['sha256'],str) or not isinstance(old['marker'],list): raise RuntimeError('runtime_distribution_identity_malformed')
   if old['marker'] == current: files.append(old); continue
   if h(p) != old['sha256']: raise RuntimeError('runtime_distribution_file_changed')
   files.append(old); continue
  files.append({'path':rel,'sha256':h(p),'marker':current})
 packages[name]={'version':dist.version,'files':files}
print(json.dumps({'python':{'version':sys.version,'executable':sys.executable,'implementation':sys.implementation.name,'unicode_version':unicodedata.unidata_version,'prefix':sys.prefix,'base_prefix':sys.base_prefix},'packages':packages},sort_keys=True,separators=(',',':')))
"""
    try:
        run = subprocess.run([str(interpreter), "-I", "-S", "-B", "-c", code], shell=False, check=False, text=True, capture_output=True,
                             input=_compact({"policy": {"distributions": policy["distributions"], "roots": policy["roots"], "excluded": policy["excluded"]}, "expected": expected, "import_roots": import_roots or []}).decode("utf-8"),
                             env={"PATH":"/usr/bin:/bin", "SOTTO_OFFLINE":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PYTHONDONTWRITEBYTECODE":"1"}, timeout=120)
        if run.returncode or len(run.stdout) > 32 * 1024 * 1024: raise RuntimeError("release runtime probe failed")
        value = json.loads(run.stdout)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise RuntimeError("release runtime probe failed") from exc
    if not isinstance(value,dict) or not isinstance(value.get("python"),dict) or not isinstance(value.get("packages"),dict):
        raise RuntimeError("release runtime probe malformed")
    return value


def _release_body(source: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    return {"schema": RELEASE_SCHEMA, "source_manifest": source, "runtime": runtime,
            "entrypoints": {"app": "sotto.py", "worker": "adaptive_worker.py", "bootstrap": "sealed_release_bootstrap.py",
                            "teacher_provision": "teacher_provision.py", "calibration_manifest": "calibration_manifest_builder.py",
                            "calibration_worker": "calibration_worker.py"}}


def _release_digest(body: dict[str, Any]) -> str:
    return _digest_bytes(_compact(body))


def stage_release(releases_root: Path | str, *, source_root: Path | str | None = None,
                  runtime: dict[str, Any] | None = None) -> Path:
    """Atomically publish one content-addressed, private source bundle."""
    root = Path(releases_root); _create_private_dir(root)
    source_root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    source = runtime_source_manifest(source_root); closure = runtime if runtime is not None else runtime_closure()
    body = _release_body(source, closure); digest = _release_digest(body); final = root / digest
    if final.exists() or os.path.lexists(final):
        verify_release(final, runtime_validator=lambda value: value == closure)
        return final
    stage = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=root)); os.chmod(stage, 0o700)
    try:
        source_dir = stage / "source"; _create_private_dir(source_dir)
        for item in source["files"]:
            relative = _safe_relative(item.get("path")); expected = item.get("sha256")
            origin = source_root / relative
            _regular(origin)
            data = origin.read_bytes()
            if _digest_bytes(data) != expected:
                raise RuntimeError("source changed during release staging")
            parent = source_dir / relative.parent
            _create_private_dir(parent)
            _write_private(source_dir / relative, data)
        manifest = {**body, "release_digest": digest}
        _write_private(stage / "manifest.json", _compact(manifest))
        _fsync_dir(source_dir); _fsync_dir(stage)
        if os.path.lexists(final):
            raise RuntimeError("release destination already exists")
        os.rename(stage, final); _fsync_dir(root)
        verify_release(final, runtime_validator=lambda value: value == closure)
        return final
    except BaseException:
        if stage.exists() and stage.parent == root: shutil.rmtree(stage, ignore_errors=True)
        raise


def _read_manifest(bundle: Path) -> dict[str, Any]:
    _private_dir(bundle); target = bundle / "manifest.json"; _regular(target, mode=0o600)
    try: value = json.loads(target.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc: raise RuntimeError("release manifest is malformed") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "source_manifest", "runtime", "entrypoints", "release_digest"}:
        raise RuntimeError("release manifest is malformed")
    body = {key:value[key] for key in ("schema", "source_manifest", "runtime", "entrypoints")}
    if (value["schema"] != RELEASE_SCHEMA or not isinstance(value.get("release_digest"), str) or
            value["release_digest"] != _release_digest(body) or bundle.name != value["release_digest"]):
        raise RuntimeError("release manifest identity mismatch")
    return value


def verify_release(bundle: Path | str, *, runtime_validator: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    """Verify source bytes, strict modes/tree, and optionally runtime closure."""
    root = Path(bundle); manifest = _read_manifest(root); source = manifest["source_manifest"]
    for child in root.iterdir():
        if child.name not in {"source", "manifest.json"} or stat.S_ISLNK(os.lstat(child).st_mode):
            raise RuntimeError("release contains extra artifact")
    if not isinstance(source, dict) or set(source) != {"schema", "files", "digest"} or not isinstance(source["files"], list):
        raise RuntimeError("release source manifest is malformed")
    source_body = {"schema": source["schema"], "files": source["files"]}
    if source["digest"] != _digest_bytes(_compact(source_body)):
        raise RuntimeError("release source manifest identity mismatch")
    source_dir = root / "source"; _private_dir(source_dir)
    expected_files: set[Path] = set(); expected_dirs: set[Path] = {Path(".")}
    for item in source["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item["sha256"], str):
            raise RuntimeError("release source manifest is malformed")
        relative = _safe_relative(item["path"]); expected_files.add(relative)
        expected_dirs.update(relative.parents)
        target = source_dir / relative; _regular(target, mode=0o600)
        if _digest_bytes(target.read_bytes()) != item["sha256"]:
            raise RuntimeError("release source artifact changed")
    for current, dirs, files in os.walk(source_dir, followlinks=False):
        base = Path(current); rel = base.relative_to(source_dir)
        if rel not in expected_dirs: raise RuntimeError("release contains extra directory")
        _private_dir(base)
        for name in [*dirs, *files]:
            path = base / name
            if stat.S_ISLNK(os.lstat(path).st_mode): raise RuntimeError("release contains symlink")
        for name in files:
            if (base / name).relative_to(source_dir) not in expected_files: raise RuntimeError("release contains extra file")
    if runtime_validator is not None and not runtime_validator(manifest["runtime"]):
        raise RuntimeError("release runtime closure changed")
    return manifest


def _pointer_target(root: Path, name: str) -> Path | None:
    pointer = root / name
    if not os.path.lexists(pointer): return None
    try:
        if not stat.S_ISLNK(os.lstat(pointer).st_mode): raise RuntimeError("release pointer is unsafe")
        target = os.readlink(pointer)
    except OSError as exc: raise RuntimeError("release pointer is unsafe") from exc
    if len(target) != 64 or any(ch not in _HEX for ch in target): raise RuntimeError("release pointer is unsafe")
    return root / target


def _replace_pointer(root: Path, name: str, target: Path) -> None:
    if target.parent != root or len(target.name) != 64 or any(ch not in _HEX for ch in target.name): raise RuntimeError("release pointer target is unsafe")
    temporary = root / f".{name}.{os.urandom(8).hex()}.tmp"; os.symlink(target.name, temporary); os.replace(temporary, root / name); _fsync_dir(root)


def activate_release(releases_root: Path | str, digest: str, *, runtime_validator: Callable[[dict[str, Any]], bool] | None = None) -> Path:
    root = Path(releases_root); _create_private_dir(root)
    if len(digest) != 64 or any(ch not in _HEX for ch in digest): raise RuntimeError("release digest is unsafe")
    target = root / digest; verify_release(target, runtime_validator=runtime_validator)
    previous = _pointer_target(root, "current")
    if previous is not None:
        verify_release(previous, runtime_validator=runtime_validator)
        # A repeated activation of the current immutable bundle must not churn
        # either pointer: LKG remains the preceding verified release.
        if previous == target:
            return target
        _replace_pointer(root, "lkg", previous)
    _replace_pointer(root, "current", target)
    return target


def rollback_release(releases_root: Path | str, *, runtime_validator: Callable[[dict[str, Any]], bool] | None = None) -> Path:
    root = Path(releases_root); _private_dir(root); target = _pointer_target(root, "lkg")
    if target is None: raise RuntimeError("no verified LKG release")
    verify_release(target, runtime_validator=runtime_validator)
    previous = _pointer_target(root, "current")
    if previous is not None: verify_release(previous, runtime_validator=runtime_validator)
    _replace_pointer(root, "current", target)
    if previous is not None: _replace_pointer(root, "lkg", previous)
    return target


def prepare_logs(log_root: Path | str) -> dict[str, Path]:
    root = Path(log_root); _create_private_dir(root); result = {}
    for name in ("app.log", "app-error.log", "worker.log", "worker-error.log"):
        target = root / name
        if not os.path.lexists(target): _write_private(target, b"")
        _regular(target, mode=0o600); result[name] = target
    return result
