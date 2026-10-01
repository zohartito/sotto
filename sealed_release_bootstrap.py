"""Stdlib-only verified entrypoint for a sealed Sotto release.

It is intentionally executable by ``/usr/bin/python3`` before the pinned
external runtime is trusted.  It never imports application code from a mutable
checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import types
from pathlib import Path

# This module is deliberately executed before any sealed application import.
# Keep the sealed source tree immutable across restarts.
sys.dont_write_bytecode = True


def _canonical_isolation() -> bool:
    """Only launchd's ``-I -S -B`` boundary may execute sealed code."""
    flags=sys.flags
    return bool(flags.isolated and flags.no_site and flags.dont_write_bytecode)


def _read_sealed(path: Path) -> bytes:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("sealed release artifact is unsafe")
    return path.read_bytes()


def _sha(path: Path) -> str:
    return hashlib.sha256(_read_sealed(path)).hexdigest()


def _current(releases: Path) -> Path:
    pointer = releases / "current"; info = os.lstat(pointer)
    if not stat.S_ISLNK(info.st_mode): raise RuntimeError("sealed release pointer is unsafe")
    name = os.readlink(pointer)
    if len(name) != 64 or any(char not in "0123456789abcdef" for char in name): raise RuntimeError("sealed release pointer is unsafe")
    return releases / name


def verify(releases: Path) -> tuple[Path, dict]:
    bundle = _current(releases); info = os.lstat(bundle)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700: raise RuntimeError("sealed release is unsafe")
    manifest_path = bundle / "manifest.json"
    try: _sha(manifest_path); manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc: raise RuntimeError("sealed release manifest is malformed") from exc
    body = {key:manifest.get(key) for key in ("schema", "source_manifest", "runtime", "entrypoints")}
    if (manifest.get("release_digest") != hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest() or
            bundle.name != manifest.get("release_digest")): raise RuntimeError("sealed release identity changed")
    source = manifest.get("source_manifest")
    if not isinstance(source,dict) or source.get("digest") != hashlib.sha256(json.dumps({"schema":source.get("schema"),"files":source.get("files")},sort_keys=True,separators=(",",":")).encode()).hexdigest(): raise RuntimeError("sealed source manifest changed")
    root = bundle / "source"; source_info = os.lstat(root)
    if stat.S_ISLNK(source_info.st_mode) or not stat.S_ISDIR(source_info.st_mode) or stat.S_IMODE(source_info.st_mode) != 0o700: raise RuntimeError("sealed source root is unsafe")
    if {path.name for path in bundle.iterdir()} != {"source", "manifest.json"}: raise RuntimeError("sealed release contains extra artifact")
    expected_files=set(); expected_dirs={Path(".")}
    for item in source.get("files",[]):
        if not isinstance(item,dict) or set(item) != {"path","sha256"}: raise RuntimeError("sealed source manifest is malformed")
        relative=Path(item["path"])
        if relative.is_absolute() or relative.as_posix()!=item["path"] or any(part in {"", ".", ".."} for part in relative.parts): raise RuntimeError("sealed source path is unsafe")
        expected_files.add(relative); expected_dirs.update(relative.parents)
        if _sha(root / relative) != item["sha256"]: raise RuntimeError("sealed source changed")
    for current, directories, files in os.walk(root, followlinks=False):
        relative=Path(current).relative_to(root); current_info=os.lstat(current)
        if relative not in expected_dirs or stat.S_ISLNK(current_info.st_mode) or stat.S_IMODE(current_info.st_mode) != 0o700: raise RuntimeError("sealed source tree is unsafe")
        for name in [*directories,*files]:
            if stat.S_ISLNK(os.lstat(Path(current)/name).st_mode): raise RuntimeError("sealed source tree is unsafe")
        if any((Path(current)/name).relative_to(root) not in expected_files for name in files): raise RuntimeError("sealed source contains extra artifact")
    return bundle, manifest


def _verify_runtime(runtime: dict) -> Path:
    try:
        python = runtime["python"]; ffmpeg = runtime["ffmpeg"]
        configured=Path(python["configured_path"]); resolved=Path(python["resolved_path"])
        if (not configured.is_absolute() or _launcher_chain(configured) != python["launcher_chain"] or
                _launcher_terminal(configured) != resolved or _sha_external(resolved) != python["sha256"]): raise RuntimeError
        policy = runtime["distribution_policy"]
        if (not isinstance(policy,dict) or set(policy) != {"schema","digest","requirements_sha256","roots","excluded","distributions"} or
                policy.get("schema") != 1 or not isinstance(policy.get("digest"),str) or not isinstance(policy.get("requirements_sha256"),str) or
                not isinstance(policy.get("roots"),list) or policy.get("excluded") != ["pip","setuptools","wheel"] or not isinstance(policy.get("distributions"),list)):
            raise RuntimeError
        roots=runtime.get("import_roots")
        if not isinstance(roots,list) or not roots or [_sealed_import_root(Path(str(row.get("root","")))) for row in roots] != roots: raise RuntimeError
        observed = _runtime_probe(configured, policy=policy, expected=runtime.get("packages"), import_roots=roots)
        if (not isinstance(observed,dict) or observed.get("python") != {key:python.get(key) for key in ("version","executable","implementation","unicode_version","prefix","base_prefix")} or
                observed.get("packages") != runtime.get("packages")):
            raise RuntimeError
        # The probe is the long window: recompute the exact import-root trees
        # again after the isolated child returns.
        if [_sealed_import_root(Path(str(row.get("root","")))) for row in roots] != roots: raise RuntimeError
        ff_configured=Path(ffmpeg["configured_path"]); ff_resolved=Path(ffmpeg["resolved_path"])
        if not ff_configured.is_absolute() or ff_configured.resolve(strict=True) != ff_resolved or _sha_external(ff_resolved) != ffmpeg["sha256"]: raise RuntimeError
        result=subprocess.run([str(ff_resolved),"-version"],shell=False,check=False,text=True,capture_output=True,env={"PATH":"/usr/bin:/bin","SOTTO_OFFLINE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},timeout=10)
        first=result.stdout.splitlines()[0] if result.returncode==0 and result.stdout.splitlines() else ""
        if first != ffmpeg["version_first_line"] or hashlib.sha256(first.encode()).hexdigest() != ffmpeg["version_sha256"]: raise RuntimeError
    except (KeyError,OSError,ValueError,subprocess.SubprocessError,RuntimeError) as exc: raise RuntimeError("sealed runtime closure changed") from exc
    return configured


def _launcher_chain(configured: Path) -> list[dict[str, str]]:
    if not configured.is_absolute(): raise RuntimeError("sealed runtime launcher is unsafe")
    current=configured; result=[]; seen=set()
    for _ in range(16):
        info=os.lstat(current); identity=(info.st_dev,info.st_ino)
        if identity in seen: raise RuntimeError("sealed runtime launcher loop")
        seen.add(identity)
        if stat.S_ISLNK(info.st_mode):
            link=os.readlink(current); result.append({"path":str(current),"link_target":link,"sha256":hashlib.sha256(link.encode("utf-8")).hexdigest()})
            current=Path(link) if os.path.isabs(link) else current.parent/link
            continue
        if not stat.S_ISREG(info.st_mode) or not os.access(current,os.X_OK): raise RuntimeError("sealed runtime launcher is unsafe")
        result.append({"path":str(current),"sha256":_sha_external(current)})
        return result
    raise RuntimeError("sealed runtime launcher chain is too deep")


def _launcher_terminal(configured: Path) -> Path:
    chain=_launcher_chain(configured)
    try:
        return Path(chain[-1]["path"]).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("sealed runtime launcher is unavailable") from exc


def _sha_external(path: Path) -> str:
    info=os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): raise RuntimeError("sealed runtime artifact is unsafe")
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _stable_marker(info: os.stat_result, *, volatile: bool = False) -> list[int]:
    """Cross-boot marker; APFS ``st_dev`` renumbering is not content drift."""
    return [info.st_ino, info.st_size,
            0 if volatile else info.st_mtime_ns,
            0 if volatile else info.st_ctime_ns]


def _sealed_import_root(root: Path) -> dict:
    try: info=os.lstat(root)
    except OSError as exc: raise RuntimeError("sealed import root unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError("sealed import root unsafe")
    rows=[]
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs.sort(); files.sort(); base=Path(directory).relative_to(root)
        for name in [*dirs,*files]:
            path=Path(directory)/name; entry=os.lstat(path); rel=(base/name).as_posix()
            if stat.S_ISLNK(entry.st_mode) or not (stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode)): raise RuntimeError("sealed import root unsafe")
            # ``__pycache__`` directory timestamps churn from interpreter temp
            # files even under -B; contained bytes and entries stay authoritative.
            volatile=stat.S_ISDIR(entry.st_mode) and name == "__pycache__"
            row={"path":rel,"type":"dir" if stat.S_ISDIR(entry.st_mode) else "file","marker":_stable_marker(entry,volatile=volatile)}
            if stat.S_ISREG(entry.st_mode): row["sha256"]=_sha_external(path)
            rows.append(row)
    return {"root":str(root),"marker":_stable_marker(info),"entries":rows}


def _runtime_probe(interpreter: Path, *, policy: dict, expected: dict | None, import_roots: list[dict] | None = None) -> dict:
    code="""import os,sys,json,hashlib,stat
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
import importlib.metadata as m,re,unicodedata
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
 if not isinstance(locked_row,dict) or set(locked_row) != {'name','version'} or not isinstance(locked_row['name'],str) or not isinstance(locked_row['version'],str): raise RuntimeError('runtime_policy_malformed')
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
 if previous is not None and (not isinstance(previous,dict) or previous.get('version') != dist.version or not isinstance(previous.get('files'),list)): raise RuntimeError('runtime_distribution_identity_malformed')
 prior={} if previous is None else {row.get('path'):row for row in previous['files'] if isinstance(row,dict)}
 if previous is not None and (len(prior) != len(previous['files']) or set(prior) != set(discovered)): raise RuntimeError('runtime_distribution_file_set_changed')
 files=[]
 for rel,(p,s) in discovered.items():
  old=prior.get(rel); current=marker(s)
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
        result=subprocess.run([str(interpreter),"-I","-S","-B","-c",code],shell=False,check=False,text=True,capture_output=True,
                              input=json.dumps({"policy":{"distributions":policy["distributions"],"roots":policy["roots"],"excluded":policy["excluded"]},"expected":expected,"import_roots":import_roots or []},sort_keys=True,separators=(",",":")),
                              env={"PATH":"/usr/bin:/bin","SOTTO_OFFLINE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONDONTWRITEBYTECODE":"1"},timeout=120)
        if result.returncode or len(result.stdout)>32*1024*1024: raise RuntimeError
        return json.loads(result.stdout)
    except (OSError,ValueError,json.JSONDecodeError,subprocess.SubprocessError) as exc:
        raise RuntimeError("sealed runtime package closure changed") from exc


def _declared_source_sha(manifest: dict, relative: str) -> str:
    """The manifest-bound sha256 for one sealed source artifact."""
    source=manifest.get("source_manifest")
    rows=source.get("files") if isinstance(source,dict) else None
    for item in rows if isinstance(rows,list) else ():
        if isinstance(item,dict) and item.get("path") == relative:
            value=item.get("sha256")
            if isinstance(value,str) and len(value) == 64: return value
    raise RuntimeError("sealed source manifest lacks artifact")


def main(argv: list[str] | None = None) -> int:
    if not _canonical_isolation(): raise RuntimeError("sealed bootstrap isolation is required")
    # A sealed process is unconditionally offline: force the flags here so a
    # manual invocation without the launchd environment cannot reach the Hub.
    for name in ("SOTTO_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","PYTHONDONTWRITEBYTECODE"): os.environ[name]="1"
    parser=argparse.ArgumentParser(); parser.add_argument("--release-root",required=True); parser.add_argument("--entry",required=True); args,rest=parser.parse_known_args(argv)
    releases=Path(args.release_root); bundle,manifest=verify(releases); runtime=_verify_runtime(manifest["runtime"])
    # The runtime closure above was verified for THIS release only.  Pin the
    # content digest so a concurrently activated other release cannot execute
    # under it; a refused start lets the supervisor relaunch coherently.
    sealed_digest=bundle.name
    entry=Path(args.entry)
    if entry.is_absolute() or entry.as_posix()!=args.entry or any(part in {"", ".", ".."} for part in entry.parts): raise RuntimeError("sealed entry is unsafe")
    if args.entry not in set(manifest.get("entrypoints", {}).values()): raise RuntimeError("sealed entry is not declared")
    expected=manifest["runtime"]["python"]
    if not _running_pinned_launcher(expected):
        env=dict(os.environ); env["PYTHONDONTWRITEBYTECODE"]="1"
        # Close the probe-to-exec window: the sealed tree is re-verified and
        # the bootstrap bytes are compared to their manifest-bound hash
        # immediately before the configured launcher starts.
        bundle,manifest=verify(releases)
        if bundle.name != sealed_digest: raise RuntimeError("sealed release changed during startup")
        bootstrap=bundle/"source"/"sealed_release_bootstrap.py"
        if _sha(bootstrap) != _declared_source_sha(manifest,"sealed_release_bootstrap.py"): raise RuntimeError("sealed bootstrap changed")
        os.execve(str(runtime),[str(runtime),"-I","-S","-B",str(bootstrap),"--release-root",str(releases),"--entry",args.entry,*rest],env)
    # ``-S`` intentionally disables .pth/sitecustomize processing.  Only the
    # pinned venv import root is restored after runtime closure verification.
    # Final authority: ONE re-verified snapshot supplies the manifest, the
    # entry declaration, the import roots and the exact target bytes, so a
    # concurrently switched ``current`` pointer cannot mix two releases.
    bundle,manifest=verify(releases)
    if bundle.name != sealed_digest: raise RuntimeError("sealed release changed during startup")
    if args.entry not in set(manifest.get("entrypoints", {}).values()): raise RuntimeError("sealed entry is not declared")
    roots=manifest["runtime"].get("import_roots")
    if not isinstance(roots,list) or len(roots)!=1: raise RuntimeError("sealed runtime import roots are unsafe")
    target=bundle/"source"/entry; data=_read_sealed(target)
    if hashlib.sha256(data).hexdigest() != _declared_source_sha(manifest,args.entry): raise RuntimeError("sealed target changed")
    # The mutable external import root is recomputed LAST, immediately before
    # insertion, so no later long check reopens a window on it.
    if _sealed_import_root(Path(str(roots[0]["root"]))) != roots[0]: raise RuntimeError("sealed runtime import roots changed")
    sys.argv=[str(target),*rest]
    sys.path.insert(0,str(bundle/"source")); sys.path.insert(1,str(roots[0]["root"]))
    # Execute exactly the bytes that were verified: no path reopen through
    # runpy, so a post-verification file swap cannot change what runs.
    module=types.ModuleType("__main__"); module.__dict__["__file__"]=str(target)
    sys.modules["__main__"]=module
    exec(compile(data,str(target),"exec"),module.__dict__)
    return 0


def _running_pinned_launcher(expected: dict) -> bool:
    return (isinstance(expected,dict) and sys.prefix == expected.get("prefix") and
            sys.base_prefix == expected.get("base_prefix") and
            str(Path(sys.executable)) == expected.get("configured_path"))


if __name__ == "__main__": raise SystemExit(main())
