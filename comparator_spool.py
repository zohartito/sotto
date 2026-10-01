"""FD-relative private comparator WAV spool operations."""
from __future__ import annotations

import hashlib
import os
import stat
import uuid
import wave
import io
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_RELATIVE_ROOT=("adaptive-learning","silver","evidence","comparators")


def _open_owned_dirfd(parent_fd: int, part: str, *, create: bool=False) -> int:
    if not isinstance(part,str) or not part or part in {".",".."} or "/" in part or "\\" in part:
        raise ValueError("unsafe comparator directory component")
    try:
        before=os.stat(part,dir_fd=parent_fd,follow_symlinks=False)
    except FileNotFoundError:
        if not create: raise
        try: os.mkdir(part,0o700,dir_fd=parent_fd)
        except FileExistsError: pass
        before=os.stat(part,dir_fd=parent_fd,follow_symlinks=False)
    child=os.open(part,_flags(),dir_fd=parent_fd)
    try:
        after=os.fstat(child)
        if (not stat.S_ISDIR(after.st_mode) or (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino)
                or after.st_uid != os.getuid() or stat.S_IMODE(after.st_mode) != 0o700):
            raise RuntimeError("unsafe comparator spool directory")
        return child
    except BaseException:
        os.close(child); raise


class ComparatorSpool:
    """A fixed, base-dir-bound comparator spool authority."""
    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir=Path(base_dir).absolute()
        self.root=self.base_dir.joinpath(*_RELATIVE_ROOT)

    def _open_chain(self, *, create: bool=False) -> int:
        """Open the fixed chain; final root is never reopened by path."""
        fd=_open_base_dirfd(self.base_dir)
        try:
            for index,part in enumerate(_RELATIVE_ROOT):
                child=_open_owned_dirfd(fd,part,create=create and index==len(_RELATIVE_ROOT)-1)
                os.close(fd); fd=child
            return fd
        except BaseException:
            os.close(fd); raise

    def read(self,name: str,expected_pcm_sha: str) -> bytes:
        if not _name(name): raise ValueError("unsafe comparator spool name")
        with self._opened() as fd:
            value=_read_fd(fd,name)
        if _canonical_sha(value) != expected_pcm_sha: raise RuntimeError("comparator spool digest mismatch")
        return value

    def write(self,name: str,data: bytes,expected_pcm_sha: str) -> None:
        if not _name(name) or _canonical_sha(data) != expected_pcm_sha: raise ValueError("invalid comparator spool")
        with self._opened(create=True) as directory:
            temp=None; published=None; renamed=False
            try:
                try:
                    if _read_fd(directory,name) == data: return
                    raise RuntimeError("comparator spool conflict")
                except FileNotFoundError: pass
                temp="."+name+"."+uuid.uuid4().hex+".tmp"; fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
                try:
                    offset=0
                    while offset<len(data):
                        count=os.write(fd,data[offset:])
                        if count <= 0: raise OSError("short comparator spool write")
                        offset+=count
                    os.fsync(fd)
                    info=os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o600: raise RuntimeError("unsafe comparator temp")
                    published=(info.st_dev,info.st_ino)
                finally: os.close(fd)
                os.replace(temp,name,src_dir_fd=directory,dst_dir_fd=directory)
                renamed=True; temp=None
                value=_read_fd(directory,name)
                if value != data or _canonical_sha(value) != expected_pcm_sha: raise RuntimeError("comparator spool mismatch")
                os.fsync(directory); temp=None
            except BaseException:
                if temp is not None:
                    try: os.unlink(temp,dir_fd=directory)
                    except FileNotFoundError: pass
                if renamed and published is not None:
                    try:
                        info=os.stat(name,dir_fd=directory,follow_symlinks=False)
                        if (info.st_dev,info.st_ino)==published and stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o600: os.unlink(name,dir_fd=directory)
                        os.fsync(directory)
                    except OSError: pass
                raise

    @contextmanager
    def _opened(self, *, create: bool=False) -> Iterator[int]:
        fd=self._open_chain(create=create)
        try: yield fd
        finally: os.close(fd)

    def unlink(self,name: str,expected_pcm_sha: str) -> bool:
        if not _name(name): raise ValueError("unsafe comparator spool name")
        try:
            with self._opened() as fd:
                try: value=_read_fd(fd,name)
                except FileNotFoundError: return False
                if _canonical_sha(value) != expected_pcm_sha: raise RuntimeError("comparator spool digest mismatch")
                os.unlink(name,dir_fd=fd); os.fsync(fd)
        except FileNotFoundError:
            return False
        return True

    def scrub(self,targets: dict[str,str] | None=None) -> bool:
        if targets is not None:
            if type(targets) is not dict:
                return False
            if any(
                not _name(name)
                or not isinstance(digest,str)
                or len(digest) != 64
                or digest.lower() != digest
                or any(character not in "0123456789abcdef" for character in digest)
                for name,digest in targets.items()
            ):
                return False
        if not _dirfd_authority():
            # Windows has no no-follow directory handles, so no comparator WAV
            # is ever written there (the adaptive lane is macOS-only).  Only an
            # absent spool is complete; anything present keeps the Clear fence.
            return not os.path.lexists(self.root)
        try:
            with self._opened() as fd:
                names=os.listdir(fd)
                if any(not _name(name) for name in names):
                    return False
                # This is deliberately a complete first pass: an unsafe or malformed
                # sibling fences Clear rather than partially deleting private evidence.
                values={name: _read_fd(fd,name) for name in names}
                digests={name: _canonical_sha(value) for name,value in values.items()}
                selected=names if targets is None else [name for name in names if name in targets]
                if targets is not None and any(digests[name] != targets[name] for name in selected):
                    return False
                for name in selected:
                    os.unlink(name,dir_fd=fd)
                os.fsync(fd)
            return True
        except FileNotFoundError:
            # A missing fixed chain/root contains no comparator authority;
            # Clear is idempotently complete in that state.
            return True
        except (OSError,ValueError,RuntimeError,TypeError,wave.Error):
            return False


def _dirfd_authority() -> bool:
    return hasattr(os,"getuid") and hasattr(os,"O_NOFOLLOW") and hasattr(os,"O_DIRECTORY")


def _flags() -> int:
    if not hasattr(os,"O_NOFOLLOW") or not hasattr(os,"O_DIRECTORY"):
        raise RuntimeError("platform lacks no-follow directory authority")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _name(name: str) -> bool:
    return isinstance(name,str) and name.endswith(".wav") and "/" not in name and "\\" not in name and name not in {".wav",".."}


def _canonical_sha(data: bytes) -> str:
    with wave.open(io.BytesIO(data), "rb") as handle:
        if (handle.getnchannels(),handle.getsampwidth(),handle.getframerate(),handle.getcomptype()) != (1,2,16_000,"NONE"):
            raise ValueError("invalid comparator WAV")
        return hashlib.sha256(handle.readframes(handle.getnframes())).hexdigest()


def _open_base_dirfd(base_dir: Path) -> int:
    if not base_dir.is_absolute() or len(base_dir.parts) <= 1 or any(part in {".",".."} for part in base_dir.parts):
        raise ValueError("unsafe comparator base")
    uid=os.getuid(); fd=os.open("/",_flags())
    try:
        parts=base_dir.parts[1:]
        for index,part in enumerate(parts):
            before=os.stat(part,dir_fd=fd,follow_symlinks=False)
            child=os.open(part,_flags(),dir_fd=fd)
            try:
                after=os.fstat(child)
            except BaseException:
                os.close(child); raise
            if not stat.S_ISDIR(after.st_mode) or (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):
                os.close(child); raise RuntimeError("unsafe comparator base")
            valid=(after.st_uid==uid and stat.S_IMODE(after.st_mode)==0o700) if index==len(parts)-1 else (after.st_uid in {0,uid} and not (stat.S_IMODE(after.st_mode)&0o022))
            if not valid:
                os.close(child); raise RuntimeError("unsafe comparator base")
            os.close(fd); fd=child
        return fd
    except BaseException:
        os.close(fd); raise


def _read_fd(directory: int, name: str) -> bytes:
    if not _name(name): raise ValueError("unsafe comparator spool name")
    before=os.stat(name,dir_fd=directory,follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600):
        raise RuntimeError("unsafe comparator spool")
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or (before.st_dev,before.st_ino)!=(info.st_dev,info.st_ino)):
            raise RuntimeError("unsafe comparator spool")
        chunks=[]
        while data:=os.read(fd,1<<20): chunks.append(data)
        after=os.fstat(fd)
        if (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns):
            raise RuntimeError("comparator spool changed")
        return b"".join(chunks)
    finally: os.close(fd)
