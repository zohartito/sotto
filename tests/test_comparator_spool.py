import hashlib
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import wave

import numpy as np

import comparator_spool
from audio_codec import prepare_canonical
from comparator_spool import ComparatorSpool, _open_base_dirfd, _open_owned_dirfd, _read_fd


def _wav() -> tuple[bytes, str]:
    pcm=prepare_canonical(np.zeros(800,np.float32)).pcm
    raw=io.BytesIO()
    with wave.open(raw,"wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16_000); handle.writeframes(pcm)
    return raw.getvalue(),hashlib.sha256(pcm).hexdigest()


class ComparatorSpoolTests(unittest.TestCase):
    def _layout(self, base: Path) -> tuple[ComparatorSpool, Path, bytes, str]:
        base.chmod(0o700)
        parent=base
        for part in ("adaptive-learning","silver","evidence"):
            parent=parent/part
            parent.mkdir(mode=0o700)
        data,digest=_wav()
        return ComparatorSpool(base), parent/"comparators", data,digest

    def _assert_clean(self, root: Path, name: str) -> None:
        self.assertFalse((root/name).exists())
        self.assertEqual(list(root.glob("*.tmp")),[])

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_class_write_removes_published_file_when_post_publish_validation_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value)/"base"; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base)
            outside=base.parent/"comparator-spool-outside"; outside.write_bytes(b"outside")
            with mock.patch("comparator_spool._read_fd",side_effect=[FileNotFoundError(),RuntimeError("post-rename")]):
                with self.assertRaisesRegex(RuntimeError,"post-rename"):
                    spool.write("read-failure.wav",data,digest)
            self._assert_clean(root,"read-failure.wav")
            original_fsync=os.fsync
            def fail_directory_fsync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError("directory fsync")
                return original_fsync(fd)
            with mock.patch("comparator_spool.os.fsync",side_effect=fail_directory_fsync):
                with self.assertRaisesRegex(OSError,"directory fsync"):
                    spool.write("fsync-failure.wav",data,digest)
            self._assert_clean(root,"fsync-failure.wav")
            self.assertEqual(outside.read_bytes(),b"outside")

    def test_class_rejects_traversal(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value); spool,root,data,digest=self._layout(base); outside=base/"outside.wav"; outside.write_bytes(b"outside")
            for name in ("../outside.wav","x/y.wav","..","plain"):
                with self.assertRaises(ValueError): spool.unlink(name,digest)
                with self.assertRaises(ValueError): spool.write(name,data,digest)
            self.assertEqual(outside.read_bytes(),b"outside"); self.assertFalse(root.exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_absent_fixed_root_is_idempotently_empty_for_unlink_and_scrub(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value); spool,root,_data,digest=self._layout(base)
            self.assertFalse(root.exists())
            self.assertFalse(spool.unlink("missing.wav",digest))
            self.assertTrue(spool.scrub())
            self.assertFalse(root.exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_base_policy_and_private_base_open(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value)/"base"; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base)
            fd=_open_base_dirfd(base); os.close(fd)
            self.assertEqual(spool.write("ok.wav",data,digest),None)
            self.assertEqual(spool.read("ok.wav",digest),data)
            for unsafe in (Path("/"),Path(str(base)+"/..")):
                with self.assertRaises((ValueError,RuntimeError,OSError)): _open_base_dirfd(unsafe)
            base.chmod(0o755)
            with self.assertRaises(RuntimeError): _open_base_dirfd(base)
            base.chmod(0o700); link=base.parent/(base.name+"-link"); link.symlink_to(base,target_is_directory=True)
            with self.assertRaises(OSError): _open_base_dirfd(link)
            self.assertTrue((root/"ok.wav").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_open_base_dirfd_rejects_raw_symlink_and_wrong_mode(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value)/"base"; base.mkdir(); base.chmod(0o700)
            fd=_open_base_dirfd(base); os.close(fd)
            with self.assertRaises(ValueError): _open_base_dirfd(Path(str(base)+"/.."))
            with self.assertRaises(ValueError): _open_base_dirfd(Path("/"))
            base.chmod(0o755)
            with self.assertRaises(RuntimeError): _open_base_dirfd(base)
            base.chmod(0o700)
            link=base.parent/(base.name+"-link"); link.symlink_to(base,target_is_directory=True)
            with self.assertRaises(OSError): _open_base_dirfd(link)

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_open_base_dirfd_closes_child_when_fstat_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value); base.chmod(0o700); original_close=os.close; closed=[]
            def tracked_close(fd): closed.append(fd); return original_close(fd)
            with mock.patch("comparator_spool.os.close",side_effect=tracked_close), mock.patch("comparator_spool.os.fstat",side_effect=OSError("fstat")):
                with self.assertRaisesRegex(OSError,"fstat"):
                    _open_base_dirfd(base)
            self.assertGreaterEqual(len(closed),2)

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_class_scrub_validates_the_entire_direct_set_before_delete(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value)/"base"; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base); outside=base.parent/"scrub-outside"; outside.write_bytes(b"outside")
            def fresh():
                if root.exists():
                    for child in root.iterdir(): child.unlink()
                spool.write("terminal.wav",data,digest); spool.write("pending.wav",data,digest)
            fresh(); self.assertTrue(spool.scrub({"terminal.wav":digest})); self.assertFalse((root/"terminal.wav").exists()); self.assertTrue((root/"pending.wav").exists())
            fresh(); self.assertFalse(spool.scrub({"terminal.wav":"0"*64})); self.assertTrue((root/"terminal.wav").exists()); self.assertTrue((root/"pending.wav").exists())
            self.assertFalse(spool.scrub(["terminal.wav"])); self.assertFalse(spool.scrub({"terminal.wav":7}))
            for kind in ("malformed","fifo","symlink","mode"):
                fresh(); bad=root/"bad.wav"
                if kind == "malformed": bad.write_bytes(b"not a wave"); bad.chmod(0o600)
                elif kind == "fifo": os.mkfifo(bad)
                elif kind == "symlink": bad.symlink_to(outside)
                else: bad.write_bytes(data); bad.chmod(0o644)
                self.assertFalse(spool.scrub({"terminal.wav":digest}),kind)
                self.assertTrue((root/"terminal.wav").exists(),kind); self.assertTrue((root/"pending.wav").exists(),kind)
                self.assertEqual(outside.read_bytes(),b"outside",kind)
                if kind == "fifo": bad.unlink()

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_internal_and_final_unsafe_entries_reject_without_touching_outside(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            outer=Path(value); outside=outer/"outside"; outside.mkdir(); sentinel=outside/"sentinel.wav"; sentinel.write_bytes(b"outside")
            for kind in ("internal-mode","internal-link","final-link","final-file"):
                base=outer/kind; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base)
                if kind == "internal-mode": (base/"adaptive-learning").chmod(0o755)
                elif kind == "internal-link":
                    import shutil; shutil.rmtree(base/"adaptive-learning"); (base/"adaptive-learning").symlink_to(outside,target_is_directory=True)
                elif kind == "final-link": root.symlink_to(outside,target_is_directory=True)
                else: root.write_bytes(b"not-dir")
                with self.assertRaises((RuntimeError,OSError,ValueError)):
                    spool.write("unsafe.wav",data,digest)
                self.assertEqual(sentinel.read_bytes(),b"outside",kind)

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_read_rejects_direct_child_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value)/"base"; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base); spool.write("safe.wav",data,digest)
            outside=base.parent/"child-symlink-outside"; outside.write_bytes(b"outside")
            (root/"child.wav").symlink_to(outside)
            with self.assertRaises((OSError,RuntimeError)): spool.read("child.wav",digest)
            self.assertEqual(outside.read_bytes(),b"outside")

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_private_inode_mismatch_rejects_and_closes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            base=Path(value); spool,root,data,digest=self._layout(base); spool.write("one.wav",data,digest)
            parent=_open_base_dirfd(base); closed=[]; original_close=os.close; original_fstat=os.fstat
            def close(fd): closed.append(fd); return original_close(fd)
            def mismatched(fd):
                info=original_fstat(fd)
                return SimpleNamespace(st_mode=info.st_mode,st_uid=info.st_uid,st_dev=info.st_dev,st_ino=info.st_ino+1)
            with mock.patch("comparator_spool.os.close",side_effect=close), mock.patch("comparator_spool.os.fstat",side_effect=mismatched):
                with self.assertRaises(RuntimeError): _open_owned_dirfd(parent,"adaptive-learning")
            self.assertTrue(closed); original_close(parent)
            with spool._opened() as fd:
                def mismatched_file(candidate):
                    info=original_fstat(candidate)
                    if stat.S_ISREG(info.st_mode): return SimpleNamespace(st_mode=info.st_mode,st_uid=info.st_uid,st_dev=info.st_dev,st_ino=info.st_ino+1)
                    return info
                with mock.patch("comparator_spool.os.fstat",side_effect=mismatched_file):
                    with self.assertRaises(RuntimeError): _read_fd(fd,"one.wav")

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_held_root_fd_survives_root_path_swap_for_read_and_unlink(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            outer=Path(value); outside=outer/"outside"; outside.mkdir(); sentinel=outside/"one.wav"; sentinel.write_bytes(b"outside")
            for action in ("read","unlink"):
                base=outer/action; base.mkdir(mode=0o700); spool,root,data,digest=self._layout(base); spool.write("one.wav",data,digest)
                original=spool._open_chain; detached=root.with_name("detached")
                def swapped(*,create=False):
                    fd=original(create=create); root.rename(detached); root.symlink_to(outside,target_is_directory=True); return fd
                with mock.patch.object(spool,"_open_chain",side_effect=swapped):
                    result=spool.read("one.wav",digest) if action == "read" else spool.unlink("one.wav",digest)
                self.assertEqual(result,data if action == "read" else True)
                self.assertEqual(sentinel.read_bytes(),b"outside")
                if action == "unlink": self.assertFalse((detached/"one.wav").exists())

    @unittest.skipUnless(sys.platform == "darwin", "macOS-only in v1 (POSIX dirfd/getuid/symlink confinement)")
    def test_write_prepublish_failures_leave_no_target_or_temp(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as value:
            outer=Path(value); data,digest=_wav()
            for kind in ("short","file-fsync","rename"):
                base=outer/kind; base.mkdir(mode=0o700); spool,root,_,_=self._layout(base)
                if kind == "short":
                    original_write=os.write; calls=[]
                    def partial_then_zero(fd,payload):
                        if calls: return 0
                        calls.append(True); return original_write(fd,payload[:1])
                    patcher=mock.patch("comparator_spool.os.write",side_effect=partial_then_zero)
                elif kind == "file-fsync":
                    original_fsync=os.fsync
                    def fail_file(fd):
                        if stat.S_ISREG(os.fstat(fd).st_mode): raise OSError("file fsync")
                        return original_fsync(fd)
                    patcher=mock.patch("comparator_spool.os.fsync",side_effect=fail_file)
                else:
                    patcher=mock.patch("comparator_spool.os.replace",side_effect=OSError("rename"))
                with patcher, self.assertRaises(OSError): spool.write("failure.wav",data,digest)
                self._assert_clean(root,"failure.wav")


if __name__ == "__main__":
    unittest.main()
