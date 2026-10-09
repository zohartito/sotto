"""Canonical audio identity used by history, retry, and learning copies."""
from __future__ import annotations

import hashlib
import os
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CANONICAL_RATE = 16_000
CANONICAL_FORMAT = "pcm_s16le_mono_16000"


@dataclass(frozen=True)
class AudioIdentity:
    sha256: str
    sample_count: int
    sample_rate: int = CANONICAL_RATE
    format: str = CANONICAL_FORMAT
    def as_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "sample_count": self.sample_count,
                "sample_rate": self.sample_rate, "format": self.format}


@dataclass(frozen=True)
class PreparedAudio:
    """The one canonical representation shared by live ASR and persistence.

    ``asr_samples`` are decoded from ``pcm`` rather than returned from the input
    ndarray.  That makes ASR, history, and a later retry all observe precisely
    the same quantized PCM16 samples.
    """
    pcm: bytes
    asr_samples: np.ndarray
    identity: AudioIdentity


def encode_canonical(samples: np.ndarray) -> tuple[bytes, AudioIdentity]:
    return encode_pcm16(samples, CANONICAL_RATE)


def prepare_canonical(samples: np.ndarray) -> PreparedAudio:
    """Encode canonical audio once and expose the exact ASR decode of those bytes."""
    pcm, identity = encode_canonical(samples)
    decoded, decoded_identity = decode_canonical(pcm)
    if decoded_identity != identity:  # Defensive: this must hold for canonical PCM.
        raise ValueError("canonical PCM identity mismatch")
    decoded.setflags(write=False)
    return PreparedAudio(pcm, decoded, identity)


def encode_pcm16(samples: np.ndarray, sample_rate: int) -> tuple[bytes, AudioIdentity]:
    """Encode mono float samples to deterministic little-endian PCM16."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    # 1.0 maps to 32767; -1.0 maps to -32767 for backwards-compatible capture output.
    pcm = (np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()
    fmt = CANONICAL_FORMAT if sample_rate == CANONICAL_RATE else f"pcm_s16le_mono_{sample_rate}"
    return pcm, AudioIdentity(hashlib.sha256(pcm).hexdigest(), len(values), sample_rate, fmt)


def decode_canonical(pcm: bytes) -> tuple[np.ndarray, AudioIdentity]:
    if len(pcm) % 2:
        raise ValueError("PCM16 byte length must be even")
    identity = AudioIdentity(hashlib.sha256(pcm).hexdigest(), len(pcm) // 2)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, identity


def decode_pcm16(pcm: bytes, sample_rate: int) -> tuple[np.ndarray, AudioIdentity]:
    """Decode mono PCM16 and retain a complete identity at its native rate."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(pcm) % 2:
        raise ValueError("PCM16 byte length must be even")
    fmt = CANONICAL_FORMAT if sample_rate == CANONICAL_RATE else f"pcm_s16le_mono_{sample_rate}"
    identity = AudioIdentity(hashlib.sha256(pcm).hexdigest(), len(pcm) // 2, sample_rate, fmt)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, identity


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    except OSError:
        pass


def write_canonical_wav(path: Path, pcm: bytes) -> None:
    write_pcm16_wav(path, pcm, CANONICAL_RATE)


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with wave.open(str(temp), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    os.chmod(temp, 0o600)
    # r+b, not rb: Windows rejects fsync on a read-only handle (EBADF).
    with open(temp, "r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temp, path)
    os.chmod(path, 0o600)
    _fsync_dir(path.parent)


def read_canonical_wav(path: Path) -> tuple[np.ndarray, AudioIdentity]:
    with wave.open(str(path), "rb") as wav:
        if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                != (1, 2, CANONICAL_RATE, "NONE")):
            raise ValueError("not canonical 16 kHz mono PCM16 WAV")
        pcm = wav.readframes(wav.getnframes())
    return decode_canonical(pcm)


def read_pcm16_wav(path: Path) -> tuple[np.ndarray, AudioIdentity]:
    """Read a mono PCM16 WAV, preserving its native sample rate in identity."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (1, 2, "NONE"):
            raise ValueError("not mono PCM16 WAV")
        sample_rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    return decode_pcm16(pcm, sample_rate)
