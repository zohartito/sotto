"""One strict offline policy for local model and probe subprocesses."""
from __future__ import annotations

import os
from collections.abc import Mapping


OFFLINE_FLAGS=("SOTTO_OFFLINE","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE")


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1","true","yes","on"}


def offline_requested(environ: Mapping[str,str] | None = None) -> bool:
    """Any explicit offline flag fences download-capable paths."""
    source=os.environ if environ is None else environ
    return any(_truthy(source.get(name)) for name in OFFLINE_FLAGS)


def offline_subprocess_env(*, path: str = "/usr/bin:/bin") -> dict[str,str]:
    """Minimal inherited-free environment for offline local subprocesses."""
    if not isinstance(path,str) or not path.startswith("/"):
        raise ValueError("offline subprocess PATH must be absolute")
    return {"PATH":path,**{name:"1" for name in OFFLINE_FLAGS}}
