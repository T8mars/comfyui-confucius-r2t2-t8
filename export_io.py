"""Publish UTF-8 export bytes without replacing an existing file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_no_overwrite(destination: Path, data: bytes) -> Path:
    destination = Path(destination)
    if destination.is_symlink():
        raise FileExistsError(f"Refusing to follow an output symlink: {destination}")
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError(f"Different content already exists: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".r2t2-", suffix=".tmp", dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Windows rename refuses an existing target and works without
            # hard links (e.g. exFAT). POSIX rename can replace the target.
            if os.name == "nt":
                os.rename(temp_path, destination)
            else:
                os.link(temp_path, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != data:
                raise FileExistsError(f"Different content already exists: {destination}")
    finally:
        temp_path.unlink(missing_ok=True)
    return destination
