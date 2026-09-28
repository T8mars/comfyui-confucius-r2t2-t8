"""Fetch pinned FireRedVAD ONNX artifacts and Apache license from T8 mirror."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/FireRedVAD-ONNX"
REVISION = "2223a55593a85bbaf2b12d58c410da67a8822608"
BASE = f"https://huggingface.co/t8star/Confucius-R2t2-Comfy/resolve/{REVISION}/FireRedVAD-ONNX"
FILES = {
    "fireredvad_stream_vad_with_cache.onnx": (2_306_042, "b3c97836130dc34fc32d56fab551e88cf9454511de2b3c250a5e6578dee74b93"),
    "cmvn.ark": (1_311, "c87f6f13edf0f0ec7535ddfc9cc3387d9268cb234b70182d566c5e2edf3ca473"),
    "LICENSE": (11_357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, (size, sha) in FILES.items():
        target = DEST / name
        if target.exists() and target.stat().st_size == size and digest(target) == sha:
            print(f"Verified existing {name}")
            continue
        response = requests.get(f"{BASE}/{name}", timeout=(30, 120))
        response.raise_for_status()
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(response.content)
        if partial.stat().st_size != size or digest(partial) != sha:
            raise RuntimeError(f"FireRedVAD asset checksum mismatch: {name}")
        partial.replace(target)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
