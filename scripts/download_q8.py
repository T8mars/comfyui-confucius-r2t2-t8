"""Download the pinned official Q8 pair and its licenses from the T8 mirror."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_core.native import MODEL_NAME, PROJECTOR_NAME, OFFICIAL, verify_pair  # noqa: E402

REPO = "t8star/Confucius-R2t2-Comfy"
REVISION = "2223a55593a85bbaf2b12d58c410da67a8822608"
DEST = ROOT / "models/Confucius4-R2T2-GGUF"
FILES = {
    **OFFICIAL,
    "MODEL_LICENSE": (11_132, "064483c5ba1dc20907038108da4f45d5b37cd0a41a351c8a8bb98ba1af48c505"),
    "MODEL_LICENSE_zh": (7_789, "6ed2e9946bd0b0185a6c690565cf2f8de23044a513be6e1ec7c97e6995ce241b"),
}


def download(name: str) -> None:
    expected_size, expected_sha = FILES[name]
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    if target.exists() and target.stat().st_size == expected_size:
        digest = file_hash(target)
        if digest == expected_sha:
            print(f"Verified existing {name}")
            return
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        with partial.open("ab" if offset else "wb") as out:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    out.write(chunk)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download: {name}: {partial.stat().st_size}/{expected_size}")
    if file_hash(partial) != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch: {name}; partial retained for diagnosis")
    partial.replace(target)
    print(f"Verified {name}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    for filename in (MODEL_NAME, PROJECTOR_NAME, "MODEL_LICENSE", "MODEL_LICENSE_zh"):
        download(filename)
    verify_pair(DEST)
    print("Official Q8 pair ready")
