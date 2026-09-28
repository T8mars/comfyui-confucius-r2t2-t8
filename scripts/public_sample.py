"""Retrieve the upstream project's public smoke-test WAV at a pinned commit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".runtime/public-test.wav"
URL = ("https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/"
       "26d55a54ce5670cff9947a167d8ed95d569fd4d9/resources/test.wav")
SIZE = 215_724
SHA256 = "b703174ef013bf7332ab74d4b65d568e2981cff2633cb620252bb4ba18ea7f67"


def valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != SIZE:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == SHA256


def ensure_sample() -> Path:
    if valid(DEST):
        return DEST
    DEST.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=(30, 120))
    response.raise_for_status()
    partial = DEST.with_name(DEST.name + ".partial")
    partial.write_bytes(response.content)
    if not valid(partial):
        raise RuntimeError("Official public test.wav checksum mismatch")
    partial.replace(DEST)
    return DEST


if __name__ == "__main__":
    print(ensure_sample())
