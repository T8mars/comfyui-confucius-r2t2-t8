"""Pin and prepare the 20 human-recorded TaiMECS code-switching clips locally.

Run with the separate BF16 Python environment, which inherits scipy/OpenCC from
the host read-only. Dataset audio and manifests stay under ignored .runtime/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".runtime/evaluation/taimecs"
REVISION = "83f397e41840ba187cc6833e1320bd2e5fa858f1"
METADATA_SHA256 = "bf2f84a5fae2c5c233d00397d01e98944de5a4ce827a447c9c522b10a8a27f3b"
BASE_URL = f"https://huggingface.co/datasets/JacobLinCool/TaiMECS/resolve/{REVISION}/"
SOURCE_SHA256 = {
    "001.mp3": "fb2b2c7f72e1d9536a9755140977ad848fb0dd5dcc330d8e1e55f257bb432fae",
    "002.mp3": "fdace646ca489d65fc7c1ff42613e8008be6483895300381dfdc4bb8996d63d1",
    "003.mp3": "085afcb96432ddff7dc1c2ba88c5effc5f2c810e800c361b5ed7fcd0ab472440",
    "004.mp3": "9addf5701d0184314641eedee3316da9e27931dd1848b630e9a694afb56c7d4f",
    "005.mp3": "352febe49028664ec25df8472cca28abcbf5ee72bea34d2cb2f5abaa5aa30556",
    "006.mp3": "01c6285c3c1a2a5f4b2d6cf24de958bb555d9611fea507372c156b9e1b5e55db",
    "007.mp3": "d456b8559b4e3ead080e4d757221972f9730557b4e8309b13af0fe2fdd3caccf",
    "008.mp3": "8ad2aecde5b33bc6da1ac4f38ce7d93200f4f4d0b2e3cc32896aa79a163db375",
    "009.mp3": "aa6854c0c5deadec84f1afa69765045bd322fa95517bd8cd587c98c8ba829752",
    "010.mp3": "b890b83010a5487adf5ad138fbab32f11350512421fa306881f58ef9423fdeef",
    "011.mp3": "d49198c5bdb08e9b638209dc8ac6684983a3471079bd57c8a3301d7f4fd0ceda",
    "012.mp3": "432077b9e409055d684f5d0896258153de275399ff65641d60c2e7907043f0e2",
    "015.mp3": "3d1a2bc6d3805e772ed3dc93b55e66a079013075fa3372486f6155f9837bb7e8",
    "017.mp3": "bec2289b2cb7a9c3409b15849593de92974b137565cca58d0ad4751bc3391131",
    "019.mp3": "34c0b33b03f870d3685912f57aec682423a36a372f1205784d5ae2b521e08251",
    "020.mp3": "a9543d19ff0d0b97ca2bf76f37b66117cd518fe4931e95fee820f7cb67fe7b72",
    "021.mp3": "a21fa8cf9b639612aeca3d5f163447f2c7b2f755aa95d765c287e67f384ec966",
    "023.mp3": "96c8963095bfa6edf7e486a06ae9a8935b733bad17e6c84c2e4d57064c125ba8",
    "024.mp3": "d2fdb621cb6271849d67c0bc53c19658d35dbfbcbcdba9e9eb20d4ad9fef5620",
    "026.mp3": "46570b4db11f5e465140b44e52f760b06b368916c84d355c097fdf678a21b30e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, target: Path) -> None:
    if target.is_file():
        return
    request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "r2t2-evaluation/1"})
    temporary = target.with_name(target.name + ".tmp")
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language-hint", choices=("Auto", "Chinese"), default="Auto")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    metadata = DEST / "metadata.jsonl"
    download("metadata.jsonl", metadata)
    if sha256(metadata) != METADATA_SHA256:
        raise ValueError("Pinned TaiMECS metadata SHA-256 mismatch")
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    human = [row for row in rows if row["source"] == "human"]
    if len(rows) != 100 or len(human) != 20 or len({row["file_name"] for row in human}) != 20:
        raise ValueError("TaiMECS human clip selection changed")
    manifest = []
    for row in human:
        name = row["file_name"]
        if not re.fullmatch(r"\d{3}\.mp3", name):
            raise ValueError(f"Unexpected TaiMECS filename: {name}")
        source = DEST / name
        download(name, source)
        source_hash = sha256(source)
        if source_hash != SOURCE_SHA256[name]:
            raise ValueError(f"Pinned TaiMECS MP3 SHA-256 mismatch: {name}")
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        if rate != 48000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Unexpected TaiMECS audio format: {name}")
        pcm = resample_poly(audio[:, 0], 1, 3).astype(np.float32)
        target = DEST / "wav16k" / name.replace(".mp3", ".wav")
        target.parent.mkdir(exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            # libsndfile FLOAT WAV includes a time-varying PEAK timestamp;
            # PCM_24 keeps resampling output and manifest hashes reproducible.
            sf.write(temporary, pcm, 16000, format="WAV", subtype="PCM_24")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        manifest.append({"id": f"taimecs_human_{name[:-4]}",
                         "audio_path": str(target.relative_to(DEST)).replace("\\", "/"),
                         "sha256": sha256(target), "reference": row["text"],
                         "language": "Mixed", "metric": "cer", "language_hint": args.language_hint,
                         "tags": ["mixed", "human", "natural_code_switch"],
                         "source_mp3": name, "source_mp3_sha256": source_hash,
                         "source_revision": REVISION})
    output = DEST / ("human-20-chinese.jsonl" if args.language_hint == "Chinese" else "human-20.jsonl")
    temporary = output.with_name(output.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    seconds = sum(sf.info(DEST / row["audio_path"]).duration for row in manifest)
    print(json.dumps({"manifest": str(output), "human_clips": len(manifest),
                      "audio_seconds": round(seconds, 3), "revision": REVISION,
                      "language_hint": args.language_hint}))


if __name__ == "__main__":
    main()
