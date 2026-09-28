"""Prepare a pinned, local-only FLEURS test subset for ASR comparison.

The public FLEURS test audio and transcripts are CC BY 4.0. This script keeps
archives, clips, and the corpus manifest under Git-ignored .runtime only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".runtime/evaluation/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
SOURCE = "https://huggingface.co/datasets/google/fleurs/resolve/" + REVISION
SOURCES = {
    "cmn_hans_cn": {
        "language": "Chinese", "metric": "cer",
        "tsv_size": 491487,
        "tsv_sha256": "5734461648f816181d7dab5fc79204b18c4b9bc2cd5138225b25c72d18385d21",
        "tar_size": 525346466,
        "tar_sha256": "09d19ad18f5d7e91076880807e866cd16abd924c7052b55f71cdae91714fc166",
    },
    "en_us": {
        "language": "English", "metric": "wer",
        "tsv_size": 367864,
        "tsv_sha256": "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca",
        "tar_size": 289851356,
        "tar_sha256": "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(relative: str, path: Path, size: int, digest: str) -> None:
    if path.exists() and path.stat().st_size == size and sha256(path) == digest:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    if partial.exists() and partial.stat().st_size >= size:
        partial.unlink()
    url = SOURCE + "/" + relative
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    offset = 0
                with partial.open("ab" if offset else "wb") as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            if partial.stat().st_size == size and sha256(partial) == digest:
                os.replace(partial, path)
                return
            if partial.stat().st_size >= size:
                partial.unlink()
        except (OSError, requests.RequestException) as exc:
            print(json.dumps({"retry": attempt + 1, "file": path.name, "error": str(exc)}), flush=True)
    raise RuntimeError(f"Download or SHA-256 verification failed: {path.name}")


def selected_rows(tsv: Path, count: int) -> list[dict]:
    rows = []
    seen_text = set()
    for line in tsv.read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if len(columns) != 7:
            raise ValueError(f"Unexpected FLEURS TSV shape in {tsv.name}")
        _, filename, reference, _, _, source_samples, _ = columns
        if reference in seen_text:
            continue
        if not re.fullmatch(r"[0-9]+\.wav", filename):
            raise ValueError(f"Unexpected FLEURS audio filename: {filename}")
        seen_text.add(reference)
        rows.append({"filename": filename, "reference": reference,
                     "source_samples": int(source_samples)})
        if len(rows) == count:
            return rows
    raise ValueError(f"Only {len(rows)} distinct test references in {tsv.name}")


def extract_selected(archive: Path, clips: Path, rows: list[dict]) -> dict[str, Path]:
    clips.mkdir(parents=True, exist_ok=True)
    wanted = {row["filename"] for row in rows}
    found = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            filename = Path(member.name).name
            if filename not in wanted or not member.isfile():
                continue
            if filename in found:
                raise ValueError(f"Duplicate archive member: {filename}")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable archive member: {filename}")
            target = clips / filename
            descriptor, temporary_name = tempfile.mkstemp(prefix=".fleurs-", suffix=".wav", dir=clips)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output, source:
                    shutil.copyfileobj(source, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            found[filename] = target
    if found.keys() != wanted:
        raise ValueError(f"FLEURS archive missing {sorted(wanted - found.keys())[:5]}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-language", type=int, default=100)
    args = parser.parse_args()
    if not 50 <= args.per_language <= 300:
        raise ValueError("Choose 50-300 unique test sentences per language")
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    duration_samples = 0
    for config, source in SOURCES.items():
        source_dir = DEST / "source"
        tsv = source_dir / f"{config}-test.tsv"
        archive = source_dir / f"{config}-test.tar.gz"
        fetch(f"data/{config}/test.tsv", tsv, source["tsv_size"], source["tsv_sha256"])
        rows = selected_rows(tsv, args.per_language)
        print(json.dumps({"source": config, "selected": len(rows), "archive_bytes": source["tar_size"]}), flush=True)
        fetch(f"data/{config}/audio/test.tar.gz", archive, source["tar_size"], source["tar_sha256"])
        files = extract_selected(archive, DEST / "clips" / config, rows)
        for row in rows:
            audio = files[row["filename"]]
            info = sf.info(audio)
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"Unexpected FLEURS audio format: {audio}")
            samples = info.frames
            if abs(samples - row["source_samples"]) > 1:
                raise ValueError(f"FLEURS audio duration mismatch: {audio}")
            duration_samples += samples
            manifest.append({"id": f"fleurs_{config}_{audio.stem}",
                             "audio_path": str(audio.relative_to(DEST)),
                             "sha256": sha256(audio), "reference": row["reference"],
                             "language": source["language"], "metric": source["metric"],
                             "language_hint": "Auto", "tags": []})
    target = DEST / f"fleurs-test-{args.per_language}-each.jsonl"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temporary, target)
    print(json.dumps({"manifest": str(target), "cases": len(manifest),
                      "audio_seconds": duration_samples / 16000}, ensure_ascii=True))


if __name__ == "__main__":
    main()
