"""Build 20 reproducible, explicitly tagged FLEURS challenge diagnostics.

Mixed audio is concatenated speech, quiet/noise are controlled transforms, and
short means an unmodified recorded utterance of at most 4-5 seconds. These
are diagnostics, not a substitute for naturally occurring code switching.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare_fleurs_eval import DEST, SOURCES, extract_selected, selected_rows, sha256

SEED = 20260928


def read_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32")
    if rate != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected 16 kHz mono FLEURS audio: {path}")
    return audio


def record(identifier: str, path: Path, reference: str, language: str,
           tags: list[str], sources: list[str]) -> dict:
    return {"id": identifier, "audio_path": str(path.relative_to(DEST)),
            "sha256": sha256(path), "reference": reference, "language": language,
            "metric": "wer" if language == "English" else "cer",
            "language_hint": "Auto", "tags": tags,
            "source_audio": sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    base_path = DEST / ("fleurs-holdout-100-each.jsonl" if args.holdout else
                        "fleurs-test-100-each.jsonl")
    base = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines()]
    if len(base) != 200:
        raise ValueError("Prepare 100 clean cases per language first")
    used_cases = list(base)
    if args.holdout:
        development = DEST / "fleurs-test-100-each-plus-20-challenge.jsonl"
        used_cases += [json.loads(line) for line in development.read_text(encoding="utf-8").splitlines()]
    used_names = {name for item in used_cases for name in
                  item.get("source_audio", []) + [Path(item["audio_path"]).name]}
    used_refs = {item["reference"] for item in used_cases}
    for config in SOURCES:
        for line in (DEST / "source" / f"{config}-test.tsv").read_text(encoding="utf-8").splitlines():
            columns = line.split("\t")
            if columns[1] in used_names:
                used_refs.add(columns[2])
    short_limit = 5 if args.holdout else 4
    short_rows = []
    seen_short_refs = set()
    for line in (DEST / "source/en_us-test.tsv").read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        filename, reference, samples = columns[1], columns[2], int(columns[5])
        if (samples <= short_limit * 16000 and reference not in used_refs and
                filename not in used_names and reference not in seen_short_refs):
            short_rows.append({"filename": filename, "reference": reference,
                               "source_samples": samples})
            seen_short_refs.add(reference)
    if len(short_rows) < 5:
        raise ValueError("Fewer than five held-out short FLEURS utterances")
    short_rows = short_rows[:5]
    reserved = {row["reference"] for row in short_rows}
    candidates = {}
    for config in SOURCES:
        rows = selected_rows(DEST / "source" / f"{config}-test.tsv", 330)[240 if args.holdout else 100:]
        candidates[config] = [row for row in rows if row["reference"] not in reserved and
                              row["reference"] not in used_refs and row["filename"] not in used_names][:15]
        if len(candidates[config]) < 10:
            raise ValueError(f"Not enough held-out {config} challenge rows")
    mixed_pairs = [(zh, en) for zh, en in zip(candidates["cmn_hans_cn"], candidates["en_us"])
                   if zh["source_samples"] + en["source_samples"] + 3200 <= 30 * 16000][:5]
    if len(mixed_pairs) != 5:
        raise ValueError("Not enough mixed source pairs under 30 seconds")
    mixed_zh = {row["filename"] for row, _ in mixed_pairs}
    mixed_en = {row["filename"] for _, row in mixed_pairs}
    remaining_zh = [row for row in candidates["cmn_hans_cn"] if row["filename"] not in mixed_zh]
    remaining_en = [row for row in candidates["en_us"] if row["filename"] not in mixed_en]
    if len(remaining_zh) < 5 or len(remaining_en) < 5:
        raise ValueError("Not enough disjoint quiet/noise challenge sources")
    quiet = [("cmn_hans_cn", row) for row in remaining_zh[:3]] + [
        ("en_us", row) for row in remaining_en[:2]]
    noisy = [("cmn_hans_cn", row) for row in remaining_zh[3:5]] + [
        ("en_us", row) for row in remaining_en[2:5]]
    wanted = {"cmn_hans_cn": [row for row, _ in mixed_pairs] + remaining_zh[:5],
              "en_us": [row for _, row in mixed_pairs] + remaining_en[:5] + short_rows}
    files = {config: extract_selected(DEST / "source" / f"{config}-test.tar.gz",
                                      DEST / ("holdout-challenge-sources" if args.holdout else "clips") / config,
                                      rows)
             for config, rows in wanted.items()}
    challenge_dir = DEST / ("holdout-challenge" if args.holdout else "challenge")
    challenge_dir.mkdir(parents=True, exist_ok=True)
    challenges = []
    prefix = "holdout_challenge" if args.holdout else "challenge"
    for index, (zh, en) in enumerate(mixed_pairs):
        left = read_audio(files["cmn_hans_cn"][zh["filename"]])
        right = read_audio(files["en_us"][en["filename"]])
        mixed = np.concatenate((left, np.zeros(3200, dtype=np.float32), right))
        if mixed.size > 30 * 16000:
            raise ValueError("Synthetic mixed case exceeds offline segment limit")
        path = challenge_dir / f"mixed-{index:02d}.wav"
        sf.write(path, mixed, 16000, subtype="FLOAT")
        challenges.append(record(f"{prefix}_mixed_{index:02d}", path,
                                 zh["reference"] + " " + en["reference"],
                                 "Mixed", ["mixed", "synthetic"],
                                 [zh["filename"], en["filename"]]))

    for index, (config, row) in enumerate(quiet):
        audio = read_audio(files[config][row["filename"]])
        path = challenge_dir / f"quiet-{index:02d}.wav"
        sf.write(path, audio * np.float32(10 ** (-24 / 20)), 16000, subtype="FLOAT")
        challenges.append(record(f"{prefix}_quiet_{index:02d}", path,
                                 row["reference"], SOURCES[config]["language"],
                                 ["quiet", "synthetic"], [row["filename"]]))

    for index, (config, row) in enumerate(noisy):
        audio = read_audio(files[config][row["filename"]])
        rng = np.random.default_rng(SEED + index)
        noise = rng.normal(size=audio.size).astype(np.float32)
        signal_rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2))
        if signal_rms == 0 or noise_rms == 0:
            raise ValueError("Noise challenge requires nonzero audio")
        combined = audio + noise * np.float32(signal_rms / (noise_rms * 10 ** (10 / 20)))
        peak = float(np.max(np.abs(combined)))
        if peak > 1:
            combined /= peak
        path = challenge_dir / f"noise-{index:02d}.wav"
        sf.write(path, combined, 16000, subtype="FLOAT")
        challenges.append(record(f"{prefix}_noise_{index:02d}", path,
                                 row["reference"], SOURCES[config]["language"],
                                 ["noise", "synthetic"], [row["filename"]]))

    for index, row in enumerate(short_rows):
        path = files["en_us"][row["filename"]]
        if sf.info(path).duration > short_limit:
            raise ValueError("Short challenge exceeds the fixed duration limit")
        challenges.append(record(f"{prefix}_short_{index:02d}", path,
                                 row["reference"], "English", ["short"],
                                 [row["filename"]]))
    if len(challenges) != 20:
        raise AssertionError("Expected exactly 20 challenge cases")
    target = DEST / ("fleurs-holdout-100-each-plus-20-challenge.jsonl" if args.holdout else
                     "fleurs-test-100-each-plus-20-challenge.jsonl")
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in base + challenges:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    print(json.dumps({"manifest": str(target), "base": len(base),
                      "challenge": len(challenges), "tags": [item["tags"] for item in challenges]}))


if __name__ == "__main__":
    main()
