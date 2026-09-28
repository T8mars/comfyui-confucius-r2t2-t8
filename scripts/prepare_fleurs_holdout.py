"""Prepare disjoint FLEURS test read-speech cases after the development set."""

from __future__ import annotations

import json
import os
import soundfile as sf

from prepare_fleurs_eval import DEST, SOURCES, extract_selected, selected_rows, sha256

COUNT = 100
START_UNIQUE_INDEX = 120


def main() -> None:
    development = DEST / "fleurs-test-100-each-plus-20-challenge.jsonl"
    used_cases = [json.loads(line) for line in development.read_text(encoding="utf-8").splitlines()]
    used_names = {name for item in used_cases for name in
                  item.get("source_audio", []) + [item["audio_path"].split("/")[-1]]}
    used_references = {item["reference"] for item in used_cases}
    selected = []
    for config, source in SOURCES.items():
        tsv = DEST / "source" / f"{config}-test.tsv"
        # Mixed development cases store the source filenames, so recover their
        # individual references and keep all source utterances disjoint.
        for line in tsv.read_text(encoding="utf-8").splitlines():
            columns = line.split("\t")
            if columns[1] in used_names:
                used_references.add(columns[2])
        pool = selected_rows(tsv, 300)[START_UNIQUE_INDEX:]
        rows = [row for row in pool if row["filename"] not in used_names and
                row["reference"] not in used_references][:COUNT]
        if len(rows) != COUNT:
            raise ValueError(f"Not enough disjoint {config} holdout utterances")
        files = extract_selected(DEST / "source" / f"{config}-test.tar.gz",
                                 DEST / "holdout" / config, rows)
        for row in rows:
            path = files[row["filename"]]
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or abs(info.frames - row["source_samples"]) > 1:
                raise ValueError(f"Invalid holdout audio: {path}")
            selected.append({"id": f"holdout_{config}_{path.stem}",
                             "audio_path": str(path.relative_to(DEST)),
                             "sha256": sha256(path), "reference": row["reference"],
                             "language": source["language"], "metric": source["metric"],
                             "language_hint": "Auto", "tags": []})
            used_names.add(row["filename"])
            used_references.add(row["reference"])
    target = DEST / "fleurs-holdout-100-each.jsonl"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    print(json.dumps({"manifest": str(target), "cases": len(selected),
                      "audio_seconds": sum(sf.info(DEST / item["audio_path"]).duration for item in selected)}))


if __name__ == "__main__":
    main()
