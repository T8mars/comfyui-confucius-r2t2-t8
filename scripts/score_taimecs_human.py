"""Score the human TaiMECS clips with Traditional/Simplified script alignment.

Run in the isolated BF16 environment (which inherits the existing OpenCC
installation read-only). This is a diagnostic, not the release quality gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2t2_core.evaluation import aggregate, paired_delta, score  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".runtime/evaluation/taimecs/normalized-human-20.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".runtime").resolve()):
        raise ValueError("TaiMECS report must stay in ignored .runtime")
    raw = json.loads(args.report.read_text(encoding="utf-8"))
    if raw.get("mode") not in ("offline", "stream") or len(raw["cases"]) != 20 or raw.get("worker_failures"):
        raise ValueError("Expected complete 20-case paired report without worker failures")
    converter = OpenCC("t2s")
    cases = []
    for row in raw["cases"]:
        if (row["status"] != "complete" or row["language"] != "Mixed" or
                "human" not in row["tags"] or not row["id"].startswith("taimecs_human_")):
            raise ValueError("Unexpected TaiMECS case identity or status")
        reference = converter.convert(row["reference"])
        q8_text = converter.convert(row["q8_text"])
        bf16_text = converter.convert(row["bf16_text"])
        cases.append({"id": row["id"], "q8_score": score(reference, q8_text, "cer"),
                      "bf16_score": score(reference, bf16_text, "cer")})
    report = {"dataset": "JacobLinCool/TaiMECS human only", "mode": raw["mode"],
              "normalization": "OpenCC t2s on both reference and predictions, then NFKC+casefold CER",
              "source_model_sha256": raw["model_sha256"],
              "bf16_model_sha256": raw["bf16_model_sha256"],
              "q8": aggregate(cases), "bf16": aggregate(cases, "bf16_score"),
              "paired_delta": paired_delta(cases), "cases": cases,
              "release_gate": "not_applicable_single_speaker_20_cases"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "q8": report["q8"], "bf16": report["bf16"],
                      "paired_delta": report["paired_delta"],
                      "release_gate": report["release_gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
