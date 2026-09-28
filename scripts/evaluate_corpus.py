"""Score a private, hash-pinned ASR corpus with the isolated official Q8 worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bridge import manager  # noqa: E402
from r2t2_core.evaluation import BF16_MODEL_SHA256, aggregate, quality_gate, score  # noqa: E402

CONFIG = {"n_ctx": 8192, "n_batch": 1024, "n_threads": 8, "gpu_layers": -1}
LANGUAGES = {"Chinese": "cer", "English": "wer", "Mixed": "cer"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(items: list[dict], manifest: Path) -> list[dict]:
    if not items:
        raise ValueError("Evaluation manifest is empty")
    seen = set()
    seen_audio = set()
    checked = []
    for item in items:
        identifier = item["id"]
        language = item["language"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate case id: {identifier!r}")
        seen.add(identifier)
        if language not in LANGUAGES or item["metric"] != LANGUAGES[language]:
            raise ValueError(f"Metric/language mismatch in {identifier}")
        if not isinstance(item["reference"], str):
            raise ValueError(f"Reference must be text in {identifier}")
        score(item["reference"], item["reference"], item["metric"])
        if not isinstance(item.get("tags", []), list):
            raise ValueError(f"Tags must be a list in {identifier}")
        hint = item.get("language_hint", "Auto")
        if hint not in ("Auto", "Chinese", "English"):
            raise ValueError(f"Invalid language hint in {identifier}")
        path = Path(item["audio_path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve(strict=True)
        expected_hash = item["sha256"].lower()
        if len(expected_hash) != 64 or file_sha256(path) != expected_hash:
            raise ValueError(f"Audio SHA-256 mismatch in {identifier}")
        if expected_hash in seen_audio:
            raise ValueError(f"Duplicate audio SHA-256 in {identifier}")
        seen_audio.add(expected_hash)
        checked.append({**item, "audio_path": str(path), "language_hint": hint})
    return checked


def validate_baseline(entries: list[dict], path: Path) -> dict[str, dict]:
    """Reject a stale or mispaired BF16 baseline before Q8 inference starts."""
    expected = {item["id"]: item["sha256"].lower() for item in entries}
    baseline = {}
    for item in read_jsonl(path):
        identifier = item["id"]
        if identifier in baseline:
            raise ValueError(f"Duplicate BF16 case id: {identifier}")
        if identifier not in expected:
            raise ValueError(f"Unknown BF16 case id: {identifier}")
        if item.get("model_sha256") != BF16_MODEL_SHA256 or not isinstance(item.get("text"), str):
            raise ValueError(f"BF16 model identity/text missing in {identifier}")
        if item.get("audio_sha256", "").lower() != expected[identifier]:
            raise ValueError(f"BF16 audio hash mismatch in {identifier}")
        baseline[identifier] = item
    return baseline


def write_report(path: Path, report: dict) -> None:
    root = (ROOT / ".runtime").resolve()
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evaluation reports must stay in the Git-ignored .runtime directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".eval-", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Private JSONL with id, audio_path, sha256, reference, language, metric, tags")
    parser.add_argument("--mode", choices=("offline", "stream"), default="offline")
    parser.add_argument("--stream-chunk-ms", type=int, choices=(160, 320, 480, 640), default=160,
                        help="File-stream decode interval; only valid with --mode stream")
    parser.add_argument("--bf16", type=Path, help="Optional JSONL of {id, text, audio_sha256, model_sha256} BF16 offline predictions")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/evaluation/report.json")
    parser.add_argument("--require-pass", action="store_true", help="Exit nonzero unless the full quality gate passes")
    args = parser.parse_args()
    if args.mode != "stream" and args.stream_chunk_ms != 160:
        parser.error("--stream-chunk-ms applies only to --mode stream")

    entries = validate_manifest(read_jsonl(args.manifest), args.manifest.resolve())
    baseline = validate_baseline(entries, args.bf16) if args.bf16 else {}
    partial = args.output.with_name(args.output.stem + ".partial.json")
    manifest_hash = file_sha256(args.manifest)
    baseline_hash = file_sha256(args.bf16) if args.bf16 else None

    cases = []
    total_seconds = 0.0
    worker_failures = 0
    try:
        loaded = manager.load(CONFIG)
        identity = {"manifest_sha256": manifest_hash, "bf16_sha256": baseline_hash,
                    "mode": args.mode, "stream_chunk_ms": args.stream_chunk_ms,
                    "model_sha256": loaded["model_sha256"],
                    "projector_sha256": loaded["projector_sha256"], "build_id": loaded["build_id"]}
        if partial.exists():
            checkpoint = json.loads(partial.read_text(encoding="utf-8"))
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                raise ValueError("Evaluation checkpoint does not match manifest, baseline, mode or model")
            cases = checkpoint["cases"]
            worker_failures = checkpoint.get("worker_failures", 0)
            if len(cases) > len(entries) or any(case["id"] != item["id"] or
                                                  case["audio_sha256"].lower() != item["sha256"].lower()
                                                  for case, item in zip(cases, entries)):
                raise ValueError("Evaluation checkpoint case order or audio hash mismatch")
            total_seconds = sum(case["audio_seconds"] for case in cases)
            print(json.dumps({"resumed": len(cases), "total": len(entries),
                              "prior_worker_failures": worker_failures}), flush=True)

        for item in entries[len(cases):]:
            audio, rate = sf.read(item["audio_path"], dtype="float32", always_2d=True)
            if audio.shape[0] == 0 or audio.shape[1] < 1 or audio.shape[1] > 8:
                raise ValueError(f"Invalid audio shape in {item['id']}")
            pcm = np.ascontiguousarray(audio, dtype="<f4")
            try:
                result = manager.transcribe(pcm.tobytes(), {
                    "sample_rate": rate, "channels": audio.shape[1], "channel": "mean",
                    "mode": args.mode, "stream_chunk_ms": args.stream_chunk_ms,
                    "language": item["language_hint"],
                    "context": "", "hotwords": ""}, CONFIG)
            except Exception:
                worker_failures += 1
                write_report(partial, {**identity, "cases": cases,
                                       "worker_failures": worker_failures})
                raise
            metric = item["metric"]
            case = {"id": item["id"], "language": item["language"], "metric": metric,
                    "tags": item.get("tags", []), "audio_sha256": item["sha256"],
                    "audio_seconds": result["audio_seconds"], "reference": item["reference"],
                    "q8_text": result["text"], "detected_language": result["language"],
                    "input_gain": result.get("input_gain", 1.0),
                    "stream_chunk_ms": result.get("stream_chunk_ms", args.stream_chunk_ms),
                    "status": result["status"], "mode_executed": result.get("mode_executed", args.mode),
                    "q8_score": score(item["reference"], result["text"], metric)}
            if item["id"] in baseline:
                prediction = baseline[item["id"]]
                case["bf16_text"] = prediction["text"]
                case["bf16_score"] = score(item["reference"], prediction["text"], metric)
            total_seconds += result["audio_seconds"]
            cases.append(case)
            write_report(partial, {**identity, "cases": cases,
                                   "worker_failures": worker_failures})
            if len(cases) % 10 == 0:
                print(json.dumps({"processed": len(cases), "total": len(entries)}), flush=True)
        groups = {language: aggregate([case for case in cases if case["language"] == language])
                  for language in LANGUAGES}
        gate = quality_gate(cases, mode=args.mode, audio_seconds=total_seconds)
        if worker_failures and gate["status"] == "pass":
            gate = {**gate, "status": "worker_instability",
                    "message": "Transport failed during this paired run; investigate before release"}
        report = {"schema_version": 1, "mode": args.mode,
                  "stream_chunk_ms": args.stream_chunk_ms,
                  "manifest_sha256": manifest_hash, "bf16_input_sha256": baseline_hash,
                  "model_sha256": loaded["model_sha256"],
                  "projector_sha256": loaded["projector_sha256"], "build_id": loaded["build_id"],
                  "bf16_model_sha256": BF16_MODEL_SHA256 if args.bf16 else None,
                  "worker_failures": worker_failures,
                  "normalization": "NFKC+casefold; CER keeps Unicode letters/numbers; WER matches ASCII words/numbers",
                  "groups": groups, "quality_gate": gate, "cases": cases}
        write_report(args.output, report)
        partial.unlink(missing_ok=True)
        print(json.dumps({"report": str(args.output.resolve()), "cases": len(cases),
                          "groups": groups, "quality_gate": report["quality_gate"]["status"]},
                         ensure_ascii=True))
        if args.require_pass and report["quality_gate"]["status"] != "pass":
            raise SystemExit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
