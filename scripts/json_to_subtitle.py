"""Convert a saved r2t2 transcript JSON into SRT or WebVTT.

The per-segment text is only present in results written by node versions that
record it, so a JSON whose segments carry no "text" field cannot be subdivided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SAMPLE_RATE = 16_000


def cue_timestamp(seconds: float, kind: str) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    whole, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{',' if kind == 'srt' else '.'}{millis:03d}"


def render(result: dict, kind: str) -> str:
    cues = []
    for segment in result.get("segments", []):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = segment["start_sample"] / SAMPLE_RATE
        end = segment["end_sample"] / SAMPLE_RATE
        cues.append(f"{cue_timestamp(start, kind)} --> {cue_timestamp(end, kind)}\n{text}")
    if kind == "vtt":
        return "WEBVTT\n\n" + "\n\n".join(cues) + "\n" if cues else "WEBVTT\n"
    return "".join(f"{index}\n{cue}\n\n" for index, cue in enumerate(cues, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="transcript JSON produced by the node")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="destination file; defaults to the source path with the new suffix")
    parser.add_argument("--format", choices=("srt", "vtt"), default="srt")
    args = parser.parse_args()

    result = json.loads(args.source.read_text(encoding="utf-8"))
    segments = result.get("segments", [])
    if segments and not any(segment.get("text") for segment in segments):
        print("This transcript has segment boundaries but no per-segment text. Re-run the "
              "workflow with the current node version to capture it.", file=sys.stderr)
        return 1
    if not segments:
        print("This transcript carries no timed segments, so no cues can be produced.",
              file=sys.stderr)
        return 1

    destination = args.output or args.source.with_suffix(f".{args.format}")
    destination.write_text(render(result, args.format), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
