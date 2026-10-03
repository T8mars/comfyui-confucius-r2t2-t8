"""Convert a saved r2t2 transcript JSON into SRT or WebVTT.

Run this with the worker or ComfyUI interpreter; the renderer is shared with
the Save Subtitle node so both produce identical cues.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2t2_core.subtitle import render_subtitle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a saved r2t2 transcript into subtitles")
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
    if not render_subtitle(result, "srt").strip():
        print("This transcript carries no text to turn into cues.", file=sys.stderr)
        return 1

    destination = args.output or args.source.with_suffix(f".{args.format}")
    destination.write_text(render_subtitle(result, args.format), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
