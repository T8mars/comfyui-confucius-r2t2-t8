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

from subtitles import build_subtitles, render_subtitle
from export_io import atomic_write_no_overwrite


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a saved r2t2 transcript into subtitles")
    parser.add_argument("source", type=Path, help="transcript JSON produced by the node")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="destination file; defaults to the source path with the new suffix")
    parser.add_argument("--format", choices=("srt", "vtt"), default="srt")
    parser.add_argument("--offset-ms", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--whole-audio-draft", action="store_true")
    args = parser.parse_args()

    try:
        result = json.loads(args.source.read_text(encoding="utf-8-sig"))
        document = build_subtitles(result, offset_ms=args.offset_ms,
                                   allow_partial=args.allow_partial,
                                   whole_audio_draft=args.whole_audio_draft)
        content = render_subtitle(document, args.format)
        if not document["cue_count"]:
            print("No subtitles to export (empty); no file was written")
            return 0
        destination = args.output or args.source.with_suffix(f".{args.format}")
        atomic_write_no_overwrite(destination, content.encode("utf-8"))
        print(f"{destination} ({document['cue_count']} cues; {document['subtitle_status']})")
        if document["warnings"]:
            print(", ".join(document["warnings"]), file=sys.stderr)
        return 0
    except (ValueError, OSError) as exc:
        print(f"Subtitle export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
