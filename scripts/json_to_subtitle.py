"""Convert a saved r2t2 transcript JSON into SRT or WebVTT.

Run this with the worker or ComfyUI interpreter; the renderer is shared with
the Save Subtitle node so both produce identical cues.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subtitles import build_subtitles, render_subtitle


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
        data = content.encode("utf-8")
        if destination.is_symlink():
            raise ValueError("Refusing to follow an output symlink")
        if destination.exists():
            if destination.read_bytes() != data:
                raise FileExistsError(f"Different content already exists: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix=".r2t2-", suffix=".tmp", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temp_name, destination)
                except FileExistsError:
                    if destination.is_symlink() or destination.read_bytes() != data:
                        raise FileExistsError(f"Different content already exists: {destination}")
            finally:
                Path(temp_name).unlink(missing_ok=True)
        print(f"{destination} ({document['cue_count']} cues; {document['subtitle_status']})")
        if document["warnings"]:
            print(", ".join(document["warnings"]), file=sys.stderr)
        return 0
    except (ValueError, OSError) as exc:
        print(f"Subtitle export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
