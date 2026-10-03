"""SRT and WebVTT rendering for r2t2 transcripts."""

from __future__ import annotations

import math

SAMPLE_RATE = 16_000
MAX_CUE_SECONDS = 15.0
_CUE_BREAKS = frozenset("，。！？；：、,.!?;:")


def cue_timestamp(seconds: float, kind: str) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    whole, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{',' if kind == 'srt' else '.'}{millis:03d}"


def _split_text(text: str, parts: int) -> list[str]:
    """Cut at whichever punctuation sits closest to each even share of the text."""
    pieces: list[str] = []
    remaining = text
    for index in range(parts - 1):
        quota = len(text) * (index + 1) / parts - sum(len(piece) for piece in pieces)
        cut = None
        closest = None
        for position, char in enumerate(remaining):
            if char not in _CUE_BREAKS:
                continue
            distance = abs(position - quota)
            if closest is None or distance < closest:
                cut, closest = position + 1, distance
        if cut is None or cut >= len(remaining):
            break
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    pieces.append(remaining.strip())
    return [piece for piece in pieces if piece]


def _cues(start: float, end: float, text: str) -> list[tuple[float, float, str]]:
    """Split a span evenly so no cue runs longer than MAX_CUE_SECONDS.

    Time is always divided by the same share the text is, because a punctuation
    cut that yields fewer pieces would otherwise stretch the first cue past the
    limit.
    """
    duration = end - start
    if duration <= MAX_CUE_SECONDS:
        return [(start, end, text)]
    parts = math.ceil(duration / MAX_CUE_SECONDS)
    pieces = _split_text(text, parts)
    if len(pieces) != parts:
        width = len(text) / parts
        pieces = [text[round(i * width):round((i + 1) * width)].strip() for i in range(parts)]
    step = duration / parts
    return [(start + step * index,
             end if index == parts - 1 else min(start + step * (index + 1), end),
             piece) for index, piece in enumerate(pieces) if piece]


def render_subtitle(result: dict, kind: str) -> str:
    """Render a transcript as SRT or WebVTT.

    A segmented transcript yields one cue per segment, with long segments
    divided at punctuation so no cue outstays MAX_CUE_SECONDS. A file decoded
    in one pass carries no boundaries, so it becomes a single cue over the
    whole audio.
    """
    spans = []
    for segment in result.get("segments", []):
        text = (segment.get("text") or "").strip()
        if text:
            spans.append((segment["start_sample"] / SAMPLE_RATE,
                          segment["end_sample"] / SAMPLE_RATE, text))
    if not spans:
        text = (result.get("text") or "").strip()
        samples = result.get("audio_samples_16k")
        seconds = samples / SAMPLE_RATE if samples else result.get("audio_seconds")
        if text and seconds:
            spans.append((0.0, float(seconds), text))
    blocks = [f"{cue_timestamp(start, kind)} --> {cue_timestamp(end, kind)}\n{text}"
              for span in spans for start, end, text in _cues(*span)]
    if kind == "vtt":
        return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n" if blocks else "WEBVTT\n"
    return "".join(f"{index}\n{block}\n\n" for index, block in enumerate(blocks, 1))
