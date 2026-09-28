"""Reference-backed ASR scoring; this module has no inference dependencies."""

from __future__ import annotations

import random
import re
import unicodedata

BF16_MODEL_SHA256 = "cc4d5324d386c80f98a8a7b09fbcdcc813ad08a6503fe3a586ebb144ec4610dc"
CHALLENGE_TAGS = ("mixed", "quiet", "noise", "short")
SEVERE_CHALLENGE_ERROR_RATE = 0.5


def units(text: str, metric: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if metric == "cer":
        return [char for char in normalized if unicodedata.category(char)[0] in ("L", "N")]
    if metric == "wer":
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalized)
    raise ValueError(f"Unknown metric: {metric}")


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def score(reference: str, hypothesis: str, metric: str) -> dict:
    expected = units(reference, metric)
    if not expected:
        raise ValueError("Reference has no scoreable units")
    actual = units(hypothesis, metric)
    return {"metric": metric, "errors": edit_distance(expected, actual),
            "reference_units": len(expected)}


def aggregate(cases: list[dict], score_key: str = "q8_score") -> dict:
    selected = [case[score_key] for case in cases]
    errors = sum(item["errors"] for item in selected)
    count = sum(item["reference_units"] for item in selected)
    return {"cases": len(selected), "errors": errors, "reference_units": count,
            "rate": errors / count if count else None}


def paired_delta(cases: list[dict], *, seed: int = 0, draws: int = 1000) -> dict:
    if not cases or any("bf16_score" not in case for case in cases):
        raise ValueError("Every paired case needs Q8 and BF16 scores")
    rng = random.Random(seed)

    def delta(selected: list[dict]) -> float:
        q8 = aggregate(selected, "q8_score")
        bf16 = aggregate(selected, "bf16_score")
        assert q8["reference_units"] == bf16["reference_units"]
        return q8["rate"] - bf16["rate"]

    samples = sorted(delta([rng.choice(cases) for _ in cases]) for _ in range(draws))
    return {"delta": delta(cases), "ci95_low": samples[int(0.025 * (draws - 1))],
            "ci95_high": samples[int(0.975 * (draws - 1))], "bootstrap_draws": draws}


def quality_gate(cases: list[dict], *, mode: str, audio_seconds: float) -> dict:
    groups = {language: [case for case in cases if case["language"] == language]
              for language in ("Chinese", "English", "Mixed")}
    challenge_by_tag = {tag: sum(tag in case.get("tags", []) for case in cases)
                        for tag in CHALLENGE_TAGS}
    challenge = sum(bool(set(case.get("tags", [])) & set(CHALLENGE_TAGS)) for case in cases)
    coverage = {"chinese": len(groups["Chinese"]), "english": len(groups["English"]),
                "challenge": challenge, "challenge_by_tag": challenge_by_tag,
                "audio_seconds": audio_seconds}
    if any(case.get("status") != "complete" for case in cases):
        return {"status": "incomplete_results", "coverage": coverage,
                "message": "Every Q8 case must finish complete before a quality claim"}
    ready = (mode == "offline" and len(groups["Chinese"]) >= 50 and
             len(groups["English"]) >= 50 and challenge >= 20 and
             all(count >= 5 for count in challenge_by_tag.values()) and audio_seconds >= 1800)
    if not ready:
        return {"status": "insufficient_corpus", "coverage": coverage,
                "criterion": "offline; >=50 Chinese, >=50 English, >=5 each challenge type, >=20 total challenge, >=1800 s"}
    if any("bf16_score" not in case for case in cases):
        return {"status": "missing_bf16_baseline", "coverage": coverage}
    comparisons = {}
    for language in ("Chinese", "English"):
        estimate = paired_delta(groups[language])
        estimate["status"] = ("pass" if estimate["ci95_high"] <= 0.01 else
                              "fail" if estimate["ci95_low"] > 0.01 else "inconclusive")
        comparisons[language] = estimate
    severe_cases = [case["id"] for case in cases
                    if set(case.get("tags", [])) & set(CHALLENGE_TAGS) and
                    case["q8_score"]["errors"] / case["q8_score"]["reference_units"] >=
                    SEVERE_CHALLENGE_ERROR_RATE]
    overall = ("fail" if severe_cases or any(item["status"] == "fail" for item in comparisons.values()) else
               "pass" if all(item["status"] == "pass" for item in comparisons.values()) else
               "inconclusive")
    return {"status": overall, "coverage": coverage, "comparisons": comparisons,
            "threshold_absolute_increase": 0.01,
            "severe_challenge_error_rate": SEVERE_CHALLENGE_ERROR_RATE,
            "severe_challenge_cases": severe_cases}
