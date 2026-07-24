from __future__ import annotations

from dataclasses import dataclass

from .chord_symbol import PC_TO_NOTE, parse_chord
from .schemas import ChordEvent


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    key: str
    mode: str
    confidence: float
    margin: float


_MAJOR_ROOT_WEIGHTS = {
    0: 1.00,
    2: 0.56,
    4: 0.42,
    5: 0.78,
    7: 0.92,
    9: 0.68,
    11: 0.28,
}
_MINOR_ROOT_WEIGHTS = {
    0: 1.00,
    2: 0.22,
    3: 0.68,
    5: 0.78,
    7: 0.70,
    8: 0.86,
    10: 0.72,
    11: 0.24,
}


def _quality_bonus(interval: int, quality: str, mode: str) -> float:
    if mode == "major":
        expected = {
            0: "major",
            2: "minor",
            4: "minor",
            5: "major",
            7: "major",
            9: "minor",
            11: "diminished",
        }.get(interval)
    else:
        expected = {
            0: "minor",
            2: "diminished",
            3: "major",
            5: "minor",
            7: "minor",
            8: "major",
            10: "major",
            11: "diminished",
        }.get(interval)
        # A major V is common in minor keys and should support rather than
        # contradict the tonic estimate.
        if interval == 7 and quality == "major":
            return 0.24
    if expected is None:
        return 0.0
    return 0.22 if expected == quality else -0.10


def infer_key_from_chords(
    chords: list[ChordEvent],
    *,
    minimum_coverage_seconds: float = 12.0,
) -> KeyEstimate | None:
    evidence: list[tuple[int, str, float]] = []
    total_weight = 0.0
    for event in chords:
        parsed = parse_chord(event.symbol)
        if parsed.root_pc is None or parsed.unknown or parsed.no_chord:
            continue
        duration = max(0.0, event.endSeconds - event.startSeconds)
        confidence = max(0.05, min(1.0, event.confidence))
        weight = duration * confidence
        if weight <= 0.0:
            continue
        evidence.append((parsed.root_pc, parsed.quality, weight))
        total_weight += weight

    if total_weight < minimum_coverage_seconds:
        return None

    ranked: list[tuple[float, int, str]] = []
    for tonic in range(12):
        for mode, root_weights in (
            ("major", _MAJOR_ROOT_WEIGHTS),
            ("minor", _MINOR_ROOT_WEIGHTS),
        ):
            score = 0.0
            for root_pc, quality, weight in evidence:
                interval = (root_pc - tonic) % 12
                root_score = root_weights.get(interval, -0.34)
                score += weight * (
                    root_score + _quality_bonus(interval, quality, mode)
                )
            ranked.append((score / total_weight, tonic, mode))

    ranked.sort(reverse=True)
    best_score, tonic, mode = ranked[0]
    second_score = ranked[1][0]
    margin = max(0.0, best_score - second_score)
    confidence = max(
        0.0,
        min(
            0.99,
            0.42 + margin * 1.8 + max(0.0, best_score) * 0.22,
        ),
    )
    if margin < 0.055 or confidence < 0.56:
        return None
    return KeyEstimate(
        key=PC_TO_NOTE[tonic],
        mode=mode,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
    )
