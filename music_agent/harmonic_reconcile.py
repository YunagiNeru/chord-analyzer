from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .chord_symbol import parse_chord
from .schemas import AnalysisResult, ChordEvent, TempoSegment
from .validators import roman_numeral


MAJOR_SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})
MINOR_SCALE = frozenset({0, 2, 3, 5, 7, 8, 10, 11})
MAJOR_QUALITIES = {
    0: "major",
    2: "minor",
    4: "minor",
    5: "major",
    7: "major",
    9: "minor",
    11: "diminished",
}
MINOR_QUALITIES = {
    0: "minor",
    2: "diminished",
    3: "major",
    5: "minor",
    7: "minor",
    8: "major",
    10: "major",
    11: "diminished",
}

# Prefer spellings that do not introduce avoidable theoretical noise. In
# particular, pitch class 8 minor is G# minor rather than Ab minor.
MAJOR_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B")


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    key: str
    mode: str
    score: float
    margin: float


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _canonical_hint(value: str | None) -> tuple[int, str] | None:
    if not value:
        return None
    token = value.strip().replace("♯", "#").replace("♭", "b")
    lower = token.lower()
    mode = "minor" if "minor" in lower or "マイナー" in token else "major"
    note = token.split()[0].replace("major", "").replace("minor", "").strip()
    lookup = {
        "C": 0,
        "B#": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "Fb": 4,
        "E#": 5,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
        "Cb": 11,
    }
    pitch_class = lookup.get(note)
    return (pitch_class, mode) if pitch_class is not None else None


def _candidate_name(tonic: int, mode: str) -> str:
    return (MINOR_NAMES if mode == "minor" else MAJOR_NAMES)[tonic % 12]


def _event_weight(event: ChordEvent) -> float:
    duration = max(0.0, _finite(event.endSeconds) - _finite(event.startSeconds))
    confidence = max(0.15, min(1.0, _finite(event.confidence, 0.5)))
    agreement = event.agreement
    agreement_weight = max(0.35, min(1.0, _finite(agreement, 0.6))) if agreement is not None else 0.6
    return duration * confidence * agreement_weight


def _score_candidate(
    events: Sequence[ChordEvent],
    *,
    tonic: int,
    mode: str,
    hints: set[tuple[int, str]] | None = None,
) -> float:
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    expected = MINOR_QUALITIES if mode == "minor" else MAJOR_QUALITIES
    total_weight = 0.0
    total_score = 0.0

    for event in events:
        chord = parse_chord(event.symbol)
        if chord.root_pc is None or chord.unknown or chord.no_chord:
            continue
        weight = _event_weight(event)
        if weight <= 0.0:
            continue
        degree = (chord.root_pc - tonic) % 12
        score = 1.0 if degree in scale else -0.85
        expected_quality = expected.get(degree)
        if expected_quality is not None:
            if chord.quality == expected_quality:
                score += 0.62
            elif chord.quality in {"major", "minor", "diminished"}:
                score -= 0.32
        if degree == 0:
            score += 0.28
        elif degree == 7:
            score += 0.16
        elif mode == "minor" and degree in {8, 10}:
            score += 0.08
        total_score += score * weight
        total_weight += weight

    if total_weight <= 0.0:
        return -99.0
    normalized = total_score / total_weight
    if hints and (tonic, mode) in hints:
        normalized += 0.035
    return normalized


def estimate_key(
    events: Sequence[ChordEvent],
    *,
    hints: Iterable[str | None] = (),
) -> KeyEstimate:
    parsed_hints = {
        parsed
        for value in hints
        if (parsed := _canonical_hint(value)) is not None
    }
    scores: list[tuple[float, int, str]] = []
    for mode in ("major", "minor"):
        for tonic in range(12):
            scores.append(
                (
                    _score_candidate(
                        events,
                        tonic=tonic,
                        mode=mode,
                        hints=parsed_hints,
                    ),
                    tonic,
                    mode,
                )
            )
    scores.sort(key=lambda item: (-item[0], item[2], item[1]))
    best = scores[0]
    second = scores[1]
    return KeyEstimate(
        key=_candidate_name(best[1], best[2]),
        mode=best[2],
        score=round(best[0], 6),
        margin=round(best[0] - second[0], 6),
    )


def _merge_display_events(section) -> list[ChordEvent]:
    events = sorted(
        [
            chord
            for measure in section.measures
            for chord in measure.chords
        ],
        key=lambda item: (item.startSeconds, item.endSeconds, item.symbol),
    )
    merged: list[ChordEvent] = []
    for event in events:
        if (
            merged
            and merged[-1].symbol == event.symbol
            and event.startSeconds <= merged[-1].endSeconds + 0.003
        ):
            previous = merged[-1]
            previous.endSeconds = max(previous.endSeconds, event.endSeconds)
            previous.confidence = max(previous.confidence, event.confidence)
            if event.agreement is not None:
                previous.agreement = max(previous.agreement or 0.0, event.agreement)
        else:
            merged.append(event.model_copy(deep=True))
    return merged


def reconcile_result_harmony(
    result: AnalysisResult,
    *,
    reference_hints: Iterable[str] = (),
) -> None:
    section_events = {
        section.id: _merge_display_events(section)
        for section in result.sections
    }
    all_events = [
        event
        for section in result.sections
        for event in section_events[section.id]
    ]
    global_estimate = estimate_key(
        all_events,
        hints=[
            result.track.globalKey,
            *reference_hints,
        ],
    )
    result.track.globalKey = global_estimate.key
    result.track.globalMode = global_estimate.mode

    for section in result.sections:
        events = section_events[section.id]
        local = estimate_key(
            events,
            hints=[section.key, result.track.globalKey, *reference_hints],
        )
        global_score = _score_candidate(
            events,
            tonic=_canonical_hint(f"{global_estimate.key} {global_estimate.mode}")[0],
            mode=global_estimate.mode,
        )
        use_local = (
            len(events) >= 2
            and local.score >= global_score + 0.10
            and local.margin >= 0.035
        )
        chosen = local if use_local else global_estimate
        section.key = chosen.key
        section.mode = chosen.mode
        for measure in section.measures:
            for chord in measure.chords:
                chord.roman = roman_numeral(chord.symbol, chosen.key, chosen.mode)


def normalize_tempo_segments(result: AnalysisResult) -> None:
    duration = result.track.durationSeconds
    bpm = result.track.bpm
    if not bpm or duration <= 0.0:
        return
    segments = sorted(result.tempoSegments, key=lambda item: (item.startSeconds, item.endSeconds))
    if not segments:
        result.tempoSegments = [
            TempoSegment(
                startSeconds=0.0,
                endSeconds=duration,
                bpm=bpm,
                confidence=0.5,
            )
        ]
        return

    close_to_grid = all(
        item.bpm > 0.0 and abs(item.bpm - bpm) / bpm <= 0.03
        for item in segments
    )
    if close_to_grid:
        result.tempoSegments = [
            TempoSegment(
                startSeconds=0.0,
                endSeconds=duration,
                bpm=bpm,
                confidence=max(item.confidence for item in segments),
            )
        ]
        return

    normalized: list[TempoSegment] = []
    cursor = 0.0
    for item in segments:
        start = max(cursor, min(duration, item.startSeconds))
        end = max(start, min(duration, item.endSeconds))
        if start > cursor + 0.001:
            fill_bpm = normalized[-1].bpm if normalized else bpm
            normalized.append(
                TempoSegment(
                    startSeconds=cursor,
                    endSeconds=start,
                    bpm=fill_bpm,
                    confidence=0.25,
                )
            )
        if end > start + 0.001:
            normalized.append(
                item.model_copy(
                    update={
                        "startSeconds": start,
                        "endSeconds": end,
                    }
                )
            )
            cursor = end
    if cursor < duration - 0.001:
        normalized.append(
            TempoSegment(
                startSeconds=cursor,
                endSeconds=duration,
                bpm=normalized[-1].bpm if normalized else bpm,
                confidence=0.25,
            )
        )
    result.tempoSegments = normalized


def validate_tempo_coverage(result: AnalysisResult) -> list[str]:
    errors: list[str] = []
    duration = result.track.durationSeconds
    cursor = 0.0
    for index, item in enumerate(result.tempoSegments):
        if item.startSeconds > cursor + 0.002:
            errors.append(f"tempo_segment_gap:{cursor:.3f}-{item.startSeconds:.3f}")
        if item.startSeconds < cursor - 0.002:
            errors.append(f"tempo_segment_overlap:{index}")
        if item.endSeconds <= item.startSeconds:
            errors.append(f"tempo_segment_invalid:{index}")
        cursor = max(cursor, item.endSeconds)
    if cursor < duration - 0.002:
        errors.append(f"tempo_segment_gap:{cursor:.3f}-{duration:.3f}")
    return errors
