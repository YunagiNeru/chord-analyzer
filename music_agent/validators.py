from __future__ import annotations

import math
import re
from collections import defaultdict

from .beat_grid import BeatGrid
from .chord_symbol import canonicalize_note, canonicalize_symbol, parse_chord
from .schemas import (
    AnalysisResult,
    ChordEvent,
    Measure,
    ResolutionDraft,
    SectionResult,
    SectionStructureDraft,
)


NOTE_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
MAJOR_ROMAN = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")
MINOR_ROMAN = ("i", "bII", "ii", "III", "#III", "iv", "#iv", "v", "VI", "#VI", "VII", "#VII")


def normalise_sections(
    raw_sections: list[SectionStructureDraft],
    *,
    duration: float,
    max_sections: int = 8,
) -> list[SectionStructureDraft]:
    ordered = sorted(raw_sections, key=lambda item: (item.startSeconds, item.endSeconds, item.id))
    cleaned: list[SectionStructureDraft] = []
    for index, item in enumerate(ordered):
        start = max(0.0, min(duration, float(item.startSeconds)))
        end = max(start + 0.25, min(duration, float(item.endSeconds)))
        if start >= duration or end <= start:
            continue
        if cleaned and start < cleaned[-1].endSeconds:
            midpoint = max(cleaned[-1].startSeconds + 0.25, (cleaned[-1].endSeconds + start) / 2.0)
            previous = cleaned[-1]
            cleaned[-1] = previous.model_copy(update={"endSeconds": min(midpoint, end - 0.25)})
            start = cleaned[-1].endSeconds
        section_id = item.id.strip() or f"section-{index + 1}"
        cleaned.append(
            item.model_copy(
                update={
                    "id": section_id,
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                }
            )
        )

    if not cleaned:
        return [
            SectionStructureDraft(
                id="full-track",
                name="全体",
                type="other",
                startSeconds=0.0,
                endSeconds=max(0.001, duration),
                confidence=0.2,
                notes="構造境界を確定できなかったため全体区間として扱います。",
            )
        ]

    first = cleaned[0]
    if first.startSeconds > 0.25:
        cleaned.insert(
            0,
            SectionStructureDraft(
                id="leading-region",
                name="冒頭",
                type="intro",
                startSeconds=0.0,
                endSeconds=first.startSeconds,
                confidence=0.35,
                notes="先頭の未分類区間です。",
            ),
        )
    else:
        cleaned[0] = first.model_copy(update={"startSeconds": 0.0})

    for index in range(len(cleaned) - 1):
        current = cleaned[index]
        following = cleaned[index + 1]
        if following.startSeconds - current.endSeconds > 0.25:
            midpoint = (current.endSeconds + following.startSeconds) / 2.0
            cleaned[index] = current.model_copy(update={"endSeconds": round(midpoint, 3)})
            cleaned[index + 1] = following.model_copy(update={"startSeconds": round(midpoint, 3)})
        else:
            boundary = max(current.startSeconds + 0.25, following.startSeconds)
            cleaned[index] = current.model_copy(update={"endSeconds": round(boundary, 3)})
            cleaned[index + 1] = following.model_copy(update={"startSeconds": round(boundary, 3)})

    cleaned[-1] = cleaned[-1].model_copy(update={"endSeconds": round(duration, 3)})

    while len(cleaned) > max_sections:
        merge_index = min(
            range(len(cleaned) - 1),
            key=lambda index: (
                cleaned[index].endSeconds - cleaned[index].startSeconds
                + cleaned[index + 1].endSeconds - cleaned[index + 1].startSeconds,
                index,
            ),
        )
        left = cleaned[merge_index]
        right = cleaned[merge_index + 1]
        preferred = left if left.confidence >= right.confidence else right
        merged = preferred.model_copy(
            update={
                "id": f"{left.id}+{right.id}",
                "name": f"{left.name}〜{right.name}",
                "startSeconds": left.startSeconds,
                "endSeconds": right.endSeconds,
                "confidence": min(left.confidence, right.confidence),
                "notes": "セクション数上限のため隣接区間を統合しました。",
            }
        )
        cleaned[merge_index : merge_index + 2] = [merged]

    seen: set[str] = set()
    output: list[SectionStructureDraft] = []
    for index, item in enumerate(cleaned, start=1):
        section_id = item.id
        if section_id in seen:
            section_id = f"{section_id}-{index}"
        seen.add(section_id)
        output.append(item.model_copy(update={"id": section_id}))
    return output


def apply_resolution(
    chords: list[ChordEvent],
    resolution: ResolutionDraft,
) -> None:
    for choice in resolution.choices:
        symbol = canonicalize_symbol(choice.chosenSymbol)
        if symbol == "X":
            continue
        for chord in chords:
            overlap = max(
                0.0,
                min(chord.endSeconds, choice.endSeconds)
                - max(chord.startSeconds, choice.startSeconds),
            )
            if overlap <= 0:
                continue
            if symbol not in chord.alternatives and symbol != chord.symbol:
                continue
            old = chord.symbol
            chord.symbol = symbol
            chord.confidence = max(chord.confidence, choice.confidence)
            chord.agreement = max(chord.agreement or 0.0, choice.confidence)
            chord.alternatives = list(
                dict.fromkeys([old] + [value for value in chord.alternatives if value != symbol])
            )[:3]


def _key_root(key: str | None) -> int | None:
    if not key:
        return None
    match = re.match(r"\s*([A-Ga-g](?:#|b|♯|♭)?)", key)
    if not match:
        return None
    note = canonicalize_note(match.group(1))
    return NOTE_PC.get(note) if note else None


def roman_numeral(symbol: str, key: str | None, mode: str | None) -> str | None:
    chord = parse_chord(symbol)
    root = chord.root_pc
    tonic = _key_root(key)
    if root is None or tonic is None:
        return None
    interval = (root - tonic) % 12
    minor_mode = (mode or "").lower().startswith("min") or (key or "").lower().endswith("minor")
    roman = MINOR_ROMAN[interval] if minor_mode else MAJOR_ROMAN[interval]
    if chord.quality == "minor" and roman.isupper():
        roman = roman.lower()
    elif chord.quality == "major" and roman.islower():
        roman = roman.upper()
    if chord.quality == "diminished":
        roman += "°"
    return roman


def _clean_chords(
    chords: list[ChordEvent],
    *,
    section: SectionStructureDraft,
) -> list[ChordEvent]:
    output: list[ChordEvent] = []
    cursor = section.startSeconds
    for chord in sorted(chords, key=lambda item: (item.startSeconds, item.endSeconds, item.symbol)):
        start = max(section.startSeconds, cursor, float(chord.startSeconds))
        end = min(section.endSeconds, float(chord.endSeconds))
        if end - start < 0.03:
            continue
        symbol = canonicalize_symbol(chord.symbol)
        clean = chord.model_copy(
            update={
                "symbol": symbol,
                "startSeconds": round(start, 3),
                "endSeconds": round(end, 3),
            }
        )
        if output and output[-1].symbol == clean.symbol and clean.startSeconds <= output[-1].endSeconds + 0.05:
            previous = output[-1]
            previous.endSeconds = clean.endSeconds
            previous.confidence = round((previous.confidence + clean.confidence) / 2.0, 4)
            previous.alternatives = list(dict.fromkeys(previous.alternatives + clean.alternatives))[:3]
            values = [value for value in (previous.agreement, clean.agreement) if value is not None]
            previous.agreement = round(sum(values) / len(values), 4) if values else None
        else:
            output.append(clean)
        cursor = output[-1].endSeconds
    return output


def build_section_result(
    *,
    section: SectionStructureDraft,
    chords: list[ChordEvent],
    grid: BeatGrid,
    agreement: float,
    summary: str = "",
) -> SectionResult:
    clean = _clean_chords(chords, section=section)
    grouped: dict[int, list[ChordEvent]] = defaultdict(list)
    section_key = section.key
    for chord in clean:
        bar = grid.bar_for_time(chord.startSeconds)
        beat = round(grid.beat_for_time(chord.startSeconds), 2)
        updated = chord.model_copy(
            update={
                "bar": bar,
                "beat": beat,
                "roman": roman_numeral(chord.symbol, section_key, section.mode),
            }
        )
        grouped[bar].append(updated)

    measures: list[Measure] = []
    bar_duration = grid.beat_duration * grid.beats_per_bar
    for bar in sorted(grouped):
        bar_start = grid.downbeat_offset + (bar - 1) * bar_duration
        start = max(section.startSeconds, bar_start)
        end = min(section.endSeconds, bar_start + bar_duration)
        if end <= start:
            end = min(section.endSeconds, start + bar_duration)
        measures.append(
            Measure(
                bar=bar,
                startSeconds=round(max(0.0, start), 3),
                endSeconds=round(max(start + 0.03, end), 3),
                chords=grouped[bar],
            )
        )

    return SectionResult(
        id=section.id,
        name=section.name,
        type=section.type,
        startSeconds=section.startSeconds,
        endSeconds=section.endSeconds,
        key=section.key,
        mode=section.mode,
        confidence=section.confidence,
        summary=summary or section.notes,
        measures=measures,
        agreement=agreement,
    )


def validate_invariants(result: AnalysisResult) -> list[str]:
    errors: list[str] = []
    duration = result.track.durationSeconds
    previous_section_end = 0.0
    previous_bar = 0
    for section in result.sections:
        if section.startSeconds < -1e-6 or section.endSeconds > duration + 1e-3:
            errors.append(f"section_out_of_range:{section.id}")
        if section.startSeconds < previous_section_end - 1e-3:
            errors.append(f"section_overlap:{section.id}")
        previous_section_end = max(previous_section_end, section.endSeconds)
        previous_chord_end = section.startSeconds
        for measure in section.measures:
            if measure.bar < previous_bar:
                errors.append(f"bar_not_monotonic:{measure.bar}")
            previous_bar = max(previous_bar, measure.bar)
            for chord in measure.chords:
                if chord.startSeconds < section.startSeconds - 1e-3 or chord.endSeconds > section.endSeconds + 1e-3:
                    errors.append(f"chord_outside_section:{section.id}:{chord.symbol}")
                if chord.startSeconds < previous_chord_end - 1e-3:
                    errors.append(f"chord_overlap:{section.id}:{chord.symbol}")
                previous_chord_end = max(previous_chord_end, chord.endSeconds)
                if chord.beat is not None and not (1.0 <= chord.beat <= 12.0):
                    errors.append(f"invalid_beat:{section.id}:{chord.beat}")
                if not math.isfinite(chord.startSeconds) or not math.isfinite(chord.endSeconds):
                    errors.append(f"non_finite_time:{section.id}")
    return sorted(set(errors))
