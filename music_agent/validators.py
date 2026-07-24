from __future__ import annotations

import math
import re

from .beat_grid import BeatGrid
from .chord_symbol import canonicalize_note, canonicalize_symbol, parse_chord
from .schemas import (
    AnalysisResult,
    ChordEvent,
    CompactSpecialistDraft,
    Measure,
    ResolutionDraft,
    SectionResult,
    SectionStructureDraft,
    SpecialistChordDraft,
    SpecialistSectionDraft,
)


NOTE_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
MAJOR_ROMAN = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")
MINOR_ROMAN = ("i", "bII", "ii", "III", "#III", "iv", "#iv", "v", "VI", "#VI", "VII", "#VII")
ALLOWED_SECTION_TYPES = {
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "post_chorus",
    "bridge",
    "interlude",
    "solo",
    "breakdown",
    "outro",
    "other",
}
SECTION_TYPE_ALIASES = {
    "aメロ": "verse",
    "bメロ": "pre_chorus",
    "サビ": "chorus",
    "ラスサビ": "chorus",
    "イントロ": "intro",
    "間奏": "interlude",
    "ブリッジ": "bridge",
    "ソロ": "solo",
    "アウトロ": "outro",
    "ending": "outro",
    "prechorus": "pre_chorus",
    "pre-chorus": "pre_chorus",
    "postchorus": "post_chorus",
    "post-chorus": "post_chorus",
}
_MAJOR_SECTION_TYPES = {"verse", "pre_chorus", "chorus", "post_chorus"}


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _confidence(value: object, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _finite(value, default)))


def normalise_section_type(value: str | None) -> str:
    token = (value or "other").strip().lower().replace(" ", "_")
    if token in ALLOWED_SECTION_TYPES:
        return token
    for alias, resolved in SECTION_TYPE_ALIASES.items():
        if alias in token:
            return resolved
    if "verse" in token:
        return "verse"
    if "chorus" in token or "hook" in token:
        return "chorus"
    if "intro" in token:
        return "intro"
    if "outro" in token:
        return "outro"
    if "bridge" in token:
        return "bridge"
    if "interlude" in token:
        return "interlude"
    return "other"


def normalise_sections(
    raw_sections: list[SectionStructureDraft],
    *,
    duration: float,
    max_sections: int = 20,
) -> list[SectionStructureDraft]:
    """Sanitise structure without merging semantically distinct sections."""

    del max_sections  # retained for API compatibility; quality gates own the cap.
    duration = max(0.001, _finite(duration, 0.001))
    candidates: list[SectionStructureDraft] = []
    for index, item in enumerate(raw_sections):
        start = _finite(item.startSeconds, -1.0)
        end = _finite(item.endSeconds, -1.0)
        if start < 0.0 or end <= start or start >= duration:
            continue
        start = max(0.0, min(duration, start))
        end = max(start + 0.25, min(duration, end))
        if end <= start:
            continue
        candidates.append(
            item.model_copy(
                update={
                    "id": (item.id or "").strip() or f"section-{index + 1}",
                    "name": (item.name or "").strip() or f"セクション{index + 1}",
                    "type": normalise_section_type(item.type),
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                    "confidence": _confidence(item.confidence),
                    "notes": (item.notes or "").strip(),
                }
            )
        )

    ordered = sorted(candidates, key=lambda item: (item.startSeconds, item.endSeconds, item.id))
    cleaned: list[SectionStructureDraft] = []
    for item in ordered:
        start = item.startSeconds
        end = item.endSeconds
        if cleaned and start < cleaned[-1].endSeconds:
            midpoint = max(
                cleaned[-1].startSeconds + 0.25,
                (cleaned[-1].endSeconds + start) / 2.0,
            )
            previous = cleaned[-1]
            previous_end = min(midpoint, end - 0.25)
            if previous_end > previous.startSeconds:
                cleaned[-1] = previous.model_copy(update={"endSeconds": round(previous_end, 3)})
            start = cleaned[-1].endSeconds
        if end - start >= 0.25:
            cleaned.append(item.model_copy(update={"startSeconds": round(start, 3), "endSeconds": round(end, 3)}))

    if not cleaned:
        return []
    if cleaned[0].startSeconds > 0.25:
        cleaned.insert(
            0,
            SectionStructureDraft(
                id="leading-region",
                name="冒頭",
                type="intro",
                startSeconds=0.0,
                endSeconds=cleaned[0].startSeconds,
                confidence=0.35,
                notes="先頭の未分類区間です。",
            ),
        )
    else:
        cleaned[0] = cleaned[0].model_copy(update={"startSeconds": 0.0})

    for index in range(len(cleaned) - 1):
        current = cleaned[index]
        following = cleaned[index + 1]
        if following.startSeconds - current.endSeconds > 0.25:
            boundary = (current.endSeconds + following.startSeconds) / 2.0
        else:
            boundary = max(current.startSeconds + 0.25, following.startSeconds)
        cleaned[index] = current.model_copy(update={"endSeconds": round(boundary, 3)})
        cleaned[index + 1] = following.model_copy(update={"startSeconds": round(boundary, 3)})
    cleaned[-1] = cleaned[-1].model_copy(update={"endSeconds": round(duration, 3)})

    seen: set[str] = set()
    output: list[SectionStructureDraft] = []
    for index, item in enumerate(cleaned, start=1):
        section_id = item.id if item.id not in seen else f"{item.id}-{index}"
        seen.add(section_id)
        output.append(item.model_copy(update={"id": section_id}))
    return output


def compact_to_specialist(
    result: CompactSpecialistDraft,
    *,
    section: SectionStructureDraft,
    role: str,
) -> SpecialistSectionDraft:
    return SpecialistSectionDraft(
        sectionId=section.id,
        role=role,
        key=section.key,
        mode=section.mode,
        repeatedPattern=result.repeatedPattern[:16],
        chords=[
            SpecialistChordDraft(
                symbol=item.symbol,
                startSeconds=item.startSeconds,
                endSeconds=item.endSeconds,
                confidence=item.confidence,
                alternatives=item.alternatives[:3],
            )
            for item in result.chords[:96]
        ],
    )


def normalise_specialist_result(
    result: SpecialistSectionDraft,
    *,
    section: SectionStructureDraft,
    role: str,
    clip_start: float,
    clip_end: float,
    beat_duration: float,
) -> SpecialistSectionDraft:
    clip_duration = max(0.1, clip_end - clip_start)
    raw_times = [
        _finite(value, -1.0)
        for chord in result.chords
        for value in (chord.startSeconds, chord.endSeconds)
    ]
    valid_times = [value for value in raw_times if value >= 0.0]
    relative = bool(valid_times) and max(valid_times) <= clip_duration + 1.0 and section.startSeconds > 1.0
    offset = clip_start if relative else 0.0

    chords: list[SpecialistChordDraft] = []
    for item in result.chords:
        start = _finite(item.startSeconds, -1.0) + offset
        end = _finite(item.endSeconds, -1.0) + offset
        if start < 0.0:
            continue
        start = max(section.startSeconds, min(section.endSeconds, start))
        if end <= start:
            end = start + max(0.12, beat_duration)
        end = min(section.endSeconds, end)
        if end - start < 0.05:
            continue
        symbol = canonicalize_symbol(item.symbol)
        alternatives = [
            canonicalize_symbol(value)
            for value in item.alternatives
            if canonicalize_symbol(value) != symbol
        ]
        chords.append(
            item.model_copy(
                update={
                    "symbol": symbol,
                    "startSeconds": round(start, 3),
                    "endSeconds": round(end, 3),
                    "confidence": _confidence(item.confidence),
                    "alternatives": list(dict.fromkeys(alternatives))[:3],
                    "evidence": "",
                }
            )
        )

    return result.model_copy(
        update={
            "sectionId": section.id,
            "role": role,
            "key": result.key or section.key,
            "mode": result.mode or section.mode,
            "chords": chords,
            "observations": [],
        }
    )


def apply_resolution(chords: list[ChordEvent], resolution: ResolutionDraft) -> None:
    for choice in resolution.choices:
        start = _finite(choice.startSeconds, -1.0)
        end = _finite(choice.endSeconds, -1.0)
        if start < 0.0 or end <= start:
            continue
        symbol = canonicalize_symbol(choice.chosenSymbol)
        if symbol == "X":
            continue
        confidence = _confidence(choice.confidence)
        for chord in chords:
            overlap = max(0.0, min(chord.endSeconds, end) - max(chord.startSeconds, start))
            if overlap <= 0:
                continue
            if symbol not in chord.alternatives and symbol != chord.symbol:
                continue
            old = chord.symbol
            chord.symbol = symbol
            chord.confidence = max(chord.confidence, confidence)
            chord.agreement = max(chord.agreement or 0.0, confidence)
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


def _unknown_event(start: float, end: float, source: str = "ai") -> ChordEvent:
    return ChordEvent(
        symbol="X",
        startSeconds=round(start, 3),
        endSeconds=round(end, 3),
        confidence=0.05,
        source=source,
        alternatives=[],
        agreement=0.0,
    )


def _clean_chords(
    chords: list[ChordEvent],
    *,
    section: SectionStructureDraft,
) -> list[ChordEvent]:
    output: list[ChordEvent] = []
    cursor = section.startSeconds
    source = chords[0].source if chords else "ai"
    for chord in sorted(chords, key=lambda item: (item.startSeconds, item.endSeconds, item.symbol)):
        raw_start = max(section.startSeconds, float(chord.startSeconds))
        if raw_start > cursor + 0.03:
            output.append(_unknown_event(cursor, raw_start, source))
            cursor = raw_start
        start = max(section.startSeconds, cursor, raw_start)
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
    if cursor < section.endSeconds - 0.03:
        output.append(_unknown_event(cursor, section.endSeconds, source))
    if not output:
        output.append(_unknown_event(section.startSeconds, section.endSeconds, source))
    return output


def _bar_duration(grid: BeatGrid) -> float:
    return grid.beat_duration * grid.beats_per_bar


def _bar_for_time(grid: BeatGrid, seconds: float) -> int:
    bar_duration = _bar_duration(grid)
    if grid.downbeat_offset > 1e-6:
        if seconds < grid.downbeat_offset - 1e-6:
            return 1
        return int(math.floor((seconds - grid.downbeat_offset) / bar_duration)) + 2
    return int(math.floor(max(0.0, seconds) / bar_duration)) + 1


def _bar_bounds(grid: BeatGrid, bar: int) -> tuple[float, float]:
    bar_duration = _bar_duration(grid)
    if grid.downbeat_offset > 1e-6:
        if bar == 1:
            return 0.0, grid.downbeat_offset
        start = grid.downbeat_offset + (bar - 2) * bar_duration
        return start, start + bar_duration
    start = (bar - 1) * bar_duration
    return start, start + bar_duration


def _beat_for_time(grid: BeatGrid, seconds: float) -> float:
    bar = _bar_for_time(grid, seconds)
    start, _ = _bar_bounds(grid, bar)
    value = ((seconds - start) / grid.beat_duration) + 1.0
    return max(1.0, min(float(grid.beats_per_bar), value))


def build_section_result(
    *,
    section: SectionStructureDraft,
    chords: list[ChordEvent],
    grid: BeatGrid,
    agreement: float,
    summary: str = "",
) -> SectionResult:
    clean = _clean_chords(chords, section=section)
    first_bar = _bar_for_time(grid, section.startSeconds)
    last_probe = max(section.startSeconds, section.endSeconds - 1e-6)
    last_bar = _bar_for_time(grid, last_probe)
    measures: list[Measure] = []

    for bar in range(first_bar, last_bar + 1):
        raw_start, raw_end = _bar_bounds(grid, bar)
        measure_start = max(section.startSeconds, raw_start)
        measure_end = min(section.endSeconds, raw_end)
        if measure_end - measure_start < 0.03:
            continue
        measure_chords: list[ChordEvent] = []
        for chord in clean:
            start = max(measure_start, chord.startSeconds)
            end = min(measure_end, chord.endSeconds)
            if end - start < 0.03:
                continue
            measure_chords.append(
                chord.model_copy(
                    update={
                        "startSeconds": round(start, 3),
                        "endSeconds": round(end, 3),
                        "bar": bar,
                        "beat": round(_beat_for_time(grid, start), 2),
                        "roman": roman_numeral(chord.symbol, section.key, section.mode),
                    }
                )
            )
        measures.append(
            Measure(
                bar=bar,
                startSeconds=round(max(0.0, measure_start), 3),
                endSeconds=round(measure_end, 3),
                chords=measure_chords,
            )
        )

    return SectionResult(
        id=section.id,
        name=section.name,
        type=normalise_section_type(section.type),
        startSeconds=section.startSeconds,
        endSeconds=section.endSeconds,
        key=section.key,
        mode=section.mode,
        confidence=_confidence(section.confidence),
        summary=summary or section.notes,
        measures=measures,
        agreement=_confidence(agreement, 0.0),
    )


def _coverage_gaps(
    intervals: list[tuple[float, float]],
    *,
    start: float,
    end: float,
    tolerance: float = 0.035,
) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = start
    for interval_start, interval_end in sorted(intervals):
        if interval_end <= cursor + tolerance:
            cursor = max(cursor, interval_end)
            continue
        if interval_start > cursor + tolerance:
            gaps.append((cursor, interval_start))
        cursor = max(cursor, interval_end)
    if cursor < end - tolerance:
        gaps.append((cursor, end))
    return gaps


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
        if section.startSeconds > previous_section_end + 0.035:
            errors.append(f"section_gap:{previous_section_end:.3f}-{section.startSeconds:.3f}")
        previous_section_end = max(previous_section_end, section.endSeconds)

        measure_intervals = [(item.startSeconds, item.endSeconds) for item in section.measures]
        for gap_start, gap_end in _coverage_gaps(
            measure_intervals,
            start=section.startSeconds,
            end=section.endSeconds,
        ):
            errors.append(f"measure_uncovered:{section.id}:{gap_start:.3f}-{gap_end:.3f}")

        chord_intervals: list[tuple[float, float]] = []
        previous_chord_end = section.startSeconds
        for measure in section.measures:
            if measure.bar < previous_bar:
                errors.append(f"bar_not_monotonic:{measure.bar}")
            previous_bar = max(previous_bar, measure.bar)
            if measure.startSeconds < section.startSeconds - 1e-3 or measure.endSeconds > section.endSeconds + 1e-3:
                errors.append(f"measure_outside_section:{section.id}:{measure.bar}")
            for chord in measure.chords:
                chord_intervals.append((chord.startSeconds, chord.endSeconds))
                if chord.startSeconds < section.startSeconds - 1e-3 or chord.endSeconds > section.endSeconds + 1e-3:
                    errors.append(f"chord_outside_section:{section.id}:{chord.symbol}")
                if chord.startSeconds < measure.startSeconds - 1e-3 or chord.endSeconds > measure.endSeconds + 1e-3:
                    errors.append(f"chord_outside_measure:{section.id}:{measure.bar}:{chord.symbol}")
                if chord.startSeconds < previous_chord_end - 1e-3:
                    errors.append(f"chord_overlap:{section.id}:{chord.symbol}")
                previous_chord_end = max(previous_chord_end, chord.endSeconds)
                if chord.beat is not None and not (1.0 <= chord.beat <= 12.0):
                    errors.append(f"invalid_beat:{section.id}:{chord.beat}")
                if not math.isfinite(chord.startSeconds) or not math.isfinite(chord.endSeconds):
                    errors.append(f"non_finite_time:{section.id}")
        for gap_start, gap_end in _coverage_gaps(
            chord_intervals,
            start=section.startSeconds,
            end=section.endSeconds,
        ):
            errors.append(f"chord_uncovered:{section.id}:{gap_start:.3f}-{gap_end:.3f}")
    if previous_section_end < duration - 0.035:
        errors.append(f"section_gap:{previous_section_end:.3f}-{duration:.3f}")
    return sorted(set(errors))


def _boundary_distance_beats(result: AnalysisResult, seconds: float) -> float:
    bpm = result.track.bpm or 120.0
    beat_duration = 60.0 / bpm
    offset = result.downbeatOffsetSeconds or 0.0
    phase = ((seconds - offset) / beat_duration) % 1.0
    return min(phase, 1.0 - phase)


def validate_quality(result: AnalysisResult) -> list[str]:
    errors: list[str] = []
    duration = result.track.durationSeconds
    chords = [
        chord
        for section in result.sections
        for measure in section.measures
        for chord in measure.chords
    ]
    known = [chord for chord in chords if canonicalize_symbol(chord.symbol) not in {"X", "N"}]
    known_coverage = sum(chord.endSeconds - chord.startSeconds for chord in known)
    unresolved_coverage = sum(
        item.endSeconds - item.startSeconds
        for item in result.uncertainRanges
        if not item.resolved
    )

    if duration >= 90.0 and len(result.sections) < 3:
        errors.append(f"too_few_sections:{len(result.sections)}")
    if len(result.sections) > 20:
        errors.append(f"too_many_sections:{len(result.sections)}")
    minimum_chords = max(4, math.ceil(duration / 20.0))
    if len(chords) < minimum_chords:
        errors.append(f"too_few_chords:{len(chords)}<{minimum_chords}")
    if not known:
        errors.append("no_known_chords")
    if duration > 0 and known_coverage / duration < 0.45:
        errors.append(f"insufficient_known_chord_coverage:{known_coverage / duration:.3f}")
    if duration > 0 and unresolved_coverage / duration > 0.30:
        errors.append(f"excessive_unresolved_coverage:{unresolved_coverage / duration:.3f}")
    if any(
        section.id == "full-track"
        or "フォールバック" in section.summary
        or "fallback" in section.summary.lower()
        for section in result.sections
    ):
        errors.append("structure_fallback_present")
    if any("セクション数上限" in section.summary for section in result.sections):
        errors.append("semantic_sections_merged_by_limit")

    bpm = result.track.bpm or 120.0
    beats = 4
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", result.track.timeSignature or "4/4")
    if match:
        beats = max(1, int(match.group(1)))
    bar_duration = (60.0 / bpm) * beats
    for section in result.sections:
        max_bars = 16 if section.type in _MAJOR_SECTION_TYPES else 20
        section_bars = (section.endSeconds - section.startSeconds) / bar_duration
        if section_bars > max_bars + 0.35:
            errors.append(f"oversized_section:{section.id}:{section_bars:.2f}>{max_bars}")
        if section.startSeconds > 0.035 and _boundary_distance_beats(result, section.startSeconds) > 0.18:
            errors.append(f"section_boundary_off_beat:{section.id}:{section.startSeconds:.3f}")

    if result.track.bpm and result.tempoSegments:
        dominant = max(result.tempoSegments, key=lambda item: item.endSeconds - item.startSeconds)
        coverage = (dominant.endSeconds - dominant.startSeconds) / duration if duration > 0 else 0.0
        if coverage >= 0.8:
            ratio = max(result.track.bpm, dominant.bpm) / min(result.track.bpm, dominant.bpm)
            if ratio > 1.03:
                errors.append(f"tempo_summary_mismatch:{result.track.bpm:g}!={dominant.bpm:g}")
    return sorted(set(errors))
