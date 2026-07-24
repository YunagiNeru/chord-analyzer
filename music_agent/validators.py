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
    return "other"


def _same_semantic_section(
    left: SectionStructureDraft,
    right: SectionStructureDraft,
) -> bool:
    if normalise_section_type(left.type) == normalise_section_type(right.type):
        return True
    left_name = re.sub(r"\d+", "", (left.name or "").lower())
    right_name = re.sub(r"\d+", "", (right.name or "").lower())
    return bool(left_name and left_name == right_name)


def normalise_sections(
    raw_sections: list[SectionStructureDraft],
    *,
    duration: float,
    max_sections: int = 12,
) -> list[SectionStructureDraft]:
    duration = max(0.001, _finite(duration, 0.001))
    # A fixed cap of 12 merged Bメロ with サビ and Cメロ with 落ちサビ on
    # ordinary three-minute songs. Scale the safe cap with duration instead.
    effective_max_sections = max(
        max_sections,
        min(20, max(1, math.ceil(duration / 10.0))),
    )

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
        section_id = (item.id or "").strip() or f"section-{index + 1}"
        name = (item.name or "").strip() or f"セクション{index + 1}"
        candidates.append(
            item.model_copy(
                update={
                    "id": section_id,
                    "name": name,
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
        if end - start < 0.25:
            continue
        cleaned.append(item.model_copy(update={"startSeconds": round(start, 3), "endSeconds": round(end, 3)}))

    if not cleaned:
        return []

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

    while len(cleaned) > effective_max_sections:
        compatible_pairs = [
            index
            for index in range(len(cleaned) - 1)
            if _same_semantic_section(cleaned[index], cleaned[index + 1])
        ]
        if not compatible_pairs:
            # Do not destroy semantically different sections merely to hit a
            # display cap. The quality gate will reject implausibly many sections.
            break
        merge_index = min(
            compatible_pairs,
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
                "notes": "同種の隣接セクションを統合しました。",
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
                    "alternatives": list(dict.fromkeys(alternatives))[:5],
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
    bar_duration = grid.beat_duration * grid.beats_per_bar
    first_bar = grid.bar_for_time(section.startSeconds)
    last_probe = max(section.startSeconds, section.endSeconds - 1e-6)
    last_bar = grid.bar_for_time(last_probe)
    measures: list[Measure] = []

    for bar in range(first_bar, last_bar + 1):
        raw_bar_start = grid.downbeat_offset + (bar - 1) * bar_duration
        measure_start = max(section.startSeconds, raw_bar_start)
        measure_end = min(section.endSeconds, raw_bar_start + bar_duration)
        if measure_end <= measure_start:
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
                        "beat": round(grid.beat_for_time(start), 2),
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
            if measure.startSeconds < section.startSeconds - 1e-3 or measure.endSeconds > section.endSeconds + 1e-3:
                errors.append(f"measure_outside_section:{section.id}:{measure.bar}")
            for chord in measure.chords:
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
    return sorted(set(errors))


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

    if result.track.bpm and result.tempoSegments:
        dominant = max(
            result.tempoSegments,
            key=lambda item: item.endSeconds - item.startSeconds,
        )
        coverage = (dominant.endSeconds - dominant.startSeconds) / duration if duration > 0 else 0.0
        if coverage >= 0.8:
            ratio = max(result.track.bpm, dominant.bpm) / min(result.track.bpm, dominant.bpm)
            if ratio > 1.03:
                errors.append(
                    f"tempo_summary_mismatch:{result.track.bpm:g}!={dominant.bpm:g}"
                )
    return sorted(set(errors))
