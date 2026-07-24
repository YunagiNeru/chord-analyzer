from __future__ import annotations

import copy
import math
import re
from typing import Any


_TIME_SIGNATURE_RE = re.compile(r"\s*(\d+)\s*/\s*(\d+)\s*")


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _beats_per_bar(value: object) -> int:
    match = _TIME_SIGNATURE_RE.fullmatch(str(value or ""))
    if not match:
        return 4
    return max(1, min(12, int(match.group(1))))


def _dominant_tempo_segment(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    track = snapshot.get("track") or {}
    duration = max(0.001, _finite(track.get("durationSeconds"), 0.001))
    candidates = []
    for item in snapshot.get("tempoSegments") or []:
        if not isinstance(item, dict):
            continue
        start = max(0.0, min(duration, _finite(item.get("startSeconds"), -1.0)))
        end = max(0.0, min(duration, _finite(item.get("endSeconds"), -1.0)))
        bpm = _finite(item.get("bpm"), 0.0)
        confidence = max(0.0, min(1.0, _finite(item.get("confidence"), 0.5)))
        if end <= start or not 20.0 <= bpm <= 320.0:
            continue
        candidates.append(
            {
                "startSeconds": start,
                "endSeconds": end,
                "bpm": bpm,
                "confidence": confidence,
                "coverage": (end - start) / duration,
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["coverage"],
            item["confidence"],
            -item["bpm"],
        ),
    )


def choose_canonical_bpm(snapshot: dict[str, Any]) -> float:
    track = snapshot.get("track") or {}
    track_bpm = _finite(track.get("bpm"), 0.0)
    track_confidence = max(
        0.0,
        min(1.0, _finite(track.get("confidence"), 0.5)),
    )
    dominant = _dominant_tempo_segment(snapshot)

    if dominant is not None:
        segment_bpm = dominant["bpm"]
        segment_confidence = dominant["confidence"]
        if dominant["coverage"] >= 0.8 and segment_confidence >= 0.75:
            if track_bpm <= 0.0:
                return round(segment_bpm, 4)
            ratio = max(track_bpm, segment_bpm) / min(track_bpm, segment_bpm)
            octave_distance = abs(math.log2(ratio))
            is_half_or_double_conflict = abs(octave_distance - 1.0) <= 0.08
            segment_is_stronger = segment_confidence >= track_confidence + 0.08
            if is_half_or_double_conflict or segment_is_stronger:
                return round(segment_bpm, 4)

    if 20.0 <= track_bpm <= 320.0:
        return round(track_bpm, 4)
    if dominant is not None:
        return round(dominant["bpm"], 4)
    return 120.0


def _bar_and_beat(
    seconds: float,
    *,
    bpm: float,
    beats_per_bar: int,
    downbeat_offset: float,
) -> tuple[int, float]:
    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * beats_per_bar
    relative = seconds - downbeat_offset
    if relative < 0.0:
        return 1, 1.0
    bar = int(math.floor(relative / bar_duration)) + 1
    bar_start = downbeat_offset + (bar - 1) * bar_duration
    beat = ((seconds - bar_start) / beat_duration) + 1.0
    return bar, round(max(1.0, min(float(beats_per_bar), beat)), 2)


def _iter_flat_chords(section: dict[str, Any]) -> list[dict[str, Any]]:
    direct = section.get("chords")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]

    output: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for measure in section.get("measures") or []:
        if not isinstance(measure, dict):
            continue
        for chord in measure.get("chords") or []:
            if not isinstance(chord, dict):
                continue
            identity = (
                chord.get("symbol"),
                chord.get("startSeconds"),
                chord.get("endSeconds"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            output.append(copy.deepcopy(chord))
    output.sort(
        key=lambda item: (
            _finite(item.get("startSeconds"), 0.0),
            _finite(item.get("endSeconds"), 0.0),
            str(item.get("symbol") or ""),
        )
    )
    return output


def rebuild_review_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    rebuilt = copy.deepcopy(snapshot)
    track = rebuilt.setdefault("track", {})
    duration = max(0.001, _finite(track.get("durationSeconds"), 0.001))
    original_bpm = _finite(track.get("bpm"), 0.0)
    canonical_bpm = choose_canonical_bpm(rebuilt)
    signature = str(track.get("timeSignature") or "4/4")
    beats = _beats_per_bar(signature)
    downbeat = max(0.0, _finite(rebuilt.get("downbeatOffsetSeconds"), 0.0))

    max_bar_before = 0
    max_bar_after = 0
    merged_section_ids: list[str] = []

    for section in rebuilt.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "")
        section_name = str(section.get("name") or "")
        if "+" in section_id or "〜" in section_name:
            merged_section_ids.append(section_id or section_name)

        chords = _iter_flat_chords(section)
        for chord in chords:
            max_bar_before = max(max_bar_before, int(_finite(chord.get("bar"), 0.0)))
            start = max(0.0, min(duration, _finite(chord.get("startSeconds"), 0.0)))
            bar, beat = _bar_and_beat(
                start,
                bpm=canonical_bpm,
                beats_per_bar=beats,
                downbeat_offset=downbeat,
            )
            chord["bar"] = bar
            chord["beat"] = beat
            max_bar_after = max(max_bar_after, bar)
        section["chords"] = chords
        section.pop("measures", None)

    track["bpm"] = canonical_bpm
    rebuilt["tempoSegments"] = [
        {
            "startSeconds": 0.0,
            "endSeconds": round(duration, 3),
            "bpm": canonical_bpm,
            "confidence": max(
                0.5,
                _finite(
                    (_dominant_tempo_segment(snapshot) or {}).get("confidence"),
                    _finite(track.get("confidence"), 0.5),
                ),
            ),
        }
    ]
    rebuilt["rebuildDiagnostics"] = {
        "originalBpm": original_bpm,
        "canonicalBpm": canonical_bpm,
        "bpmChanged": abs(original_bpm - canonical_bpm) > 0.01,
        "maxBarBefore": max_bar_before,
        "maxBarAfter": max_bar_after,
        "semanticMergeDetected": bool(merged_section_ids),
        "mergedSectionIds": merged_section_ids,
        "sectionStructureRebuilt": False,
        "sectionStructureReason": (
            "Merged sections are reported but not split without original boundary evidence."
            if merged_section_ids
            else "No forced semantic merge was detected."
        ),
    }
    return rebuilt
