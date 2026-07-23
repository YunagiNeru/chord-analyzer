from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median

from .accuracy_profile import AccuracyProfile, load_accuracy_profile
from .schemas import DspSummary, RhythmDraft, SectionStructureDraft, TempoSegment


@dataclass(frozen=True, slots=True)
class BeatGrid:
    bpm: float
    time_signature: str
    beats_per_bar: int
    beat_duration: float
    downbeat_offset: float
    duration: float
    beat_times: tuple[float, ...]
    bar_starts: tuple[float, ...]
    score: float
    tempo_segments: tuple[TempoSegment, ...] = ()

    def slot_boundaries(self, start: float, end: float) -> list[float]:
        values = [start]
        values.extend(value for value in self.beat_times if start + 1e-6 < value < end - 1e-6)
        values.append(end)
        return sorted(set(round(value, 6) for value in values))

    def bar_for_time(self, seconds: float) -> int:
        relative = seconds - self.downbeat_offset
        if relative < 0:
            return 1
        return int(math.floor(relative / (self.beat_duration * self.beats_per_bar))) + 1

    def beat_for_time(self, seconds: float) -> float:
        bar = self.bar_for_time(seconds)
        bar_start = self.downbeat_offset + (bar - 1) * self.beat_duration * self.beats_per_bar
        value = ((seconds - bar_start) / self.beat_duration) + 1.0
        return max(1.0, min(float(self.beats_per_bar), value))


_TIME_SIGNATURE_RE = re.compile(r"\s*(\d+)\s*/\s*(\d+)\s*")


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def beats_per_bar(value: str | None) -> int:
    if not value:
        return 4
    match = _TIME_SIGNATURE_RE.fullmatch(str(value))
    if not match:
        return 4
    return max(1, min(12, int(match.group(1))))


def _normalise_bpm(value: float) -> list[float]:
    value = _finite(value, 0.0)
    if value <= 0.0:
        return []
    return [
        round(value * multiplier, 4)
        for multiplier in (0.5, 1.0, 2.0)
        if 40.0 <= value * multiplier <= 240.0
    ]


def _candidate_bpms(rhythm: RhythmDraft, dsp: DspSummary | None) -> list[float]:
    values: list[float] = []
    if rhythm.selectedBpm:
        values.extend(_normalise_bpm(rhythm.selectedBpm))
    for candidate in rhythm.bpmCandidates:
        values.extend(_normalise_bpm(candidate.bpm))
    if dsp and dsp.bpm:
        values.extend(_normalise_bpm(dsp.bpm))
    return sorted(set(values or [120.0]))


def _integer_bar_error(
    bpm: float,
    beats: int,
    sections: list[SectionStructureDraft],
) -> float:
    bar_duration = (60.0 / bpm) * beats
    errors = []
    for section in sections:
        length = section.endSeconds - section.startSeconds
        if length >= bar_duration * 0.7:
            bars = length / bar_duration
            errors.append(abs(bars - round(bars)))
    return median(errors) if errors else 0.5


def _beat_alignment_error(bpm: float, offset: float, observed: list[float]) -> float:
    if not observed:
        return 0.5
    beat_duration = 60.0 / bpm
    errors = []
    for value in observed[:2000]:
        phase = ((value - offset) / beat_duration) % 1.0
        errors.append(min(phase, 1.0 - phase))
    return median(errors) if errors else 0.5


def _section_alignment_error(
    bpm: float,
    beats: int,
    offset: float,
    sections: list[SectionStructureDraft],
) -> float:
    bar_duration = (60.0 / bpm) * beats
    errors = []
    for section in sections:
        phase = ((section.startSeconds - offset) / bar_duration) % 1.0
        errors.append(min(phase, 1.0 - phase))
    return median(errors) if errors else 0.5


def _offset_candidates(
    bpm: float,
    beats: int,
    rhythm: RhythmDraft,
    dsp: DspSummary | None,
) -> list[float]:
    bar_duration = (60.0 / bpm) * beats
    model_offset = _finite(rhythm.downbeatOffsetSeconds, 0.0)
    values = [max(0.0, min(bar_duration, model_offset))]
    if dsp and dsp.beatTimes:
        values.extend(float(value) % bar_duration for value in dsp.beatTimes[:24])
    steps = max(8, beats * 8)
    values.extend((bar_duration * index) / steps for index in range(steps))
    return sorted(set(round(value, 5) for value in values))


def _tempo_segments(
    rhythm: RhythmDraft,
    duration: float,
    selected_bpm: float,
) -> tuple[TempoSegment, ...]:
    output: list[TempoSegment] = []
    for item in rhythm.tempoSegments:
        start = max(0.0, min(duration, _finite(item.startSeconds, -1.0)))
        end = max(0.0, min(duration, _finite(item.endSeconds, -1.0)))
        bpm = _finite(item.bpm, 0.0)
        confidence = max(0.0, min(1.0, _finite(item.confidence, 0.5)))
        if start < duration and end > start and 20.0 <= bpm <= 320.0:
            output.append(
                TempoSegment(
                    startSeconds=round(start, 3),
                    endSeconds=round(end, 3),
                    bpm=round(bpm, 3),
                    confidence=confidence,
                )
            )
    if not output:
        output.append(
            TempoSegment(
                startSeconds=0.0,
                endSeconds=max(0.001, duration),
                bpm=selected_bpm,
                confidence=max(0.0, min(1.0, _finite(rhythm.confidence, 0.5))),
            )
        )
    return tuple(sorted(output, key=lambda item: item.startSeconds))


def build_beat_grid(
    *,
    rhythm: RhythmDraft,
    sections: list[SectionStructureDraft],
    duration: float,
    dsp: DspSummary | None = None,
    profile: AccuracyProfile | None = None,
) -> BeatGrid:
    profile = profile or load_accuracy_profile()
    duration = max(0.001, _finite(duration, 0.001))
    signature = (
        str(rhythm.timeSignature)
        if _TIME_SIGNATURE_RE.fullmatch(str(rhythm.timeSignature or ""))
        else "4/4"
    )
    beats = beats_per_bar(signature)
    observed = list(dsp.beatTimes) if dsp else []
    best: tuple[float, float, float] | None = None
    selected_model_bpm = _finite(rhythm.selectedBpm, 0.0) if rhythm.selectedBpm else 0.0

    for bpm in _candidate_bpms(rhythm, dsp):
        integer_error = _integer_bar_error(bpm, beats, sections)
        for offset in _offset_candidates(bpm, beats, rhythm, dsp):
            beat_error = _beat_alignment_error(bpm, offset, observed)
            section_error = _section_alignment_error(bpm, beats, offset, sections)
            model_prior = 0.0
            if selected_model_bpm > 0:
                ratio = max(bpm, selected_model_bpm) / min(bpm, selected_model_bpm)
                model_prior = (
                    min(1.0, abs(math.log2(ratio)))
                    * profile.grid_model_prior_weight
                )
            score = 1.0 - min(
                1.0,
                integer_error * profile.grid_integer_bar_weight
                + beat_error * profile.grid_beat_alignment_weight
                + section_error * profile.grid_section_alignment_weight
                + model_prior,
            )
            if best is None or score > best[0]:
                best = (score, bpm, offset)

    assert best is not None
    score, bpm, offset = best
    beat_duration = 60.0 / bpm
    first_index = math.floor((0.0 - offset) / beat_duration)
    current = offset + first_index * beat_duration
    beat_times: list[float] = []
    while current <= duration + beat_duration:
        if 0.0 <= current <= duration:
            beat_times.append(round(current, 6))
        current += beat_duration

    bar_duration = beat_duration * beats
    first_bar_index = math.floor((0.0 - offset) / bar_duration)
    current_bar = offset + first_bar_index * bar_duration
    bar_starts: list[float] = []
    while current_bar <= duration + bar_duration:
        if 0.0 <= current_bar <= duration:
            bar_starts.append(round(current_bar, 6))
        current_bar += bar_duration

    return BeatGrid(
        bpm=round(bpm, 4),
        time_signature=signature,
        beats_per_bar=beats,
        beat_duration=beat_duration,
        downbeat_offset=round(offset, 6),
        duration=duration,
        beat_times=tuple(beat_times),
        bar_starts=tuple(bar_starts),
        score=round(score, 6),
        tempo_segments=_tempo_segments(rhythm, duration, bpm),
    )
