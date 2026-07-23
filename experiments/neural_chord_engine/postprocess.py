from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from models import ChordSegment, NoChordRange, canonicalize_symbol


def merge_adjacent(
    segments: Sequence[ChordSegment],
    *,
    maximum_gap_seconds: float = 0.05,
) -> list[ChordSegment]:
    output: list[ChordSegment] = []

    for segment in segments:
        if (
            output
            and output[-1].symbol == segment.symbol
            and segment.start_seconds - output[-1].end_seconds
            <= maximum_gap_seconds
        ):
            previous = output[-1]
            total_duration = previous.duration_seconds + segment.duration_seconds
            confidence = (
                previous.confidence * previous.duration_seconds
                + segment.confidence * segment.duration_seconds
            ) / max(total_duration, 1e-9)

            output[-1] = ChordSegment(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, segment.end_seconds),
                symbol=previous.symbol,
                source=previous.source,
                confidence=float(max(0.0, min(1.0, confidence))),
                was_beat_snapped=(
                    previous.was_beat_snapped or segment.was_beat_snapped
                ),
            )
        else:
            output.append(segment)

    return output


def normalize_segments(
    segments: Iterable[ChordSegment],
    *,
    duration_seconds: float,
) -> list[ChordSegment]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    output: list[ChordSegment] = []

    for segment in sorted(
        segments,
        key=lambda item: (item.start_seconds, item.end_seconds),
    ):
        start = max(0.0, min(duration_seconds, float(segment.start_seconds)))
        end = max(start, min(duration_seconds, float(segment.end_seconds)))

        if end <= start + 1e-6:
            continue

        normalized = ChordSegment(
            start_seconds=round(start, 6),
            end_seconds=round(end, 6),
            symbol=canonicalize_symbol(segment.symbol),
            source=segment.source,
            confidence=segment.confidence,
            was_beat_snapped=segment.was_beat_snapped,
        )

        if output and normalized.start_seconds < output[-1].end_seconds:
            previous = output[-1]
            clipped_end = normalized.start_seconds

            if clipped_end > previous.start_seconds + 1e-6:
                output[-1] = ChordSegment(
                    start_seconds=previous.start_seconds,
                    end_seconds=clipped_end,
                    symbol=previous.symbol,
                    source=previous.source,
                    confidence=previous.confidence,
                    was_beat_snapped=previous.was_beat_snapped,
                )
            else:
                output.pop()

        output.append(normalized)

    return merge_adjacent(output)


def remove_short_isolated(
    segments: Sequence[ChordSegment],
    *,
    minimum_duration_seconds: float = 0.35,
) -> list[ChordSegment]:
    working = list(segments)

    if len(working) < 2:
        return working

    changed = True

    while changed and len(working) >= 2:
        changed = False

        for index, segment in enumerate(working):
            if (
                segment.duration_seconds >= minimum_duration_seconds
                or segment.symbol in {"N", "X"}
            ):
                continue

            previous = working[index - 1] if index > 0 else None
            following = working[index + 1] if index + 1 < len(working) else None

            if previous and following and previous.symbol == following.symbol:
                working[index - 1 : index + 2] = [
                    ChordSegment(
                        start_seconds=previous.start_seconds,
                        end_seconds=following.end_seconds,
                        symbol=previous.symbol,
                        source=previous.source,
                        confidence=min(
                            previous.confidence,
                            following.confidence,
                        ),
                    )
                ]
                changed = True
                break

            if previous is None and following is not None:
                working[0:2] = [
                    ChordSegment(
                        start_seconds=segment.start_seconds,
                        end_seconds=following.end_seconds,
                        symbol=following.symbol,
                        source=following.source,
                        confidence=following.confidence,
                    )
                ]
                changed = True
                break

            if following is None and previous is not None:
                working[index - 1 : index + 1] = [
                    ChordSegment(
                        start_seconds=previous.start_seconds,
                        end_seconds=segment.end_seconds,
                        symbol=previous.symbol,
                        source=previous.source,
                        confidence=previous.confidence,
                    )
                ]
                changed = True
                break

            if previous and following:
                if previous.duration_seconds >= following.duration_seconds:
                    working[index - 1 : index + 1] = [
                        ChordSegment(
                            start_seconds=previous.start_seconds,
                            end_seconds=segment.end_seconds,
                            symbol=previous.symbol,
                            source=previous.source,
                            confidence=previous.confidence,
                        )
                    ]
                else:
                    working[index : index + 2] = [
                        ChordSegment(
                            start_seconds=segment.start_seconds,
                            end_seconds=following.end_seconds,
                            symbol=following.symbol,
                            source=following.source,
                            confidence=following.confidence,
                        )
                    ]

                changed = True
                break

    return merge_adjacent(working)


def snap_to_beats(
    segments: Sequence[ChordSegment],
    beats: Sequence[float],
    *,
    maximum_distance_seconds: float = 0.15,
    minimum_duration_seconds: float = 0.25,
) -> list[ChordSegment]:
    if not segments or not beats:
        return list(segments)

    boundaries = [segments[0].start_seconds]
    boundaries.extend(segment.end_seconds for segment in segments)
    snapped_boundaries = list(boundaries)
    flags = [False] * len(boundaries)

    for index in range(1, len(boundaries) - 1):
        original = boundaries[index]
        candidate = min(beats, key=lambda beat: abs(float(beat) - original))
        candidate = float(candidate)

        if abs(candidate - original) > maximum_distance_seconds:
            continue
        if candidate - snapped_boundaries[index - 1] < minimum_duration_seconds:
            continue
        if boundaries[index + 1] - candidate < minimum_duration_seconds:
            continue

        snapped_boundaries[index] = candidate
        flags[index] = True

    output: list[ChordSegment] = []

    for index, segment in enumerate(segments):
        start = snapped_boundaries[index]
        end = snapped_boundaries[index + 1]

        if end <= start:
            continue

        output.append(
            ChordSegment(
                start_seconds=start,
                end_seconds=end,
                symbol=segment.symbol,
                source=segment.source,
                confidence=segment.confidence,
                was_beat_snapped=flags[index] or flags[index + 1],
            )
        )

    return merge_adjacent(output)


def apply_no_chord_ranges(
    segments: Sequence[ChordSegment],
    ranges: Sequence[NoChordRange],
) -> list[ChordSegment]:
    if not ranges:
        return list(segments)

    boundaries = {
        point
        for segment in segments
        for point in (segment.start_seconds, segment.end_seconds)
    }
    boundaries.update(
        point
        for item in ranges
        for point in (item.start_seconds, item.end_seconds)
    )
    ordered = sorted(boundaries)
    output: list[ChordSegment] = []

    for start, end in zip(ordered[:-1], ordered[1:], strict=True):
        if end <= start:
            continue

        midpoint = (start + end) / 2.0
        silence = next(
            (
                item
                for item in ranges
                if item.start_seconds <= midpoint < item.end_seconds
            ),
            None,
        )

        if silence is not None:
            output.append(
                ChordSegment(
                    start_seconds=start,
                    end_seconds=end,
                    symbol="N",
                    source="no_chord",
                    confidence=silence.confidence,
                )
            )
            continue

        original = next(
            (
                segment
                for segment in segments
                if segment.start_seconds <= midpoint < segment.end_seconds
            ),
            None,
        )

        if original is None:
            continue

        output.append(
            ChordSegment(
                start_seconds=start,
                end_seconds=end,
                symbol=original.symbol,
                source=original.source,
                confidence=original.confidence,
                was_beat_snapped=original.was_beat_snapped,
            )
        )

    return merge_adjacent(output)
