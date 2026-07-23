from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np

from engines import (
    detect_silence_ranges,
    engine_versions,
    run_beat_this,
    run_legacy,
    run_omnizart,
)
from models import ChordSegment, canonicalize_symbol
from postprocess import (
    apply_no_chord_ranges,
    normalize_segments,
    remove_short_isolated,
    snap_to_beats,
)


def load_reference_lab(path: Path) -> list[ChordSegment]:
    segments: list[ChordSegment] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=2)

            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_number}: start end chord の3列が必要です。"
                )

            start = float(parts[0])
            end = float(parts[1])

            segments.append(
                ChordSegment(
                    start_seconds=start,
                    end_seconds=end,
                    symbol=canonicalize_symbol(parts[2]),
                    source="reference",
                )
            )

    if not segments:
        raise ValueError(f"正解ラベルが空です: {path}")

    return segments


def fill_gaps(
    segments: Sequence[ChordSegment],
    *,
    duration_seconds: float,
) -> list[ChordSegment]:
    output: list[ChordSegment] = []
    cursor = 0.0

    for segment in normalize_segments(
        segments,
        duration_seconds=duration_seconds,
    ):
        if segment.start_seconds > cursor + 1e-6:
            output.append(
                ChordSegment(
                    start_seconds=cursor,
                    end_seconds=segment.start_seconds,
                    symbol="N",
                    source="gap",
                )
            )

        output.append(segment)
        cursor = segment.end_seconds

    if cursor < duration_seconds - 1e-6:
        output.append(
            ChordSegment(
                start_seconds=cursor,
                end_seconds=duration_seconds,
                symbol="N",
                source="gap",
            )
        )

    return output


def to_harte_label(symbol: str) -> str:
    value = canonicalize_symbol(symbol)

    if value in {"N", "X"}:
        return value

    if value.endswith("maj7"):
        return f"{value[:-4]}:maj7"
    if value.endswith("m7"):
        return f"{value[:-2]}:min7"
    if value.endswith("sus4"):
        return f"{value[:-4]}:sus4"
    if value.endswith("dim"):
        return f"{value[:-3]}:dim"
    if value.endswith("7"):
        return f"{value[:-1]}:7"
    if value.endswith("m"):
        return f"{value[:-1]}:min"

    return f"{value}:maj"


def evaluate_with_mir_eval(
    reference: Sequence[ChordSegment],
    estimate: Sequence[ChordSegment],
    *,
    duration_seconds: float,
) -> dict[str, float]:
    import mir_eval

    reference_filled = fill_gaps(
        reference,
        duration_seconds=duration_seconds,
    )
    estimate_filled = fill_gaps(
        estimate,
        duration_seconds=duration_seconds,
    )

    reference_intervals = np.asarray(
        [
            [item.start_seconds, item.end_seconds]
            for item in reference_filled
        ],
        dtype=float,
    )
    estimate_intervals = np.asarray(
        [
            [item.start_seconds, item.end_seconds]
            for item in estimate_filled
        ],
        dtype=float,
    )
    reference_labels = [
        to_harte_label(item.symbol)
        for item in reference_filled
    ]
    estimate_labels = [
        to_harte_label(item.symbol)
        for item in estimate_filled
    ]

    metrics = mir_eval.chord.evaluate(
        reference_intervals,
        reference_labels,
        estimate_intervals,
        estimate_labels,
    )

    return {
        key: round(float(value), 6)
        for key, value in metrics.items()
    }


def summarize_segments(
    segments: Sequence[ChordSegment],
) -> dict[str, Any]:
    return {
        "segmentCount": len(segments),
        "shortSegmentCount": sum(
            item.duration_seconds < 0.35
            for item in segments
        ),
        "beatSnappedCount": sum(
            item.was_beat_snapped
            for item in segments
        ),
        "symbols": sorted(
            {item.symbol for item in segments}
        ),
        "totalCoveredSeconds": round(
            sum(item.duration_seconds for item in segments),
            3,
        ),
    }


def estimate_tempo(
    beats: Sequence[float],
) -> float | None:
    if len(beats) < 3:
        return None

    intervals = np.diff(
        np.asarray(beats, dtype=float)
    )
    intervals = intervals[
        intervals > 0.05
    ]

    if intervals.size == 0:
        return None

    return round(
        float(60.0 / np.median(intervals)),
        3,
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "現行DSPとOmnizart + Beat This!をA/B比較します。"
        )
    )
    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ab-test-output"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
    )
    parser.add_argument(
        "--beat-model",
        default="small0",
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
    )
    parser.add_argument(
        "--skip-beat-this",
        action="store_true",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    audio_path = args.audio.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    duration_seconds = float(
        librosa.get_duration(
            path=audio_path,
        )
    )

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise RuntimeError(
            "音声の長さを取得できませんでした。"
        )

    report: dict[str, Any] = {
        "audio": str(audio_path),
        "durationSeconds": round(
            duration_seconds,
            3,
        ),
        "beatModel": args.beat_model,
        "versions": engine_versions(),
        "engines": {},
    }

    reference = (
        load_reference_lab(
            args.reference.resolve()
        )
        if args.reference
        else None
    )

    if not args.skip_legacy:
        legacy, legacy_seconds = run_legacy(
            audio_path,
            repository_root,
        )
        legacy = normalize_segments(
            legacy,
            duration_seconds=duration_seconds,
        )
        write_json(
            output_dir / "legacy.json",
            [item.to_dict() for item in legacy],
        )
        report["engines"]["legacy"] = {
            **summarize_segments(legacy),
            "processingSeconds": round(
                legacy_seconds,
                3,
            ),
        }

        if reference:
            report["engines"]["legacy"]["metrics"] = (
                evaluate_with_mir_eval(
                    reference,
                    legacy,
                    duration_seconds=duration_seconds,
                )
            )

    omnizart_raw, omnizart_seconds, csv_path = run_omnizart(
        audio_path,
        output_dir / "omnizart",
    )
    omnizart_raw = normalize_segments(
        omnizart_raw,
        duration_seconds=duration_seconds,
    )

    if args.skip_beat_this:
        beats: list[float] = []
        downbeats: list[float] = []
        beat_seconds = 0.0
    else:
        beats, downbeats, beat_seconds = run_beat_this(
            audio_path,
            args.beat_model,
        )

    no_chord_ranges = detect_silence_ranges(
        audio_path,
        duration_seconds=duration_seconds,
    )
    processed = remove_short_isolated(
        omnizart_raw
    )
    processed = snap_to_beats(
        processed,
        beats,
    )
    processed = apply_no_chord_ranges(
        processed,
        no_chord_ranges,
    )
    processed = normalize_segments(
        processed,
        duration_seconds=duration_seconds,
    )

    write_json(
        output_dir / "omnizart-raw.json",
        [item.to_dict() for item in omnizart_raw],
    )
    write_json(
        output_dir / "omnizart-processed.json",
        [item.to_dict() for item in processed],
    )
    write_json(
        output_dir / "timing.json",
        {
            "beats": beats,
            "downbeats": downbeats,
            "estimatedBpm": estimate_tempo(beats),
            "noChordRanges": [
                item.to_dict()
                for item in no_chord_ranges
            ],
        },
    )

    report["engines"]["omnizartRaw"] = {
        **summarize_segments(omnizart_raw),
        "processingSeconds": round(
            omnizart_seconds,
            3,
        ),
        "csvPath": str(csv_path),
    }
    report["engines"]["omnizartProcessed"] = {
        **summarize_segments(processed),
        "beatProcessingSeconds": round(
            beat_seconds,
            3,
        ),
        "beatCount": len(beats),
        "downbeatCount": len(downbeats),
        "estimatedBpm": estimate_tempo(beats),
    }

    if reference:
        report["engines"]["omnizartRaw"]["metrics"] = (
            evaluate_with_mir_eval(
                reference,
                omnizart_raw,
                duration_seconds=duration_seconds,
            )
        )
        report["engines"]["omnizartProcessed"]["metrics"] = (
            evaluate_with_mir_eval(
                reference,
                processed,
                duration_seconds=duration_seconds,
            )
        )

    write_json(
        output_dir / "report.json",
        report,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        f"\nREPORT={output_dir / 'report.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
