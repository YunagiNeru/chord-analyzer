from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from music_agent.chord_symbol import parse_chord
from music_agent.schemas import AnalysisResult
from music_agent.validators import validate_invariants


@dataclass(frozen=True, slots=True)
class Label:
    start: float
    end: float
    symbol: str


def read_lab(path: Path) -> list[Label]:
    output: list[Label] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: start end chord の3列が必要です。")
        start, end = float(parts[0]), float(parts[1])
        if end <= start:
            raise ValueError(f"{path}:{line_number}: end must be greater than start")
        output.append(Label(start, end, parts[2]))
    if not output:
        raise ValueError("reference labels are empty")
    return output


def prediction_labels(result: AnalysisResult) -> list[Label]:
    return [
        Label(chord.startSeconds, chord.endSeconds, chord.symbol)
        for section in result.sections
        for measure in section.measures
        for chord in measure.chords
    ]


def label_at(labels: list[Label], time: float) -> str:
    for item in labels:
        if item.start <= time < item.end:
            return item.symbol
    return "N"


def weighted_scores(reference: list[Label], predicted: list[Label]) -> tuple[float, float]:
    boundaries = sorted(
        {
            value
            for item in reference + predicted
            for value in (item.start, item.end)
        }
    )
    root_correct = 0.0
    majmin_correct = 0.0
    total = 0.0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        ref = parse_chord(label_at(reference, midpoint))
        pred = parse_chord(label_at(predicted, midpoint))
        duration = end - start
        total += duration
        if ref.no_chord and pred.no_chord:
            root_correct += duration
            majmin_correct += duration
            continue
        if ref.root_pc == pred.root_pc and ref.root_pc is not None:
            root_correct += duration
            if ref.quality == pred.quality:
                majmin_correct += duration
    return (
        root_correct / total if total else 0.0,
        majmin_correct / total if total else 0.0,
    )


def boundary_recall(
    reference: list[Label],
    predicted: list[Label],
    *,
    tolerance_seconds: float,
) -> float:
    reference_boundaries = [item.start for item in reference[1:]]
    predicted_boundaries = [item.start for item in predicted[1:]]
    if not reference_boundaries:
        return 1.0
    matched = sum(
        any(
            abs(reference_time - predicted_time) <= tolerance_seconds
            for predicted_time in predicted_boundaries
        )
        for reference_time in reference_boundaries
    )
    return matched / len(reference_boundaries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = AnalysisResult.model_validate_json(args.prediction.read_text(encoding="utf-8"))
    reference = read_lab(args.reference)
    predicted = prediction_labels(result)
    root, majmin = weighted_scores(reference, predicted)
    bpm = result.track.bpm or 120.0
    half_beat = 30.0 / bpm
    invariant_errors = validate_invariants(result)
    boundary_score = boundary_recall(
        reference,
        predicted,
        tolerance_seconds=half_beat,
    )
    payload = {
        "rootWeightedAccuracy": round(root, 6),
        "majorMinorWeightedAccuracy": round(majmin, 6),
        "changeBoundaryRecallHalfBeat": round(boundary_score, 6),
        "invariantErrors": invariant_errors,
        "gates": {
            "root": root >= 0.85,
            "majorMinor": majmin >= 0.80,
            "boundary": boundary_score >= 0.80,
            "invariants": not invariant_errors,
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(payload["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
