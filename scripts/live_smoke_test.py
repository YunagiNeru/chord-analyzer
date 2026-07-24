from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import music_agent.pipeline_v2 as pipeline_v2_module
from music_agent import MusicCoordinatorAgent
from music_agent.chord_symbol import canonicalize_symbol
from music_agent.harmonic_reconcile import validate_tempo_coverage
from music_agent.validators import validate_invariants


def _display_chords(result) -> list:
    return [
        chord
        for section in result.sections
        for measure in section.measures
        for chord in measure.chords
    ]


def _actual_chord_state_count(result) -> int:
    count = 0
    for section in result.sections:
        previous_symbol: str | None = None
        previous_end = -1.0
        events = sorted(
            [
                chord
                for measure in section.measures
                for chord in measure.chords
            ],
            key=lambda item: (
                item.startSeconds,
                item.endSeconds,
                item.symbol,
            ),
        )
        for chord in events:
            symbol = canonicalize_symbol(chord.symbol)
            if (
                symbol != previous_symbol
                or chord.startSeconds > previous_end + 0.003
            ):
                count += 1
            previous_symbol = symbol
            previous_end = max(previous_end, chord.endSeconds)
    return count


def _rejected_path(output: Path) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.rejected{suffix}")


def _boundary_distance_beats(result, seconds: float) -> float:
    bpm = result.track.bpm or 120.0
    beat_duration = 60.0 / bpm
    offset = result.downbeatOffsetSeconds or 0.0
    phase = ((seconds - offset) / beat_duration) % 1.0
    return min(phase, 1.0 - phase)


def _beats_per_bar(result) -> int:
    signature = str(result.track.timeSignature or "4/4")
    try:
        return max(1, int(signature.split("/", 1)[0]))
    except (TypeError, ValueError, IndexError):
        return 4


def _section_bars(result, section) -> float:
    bpm = result.track.bpm or 120.0
    bar_duration = (60.0 / bpm) * _beats_per_bar(result)
    return (
        section.endSeconds - section.startSeconds
    ) / max(0.001, bar_duration)


def _maximum_section_bars(result) -> float:
    return max(
        (_section_bars(result, section) for section in result.sections),
        default=0.0,
    )


def _merged_section_state_count(section) -> int:
    count = 0
    previous_symbol: str | None = None
    previous_end = -1.0
    events = sorted(
        [
            chord
            for measure in section.measures
            for chord in measure.chords
        ],
        key=lambda item: (
            item.startSeconds,
            item.endSeconds,
            item.symbol,
        ),
    )
    for chord in events:
        symbol = canonicalize_symbol(chord.symbol)
        if (
            symbol != previous_symbol
            or chord.startSeconds > previous_end + 0.003
        ):
            count += 1
        previous_symbol = symbol
        previous_end = max(previous_end, chord.endSeconds)
    return count


def _suspicious_static_sections(result) -> list[dict[str, object]]:
    major_types = {"verse", "pre_chorus", "chorus", "post_chorus"}
    output: list[dict[str, object]] = []
    for section in result.sections:
        bars = _section_bars(result, section)
        agreement = section.agreement if section.agreement is not None else 0.0
        states = _merged_section_state_count(section)
        if (
            section.type in major_types
            and bars >= 7.5
            and states <= 1
            and agreement < 0.60
        ):
            output.append(
                {
                    "id": section.id,
                    "name": section.name,
                    "bars": round(bars, 4),
                    "stateCount": states,
                    "agreement": round(agreement, 4),
                }
            )
    return output


def _section_key_modes(result) -> list[dict[str, object]]:
    return [
        {
            "id": section.id,
            "name": section.name,
            "type": section.type,
            "startSeconds": section.startSeconds,
            "endSeconds": section.endSeconds,
            "key": section.key,
            "mode": section.mode,
        }
        for section in result.sections
    ]


def _reference_facts(result) -> list[dict[str, object]]:
    return [
        {
            "title": source.title,
            "url": source.url,
            "facts": list(source.facts),
        }
        for source in result.referenceSources
    ]


def _result_summary(result, *, elapsed: float, status: str) -> dict[str, object]:
    invariant_errors = validate_invariants(result)
    invariant_errors.extend(validate_tempo_coverage(result))
    invariant_errors = sorted(set(invariant_errors))
    quality_errors = pipeline_v2_module.validate_quality(result)
    display_chords = _display_chords(result)
    known_chords = [
        chord
        for chord in display_chords
        if canonicalize_symbol(chord.symbol) not in {"X", "N"}
    ]
    unresolved = [
        item
        for item in result.uncertainRanges
        if not item.resolved
    ]
    unresolved_coverage = sum(
        item.endSeconds - item.startSeconds
        for item in unresolved
    )
    diagnostics = result.diagnostics
    required_failures = (
        diagnostics.requiredModelFailures
        if diagnostics
        else []
    )
    section_starts = [
        section.startSeconds
        for section in result.sections
        if section.startSeconds > 0.035
    ]
    maximum_boundary_error = max(
        (
            _boundary_distance_beats(result, value)
            for value in section_starts
        ),
        default=0.0,
    )
    merged_sections = [
        section.id
        for section in result.sections
        if "+" in section.id
        or "〜" in section.name
        or "セクション数上限" in section.summary
    ]
    actual_state_count = _actual_chord_state_count(result)
    return {
        "status": status,
        "elapsedSeconds": round(elapsed, 3),
        "analysisVersion": result.analysisVersion,
        "title": result.track.title,
        "artist": result.track.artist,
        "durationSeconds": result.track.durationSeconds,
        "bpm": result.track.bpm,
        "timeSignature": result.track.timeSignature,
        "globalKey": result.track.globalKey,
        "globalMode": result.track.globalMode,
        "tempoSegments": [
            item.model_dump()
            for item in result.tempoSegments
        ],
        "sectionCount": len(result.sections),
        "sectionKeyModes": _section_key_modes(result),
        "mergedSections": merged_sections,
        "maximumSectionBars": round(
            _maximum_section_bars(result),
            4,
        ),
        "maximumBoundaryErrorBeats": round(
            maximum_boundary_error,
            6,
        ),
        "chordStateCount": actual_state_count,
        "diagnosticChordStateCount": (
            diagnostics.chordStateCount
            if diagnostics
            else None
        ),
        "displayChordEventCount": len(display_chords),
        "knownDisplayChordEventCount": len(known_chords),
        "suspiciousStaticSections": _suspicious_static_sections(result),
        "uncertainRangeCount": len(result.uncertainRanges),
        "unresolvedRangeCount": len(unresolved),
        "unresolvedCoverageSeconds": round(
            unresolved_coverage,
            3,
        ),
        "unresolvedCoverageRatio": round(
            unresolved_coverage / result.track.durationSeconds,
            6,
        ) if result.track.durationSeconds > 0 else 0.0,
        "unresolvedReasons": sorted(
            {item.reason for item in unresolved}
        ),
        "referenceFacts": _reference_facts(result),
        "requiredModelFailures": required_failures,
        "invariantErrors": invariant_errors,
        "qualityErrors": quality_errors,
        "diagnostics": (
            diagnostics.model_dump()
            if diagnostics
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Accuracy V2 pipeline against a real YouTube URL."
    )
    parser.add_argument("--youtube-url", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("accuracy-artifacts/live-smoke-result.json"),
    )
    args = parser.parse_args()

    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
    )
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is required")
    os.environ.setdefault("ANALYSIS_PIPELINE", "v2")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash")
    os.environ.setdefault(
        "GEMINI_RESOLVER_MODEL",
        os.environ["GEMINI_MODEL"],
    )
    os.environ.setdefault("MODEL_MAX_PARALLEL_CALLS", "4")
    os.environ.setdefault("MAX_RESOLVER_CALLS", "12")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}
    original_pipeline_validate_quality = (
        pipeline_v2_module.validate_quality
    )

    def capture_quality_result(result):
        captured["result"] = result
        return original_pipeline_validate_quality(result)

    pipeline_v2_module.validate_quality = capture_quality_result
    started = time.perf_counter()
    try:
        result = MusicCoordinatorAgent().analyze_youtube(
            url=args.youtube_url
        )
    except Exception as exc:  # noqa: BLE001 - command boundary
        elapsed = time.perf_counter() - started
        rejected_result = captured.get("result")
        if rejected_result is not None:
            rejected_path = _rejected_path(args.output)
            rejected_path.write_text(
                rejected_result.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            summary = _result_summary(
                rejected_result,
                elapsed=elapsed,
                status="rejected",
            )
            summary.update(
                {
                    "errorType": type(exc).__name__,
                    "message": str(exc),
                    "rejectedResult": str(
                        rejected_path.resolve()
                    ),
                }
            )
        else:
            summary = {
                "status": "failed",
                "elapsedSeconds": round(elapsed, 3),
                "errorType": type(exc).__name__,
                "message": str(exc),
            }
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 10
    finally:
        pipeline_v2_module.validate_quality = (
            original_pipeline_validate_quality
        )

    elapsed = time.perf_counter() - started
    summary = _result_summary(
        result,
        elapsed=elapsed,
        status="passed",
    )
    args.output.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"RESULT={args.output.resolve()}")

    if summary["requiredModelFailures"]:
        return 7
    if summary["invariantErrors"]:
        return 2
    if summary["qualityErrors"]:
        return 6
    if result.analysisVersion != "2.0":
        return 3
    if not math.isfinite(elapsed) or elapsed > 890.0:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
