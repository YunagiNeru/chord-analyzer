from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from music_agent import MusicCoordinatorAgent
from music_agent.validators import validate_invariants, validate_quality


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

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is required")
    os.environ.setdefault("ANALYSIS_PIPELINE", "v2")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash")
    os.environ.setdefault("GEMINI_RESOLVER_MODEL", os.environ["GEMINI_MODEL"])
    os.environ.setdefault("MODEL_MAX_PARALLEL_CALLS", "4")
    os.environ.setdefault("MAX_RESOLVER_CALLS", "4")

    started = time.perf_counter()
    try:
        result = MusicCoordinatorAgent().analyze_youtube(url=args.youtube_url)
    except Exception as exc:  # noqa: BLE001 - command boundary
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "status": "failed",
                    "elapsedSeconds": round(elapsed, 3),
                    "errorType": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 10

    elapsed = time.perf_counter() - started
    invariant_errors = validate_invariants(result)
    quality_errors = validate_quality(result)
    chord_count = sum(
        len(measure.chords)
        for section in result.sections
        for measure in section.measures
    )
    known_chord_count = sum(
        chord.symbol not in {"X", "N"}
        for section in result.sections
        for measure in section.measures
        for chord in measure.chords
    )
    summary = {
        "status": "passed" if not invariant_errors and not quality_errors else "failed",
        "elapsedSeconds": round(elapsed, 3),
        "analysisVersion": result.analysisVersion,
        "title": result.track.title,
        "artist": result.track.artist,
        "durationSeconds": result.track.durationSeconds,
        "bpm": result.track.bpm,
        "timeSignature": result.track.timeSignature,
        "globalKey": result.track.globalKey,
        "sectionCount": len(result.sections),
        "chordCount": chord_count,
        "knownChordCount": known_chord_count,
        "uncertainRangeCount": len(result.uncertainRanges),
        "unresolvedRangeCount": sum(not item.resolved for item in result.uncertainRanges),
        "invariantErrors": invariant_errors,
        "qualityErrors": quality_errors,
        "diagnostics": result.diagnostics.model_dump() if result.diagnostics else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"RESULT={args.output.resolve()}")

    if invariant_errors:
        return 2
    if quality_errors:
        return 6
    if result.analysisVersion != "2.0":
        return 3
    if elapsed > 890.0:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
