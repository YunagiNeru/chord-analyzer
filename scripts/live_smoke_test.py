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

import music_agent.pipeline_v2 as pipeline_v2_module
from music_agent import MusicCoordinatorAgent
from music_agent.validators import validate_invariants, validate_quality


def _chords(result) -> list:
    return [
        chord
        for section in result.sections
        for measure in section.measures
        for chord in measure.chords
    ]


def _rejected_path(output: Path) -> Path:
    suffix = output.suffix or ".json"
    return output.with_name(f"{output.stem}.rejected{suffix}")


def _result_summary(result, *, elapsed: float, status: str) -> dict[str, object]:
    invariant_errors = validate_invariants(result)
    quality_errors = validate_quality(result)
    chords = _chords(result)
    known_chords = [chord for chord in chords if chord.symbol not in {"X", "N"}]
    unresolved = [item for item in result.uncertainRanges if not item.resolved]
    unresolved_coverage = sum(
        item.endSeconds - item.startSeconds
        for item in unresolved
    )
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
        "sectionCount": len(result.sections),
        "chordCount": len(chords),
        "knownChordCount": len(known_chords),
        "uncertainRangeCount": len(result.uncertainRanges),
        "unresolvedRangeCount": len(unresolved),
        "unresolvedCoverageSeconds": round(unresolved_coverage, 3),
        "unresolvedCoverageRatio": round(
            unresolved_coverage / result.track.durationSeconds,
            6,
        ) if result.track.durationSeconds > 0 else 0.0,
        "unresolvedReasons": sorted(
            {
                item.reason
                for item in unresolved
            }
        ),
        "invariantErrors": invariant_errors,
        "qualityErrors": quality_errors,
        "diagnostics": result.diagnostics.model_dump() if result.diagnostics else None,
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

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is required")
    os.environ.setdefault("ANALYSIS_PIPELINE", "v2")
    os.environ.setdefault("GEMINI_MODEL", "gemini-3.5-flash")
    os.environ.setdefault("GEMINI_RESOLVER_MODEL", os.environ["GEMINI_MODEL"])
    os.environ.setdefault("MODEL_MAX_PARALLEL_CALLS", "4")
    os.environ.setdefault("MAX_RESOLVER_CALLS", "4")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}
    original_pipeline_validate_quality = pipeline_v2_module.validate_quality

    def capture_quality_result(result):
        captured["result"] = result
        return original_pipeline_validate_quality(result)

    pipeline_v2_module.validate_quality = capture_quality_result
    started = time.perf_counter()
    try:
        result = MusicCoordinatorAgent().analyze_youtube(url=args.youtube_url)
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
                    "rejectedResult": str(rejected_path.resolve()),
                }
            )
        else:
            summary = {
                "status": "failed",
                "elapsedSeconds": round(elapsed, 3),
                "errorType": type(exc).__name__,
                "message": str(exc),
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 10
    finally:
        pipeline_v2_module.validate_quality = original_pipeline_validate_quality

    elapsed = time.perf_counter() - started
    summary = _result_summary(result, elapsed=elapsed, status="passed")
    args.output.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"RESULT={args.output.resolve()}")

    if summary["invariantErrors"]:
        return 2
    if summary["qualityErrors"]:
        return 6
    if result.analysisVersion != "2.0":
        return 3
    if elapsed > 890.0:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
