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

from google import genai
from google.genai import types

from music_agent.accuracy_v2_finalization import SemanticStructureRefinementDraft
from music_agent.diagnostics import DiagnosticsRecorder
from music_agent.model_gateway import ModelGateway
from music_agent.schemas import CompactResolutionDraft, CompactSpecialistDraft


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the live Vertex AI structured-output contracts used by "
            "Accuracy V2 specialists, resolvers, and semantic refinement."
        )
    )
    parser.parse_args()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is required")

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )
    diagnostics = DiagnosticsRecorder(pipeline="accuracy-v2-schema-probe")
    gateway = ModelGateway(
        client=client,
        model=model,
        diagnostics=diagnostics,
        max_parallel_calls=1,
    )

    started = time.perf_counter()
    try:
        specialist = gateway.generate_typed(
            contents=[
                "Synthetic contract probe only. Return one C major chord state "
                "from 0.0 to 2.0 seconds with high confidence."
            ],
            schema=CompactSpecialistDraft,
            system_instruction=(
                "Return only the requested compact chord-state JSON. "
                "Do not add prose or extra fields."
            ),
            temperature=0.0,
            max_output_tokens=512,
            retries=2,
            diagnostic_label="probe-compact-specialist",
        )
        resolution = gateway.generate_typed(
            contents=[
                "Synthetic contract probe only. Choose C from the candidates C and G."
            ],
            schema=CompactResolutionDraft,
            system_instruction=(
                "Return only one compact resolver decision. "
                "chosenSymbol must be C and confidence must be a finite number."
            ),
            temperature=0.0,
            max_output_tokens=512,
            retries=2,
            diagnostic_label="probe-compact-resolution",
        )
        semantic = gateway.generate_typed(
            contents=[
                "Synthetic contract probe only. Split 0-32 seconds into two meaningful "
                "song sections. The first is an instrumental intro from 0-16 seconds, "
                "the second is a vocal verse from 16-32 seconds. Give each a distinct "
                "name and summary."
            ],
            schema=SemanticStructureRefinementDraft,
            system_instruction=(
                "Return only semantic song-section JSON. Never use arbitrary A1/A2 names. "
                "Every section requires a distinct summary."
            ),
            temperature=0.0,
            max_output_tokens=6_144,
            retries=2,
            diagnostic_label="probe-semantic-refinement",
        )
    except Exception as exc:  # noqa: BLE001 - command boundary
        print(
            json.dumps(
                {
                    "status": "failed",
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "model": model,
                    "errorType": type(exc).__name__,
                    "message": str(exc),
                    "diagnostics": diagnostics.build([]).model_dump(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 10

    specialist_payload = specialist.model_dump()
    resolution_payload = resolution.model_dump()
    semantic_payload = semantic.model_dump()
    valid = (
        bool(specialist.chords)
        and specialist.chords[0].endSeconds > specialist.chords[0].startSeconds
        and resolution.chosenSymbol == "C"
        and len(semantic.sections) >= 2
        and len({item.summary for item in semantic.sections if item.summary}) >= 2
    )
    print(
        json.dumps(
            {
                "status": "passed" if valid else "failed",
                "elapsedSeconds": round(time.perf_counter() - started, 3),
                "model": model,
                "compactSpecialist": specialist_payload,
                "compactResolution": resolution_payload,
                "semanticRefinement": semantic_payload,
                "diagnostics": diagnostics.build([]).model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if valid else 11


if __name__ == "__main__":
    raise SystemExit(main())
