from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from music_agent.review_rebuilder import rebuild_review_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repair BPM, tempoSegments, bar numbers, and beat positions in an "
            "existing Accuracy V2 review JSON without calling Vertex AI."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("input JSON root must be an object")

    rebuilt = rebuild_review_snapshot(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "track": rebuilt.get("track"),
                "rebuildDiagnostics": rebuilt.get("rebuildDiagnostics"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
