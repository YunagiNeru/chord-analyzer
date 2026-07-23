from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from youtube_source import download_youtube_audio


def _write_json(path: Path, payload: Any) -> None:
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
            "YouTube URLから実音声を取得し、"
            "現行DSPとOmnizart + Beat This!をA/B比較します。"
        )
    )
    parser.add_argument(
        "--youtube-url",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ab-test-output/youtube"),
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
        "--max-duration-seconds",
        type=float,
        default=1_800.0,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/chord-neural-youtube/current"),
    )
    parser.add_argument(
        "--skip-legacy",
        action="store_true",
    )
    parser.add_argument(
        "--skip-beat-this",
        action="store_true",
    )
    parser.add_argument(
        "--keep-work-audio",
        action="store_true",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    evaluate_script = script_dir / "evaluate.py"
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.work_dir.resolve()

    source = download_youtube_audio(
        args.youtube_url,
        work_dir,
        max_duration_seconds=args.max_duration_seconds,
    )

    source_payload = {
        "type": "youtube",
        "videoId": source.video_id,
        "title": source.title,
        "uploader": source.uploader,
        "webpageUrl": source.webpage_url,
        "durationSeconds": round(source.duration_seconds, 3),
        "temporaryAudioPath": str(source.audio_path),
    }
    _write_json(
        output_dir / "youtube-source.json",
        source_payload,
    )

    command = [
        sys.executable,
        str(evaluate_script),
        "--audio",
        str(source.audio_path),
        "--output",
        str(output_dir),
        "--beat-model",
        args.beat_model,
    ]

    if args.reference:
        command.extend(
            ["--reference", str(args.reference.resolve())]
        )
    if args.skip_legacy:
        command.append("--skip-legacy")
    if args.skip_beat_this:
        command.append("--skip-beat-this")

    try:
        subprocess.run(command, check=True)

        report_path = output_dir / "report.json"
        if report_path.is_file():
            report = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            report["source"] = source_payload
            _write_json(report_path, report)

        print(f"\nYOUTUBE_SOURCE={output_dir / 'youtube-source.json'}")
        print(f"REPORT={report_path}")
        return 0
    finally:
        if not args.keep_work_audio:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
