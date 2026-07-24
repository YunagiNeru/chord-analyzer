from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import mimetypes
import os
import socket
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from music_agent import MusicCoordinatorAgent
from music_agent.agents import SourceInspectorAgent
from music_agent.schemas import AnalysisResult, QuestionRequest, QuestionResponse

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_MODEL_AUDIO_BYTES = 14 * 1024 * 1024
MAX_DURATION_SECONDS = 12 * 60
DOWNLOAD_REDIRECT_LIMIT = 3
ALLOWED_AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".m4a",
    ".aif",
    ".aiff",
    ".ogg",
    ".opus",
    ".webm",
}

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("chord-analyzer")

app = FastAPI(
    title="Music Chord Analyzer",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@lru_cache(maxsize=1)
def get_coordinator() -> MusicCoordinatorAgent:
    return MusicCoordinatorAgent()


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin-allow-popups"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://www.youtube.com https://s.ytimg.com https://www.gstatic.com https://www.google.com; "
        "style-src 'self' 'unsafe-inline' https://www.gstatic.com https://www.google.com; "
        "img-src 'self' data: blob: https:; "
        "media-src 'self' blob: https:; "
        "frame-src https://www.youtube.com https://www.youtube-nocookie.com https://www.google.com; "
        "connect-src 'self' https://www.youtube.com https://www.google.com https://www.gstatic.com; "
        "font-src 'self' data:; "
        "object-src 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


@app.exception_handler(ValidationError)
async def validation_error_handler(_: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "message": str(exc)},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(INDEX_FILE, media_type="text/html; charset=utf-8")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "music-chord-analyzer",
        "version": app.version,
        "pipeline": os.environ.get("ANALYSIS_PIPELINE", "v2"),
        "release": os.environ.get("APP_RELEASE_SHA", "development"),
    }


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze(
    source_url: str | None = Form(default=None),
    audio_file: UploadFile | None = File(default=None),
) -> AnalysisResult:
    source_url = (source_url or "").strip()
    if bool(source_url) == bool(audio_file):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "choose_one_source",
                "message": "URLまたは音声ファイルのどちらか一方だけを指定してください。",
            },
        )

    try:
        if source_url:
            return await _analyze_url(source_url)
        assert audio_file is not None
        return await _analyze_upload(audio_file)
    except HTTPException:
        raise
    except subprocess.CalledProcessError as exc:
        logger.warning("ffmpeg/ffprobe failed: %s", exc.stderr)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_audio",
                "message": "音声ファイルを読み取れませんでした。対応形式か、ファイルが破損していないか確認してください。",
            },
        ) from exc
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "analysis_input_error", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("analysis failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "analysis_failed",
                "message": (
                    "解析に失敗しました。音源へのアクセス、Vertex AIの一時障害、"
                    "または品質基準を満たさない解析結果が原因の可能性があります。"
                    "時間を置いて再実行するか、別の音源を指定してください。"
                ),
                "errorType": type(exc).__name__,
            },
        ) from exc


@app.post("/api/question", response_model=QuestionResponse)
async def question(payload: QuestionRequest) -> QuestionResponse:
    try:
        coordinator = get_coordinator()
        return await asyncio.to_thread(
            coordinator.answer_question,
            payload.question,
            payload.analysis,
        )
    except Exception as exc:
        logger.exception("question failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "question_failed",
                "message": "質問への回答生成に失敗しました。",
                "errorType": type(exc).__name__,
            },
        ) from exc


async def _analyze_url(url: str) -> AnalysisResult:
    _validate_url_syntax(url)
    source_kind = SourceInspectorAgent.classify_url(url)

    if source_kind in {"spotify", "apple_music"}:
        service_name = "Spotify" if source_kind == "spotify" else "Apple Music"
        raise HTTPException(
            status_code=422,
            detail={
                "code": "source_requires_audio_upload",
                "service": service_name,
                "message": (
                    f"{service_name}の公式URLから解析用音声を取得することはできません。"
                    "同じ楽曲のYouTube URL、または権利を持つ音声ファイルを指定してください。"
                ),
            },
        )

    coordinator = get_coordinator()
    if source_kind == "youtube":
        return await asyncio.to_thread(coordinator.analyze_youtube, url=url)

    with tempfile.TemporaryDirectory(prefix="chord-url-") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / "source_audio"
        content_type = await _download_public_audio(url, source_path)
        suffix = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or ".audio"
        source_with_suffix = source_path.with_suffix(suffix)
        source_path.replace(source_with_suffix)
        dsp_path, model_path, _ = _prepare_audio(source_with_suffix, temp_path)
        return await asyncio.to_thread(
            coordinator.analyze_upload,
            dsp_path=dsp_path,
            model_audio_path=model_path,
            source_label=url,
            source_type="direct_audio_url",
        )


async def _analyze_upload(upload: UploadFile) -> AnalysisResult:
    filename = Path(upload.filename or "uploaded-audio").name
    suffix = Path(filename).suffix.lower()
    content_type = (upload.content_type or "").lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES and not content_type.startswith("audio/"):
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_media_type",
                "message": "MP3、WAV、FLAC、AAC、M4A、AIFF、OGG、OPUS、WebMを指定してください。",
            },
        )

    with tempfile.TemporaryDirectory(prefix="chord-upload-") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / f"source{suffix or '.audio'}"
        await _save_upload(upload, source_path)
        dsp_path, model_path, _ = _prepare_audio(source_path, temp_path)
        coordinator = get_coordinator()
        return await asyncio.to_thread(
            coordinator.analyze_upload,
            dsp_path=dsp_path,
            model_audio_path=model_path,
            source_label=filename,
            source_type="upload",
        )


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    total = 0
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "file_too_large",
                        "message": "音声ファイルは30MB以下にしてください。",
                    },
                )
            handle.write(chunk)
    await upload.close()
    if total == 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_file", "message": "音声ファイルが空です。"},
        )


def _prepare_audio(source_path: Path, temp_path: Path) -> tuple[Path, Path, float]:
    metadata = _probe_audio(source_path)
    duration = metadata["duration"]
    if duration <= 0:
        raise ValueError("音声の再生時間を取得できませんでした。")
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "audio_too_long",
                "message": "現在のバージョンでは12分以内の音源を指定してください。",
            },
        )

    dsp_path = temp_path / "analysis.wav"
    model_path = temp_path / "model.mp3"

    _run_media_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            "-c:a",
            "pcm_s16le",
            str(dsp_path),
        ],
        timeout=180,
    )
    _run_media_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-b:a",
            "128k",
            str(model_path),
        ],
        timeout=180,
    )

    if model_path.stat().st_size > MAX_MODEL_AUDIO_BYTES:
        _run_media_command(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-map_metadata",
                "-1",
                "-vn",
                "-ac",
                "2",
                "-ar",
                "32000",
                "-b:a",
                "96k",
                str(model_path),
            ],
            timeout=180,
        )
    if model_path.stat().st_size > MAX_MODEL_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "model_payload_too_large",
                "message": "Geminiへ安全に送信できるサイズへ圧縮できませんでした。より短い音源を指定してください。",
            },
        )

    return dsp_path, model_path, duration


def _probe_audio(path: Path) -> dict[str, float]:
    result = _run_media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    payload = json.loads(result.stdout)
    return {"duration": float(payload.get("format", {}).get("duration", 0.0))}


def _run_media_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _validate_url_syntax(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_url", "message": "有効なHTTPまたはHTTPS URLを指定してください。"},
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_url", "message": "認証情報を含むURLは指定できません。"},
        )


async def _validate_public_destination(url: str) -> None:
    _validate_url_syntax(url)
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_url", "message": "URLのポート番号が不正です。"},
        ) from exc
    if port not in {None, 80, 443}:
        raise HTTPException(
            status_code=422,
            detail={"code": "unsafe_url", "message": "80番または443番以外のURLは指定できません。"},
        )

    resolved_port = port or (443 if parsed.scheme == "https" else 80)

    def resolve() -> list[str]:
        results = socket.getaddrinfo(parsed.hostname, resolved_port, type=socket.SOCK_STREAM)
        return list({item[4][0] for item in results})

    try:
        addresses = await asyncio.wait_for(asyncio.to_thread(resolve), timeout=5.0)
    except (socket.gaierror, asyncio.TimeoutError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "unresolvable_url", "message": "URLのホスト名を解決できません。"},
        ) from exc
    if not addresses:
        raise HTTPException(
            status_code=422,
            detail={"code": "unresolvable_url", "message": "URLのホスト名を解決できません。"},
        )
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise HTTPException(
                status_code=422,
                detail={"code": "unsafe_url", "message": "プライベートネットワーク宛てのURLは指定できません。"},
            )


async def _download_public_audio(url: str, destination: Path) -> str:
    current_url = url
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)
    headers = {"User-Agent": "MusicChordAnalyzer/1.0"}

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            follow_redirects=False,
        ) as client:
            for redirect_count in range(DOWNLOAD_REDIRECT_LIMIT + 1):
                await _validate_public_destination(current_url)
                async with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirect_count >= DOWNLOAD_REDIRECT_LIMIT:
                            raise HTTPException(
                                status_code=422,
                                detail={
                                    "code": "too_many_redirects",
                                    "message": "音声URLのリダイレクトが多すぎます。",
                                },
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "code": "audio_download_failed",
                                "message": (
                                    "音声URLの取得に失敗しました"
                                    f"（HTTP {response.status_code}）。"
                                ),
                            },
                        ) from exc

                    content_type = response.headers.get(
                        "content-type",
                        "application/octet-stream",
                    )
                    suffix = Path(urlparse(current_url).path).suffix.lower()
                    if (
                        not content_type.lower().startswith("audio/")
                        and suffix not in ALLOWED_AUDIO_SUFFIXES
                    ):
                        raise HTTPException(
                            status_code=415,
                            detail={
                                "code": "url_is_not_audio",
                                "message": "URLが直接参照可能な音声ファイルではありません。",
                            },
                        )

                    declared_size = response.headers.get("content-length")
                    if declared_size:
                        try:
                            declared_size_value = int(declared_size)
                        except ValueError:
                            declared_size_value = 0
                        if declared_size_value > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail={
                                    "code": "file_too_large",
                                    "message": "音声URLは30MB以下にしてください。",
                                },
                            )

                    total = 0
                    with destination.open("wb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > MAX_UPLOAD_BYTES:
                                raise HTTPException(
                                    status_code=413,
                                    detail={
                                        "code": "file_too_large",
                                        "message": "音声URLは30MB以下にしてください。",
                                    },
                                )
                            handle.write(chunk)
                    if total == 0:
                        raise HTTPException(
                            status_code=422,
                            detail={
                                "code": "empty_audio",
                                "message": "URLから取得した音声が空です。",
                            },
                        )
                    return content_type
    except HTTPException:
        raise
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "audio_download_failed",
                "message": "音声URLへ接続できませんでした。",
            },
        ) from exc

    raise HTTPException(
        status_code=422,
        detail={
            "code": "audio_download_failed",
            "message": "音声URLを取得できませんでした。",
        },
    )
