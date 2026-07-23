from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from .dsp import analyze_audio_file
from .pipeline_v2 import AccuracyPipelineV2
from .prompts import (
    ANALYSIS_QA_SYSTEM_PROMPT,
    AUDIT_SYSTEM_PROMPT,
    HARMONY_SYSTEM_PROMPT,
    QUESTION_ROUTER_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    build_audit_prompt,
    build_harmony_prompt,
    build_structure_prompt,
)
from .schemas import (
    AIAnalysisDraft,
    AnalysisResult,
    ChordEvent,
    DspSummary,
    HarmonyDraft,
    Measure,
    QuestionResponse,
    QuestionRoute,
    SectionResult,
    SourceLink,
    StructureDraft,
    TrackResult,
)


SchemaT = TypeVar("SchemaT", bound=BaseModel)
YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class SourceInspectorAgent:
    @staticmethod
    def classify_url(url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host in YOUTUBE_HOSTS or host.endswith(".youtube.com"):
            return "youtube"
        if host == "open.spotify.com" or host.endswith(".spotify.com"):
            return "spotify"
        if host == "music.apple.com" or host.endswith(".music.apple.com"):
            return "apple_music"
        return "direct_audio_url"


class BaseMusicAgent:
    def __init__(self, client: genai.Client, model: str) -> None:
        self.client = client
        self.model = model

    def _generate_typed(
        self,
        *,
        contents: list[Any],
        schema: type[SchemaT],
        system_instruction: str,
        temperature: float = 0.0,
        max_output_tokens: int = 16_384,
    ) -> SchemaT:
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, schema):
            return parsed
        if parsed is not None:
            return schema.model_validate(parsed)
        if not response.text:
            raise RuntimeError("Geminiから空の応答が返されました。")
        return schema.model_validate_json(response.text)


class StructureAnalysisAgent(BaseMusicAgent):
    def analyze(
        self,
        *,
        media_part: types.Part,
        source_label: str,
        known_duration: float | None,
        dsp: DspSummary | None,
    ) -> StructureDraft:
        return self._generate_typed(
            contents=[media_part, build_structure_prompt(source_label, known_duration, dsp)],
            schema=StructureDraft,
            system_instruction=STRUCTURE_SYSTEM_PROMPT,
            temperature=0.0,
        )


class HarmonyAnalysisAgent(BaseMusicAgent):
    def analyze(
        self,
        *,
        media_part: types.Part,
        structure: StructureDraft,
        dsp: DspSummary | None,
    ) -> HarmonyDraft:
        return self._generate_typed(
            contents=[media_part, build_harmony_prompt(structure, dsp)],
            schema=HarmonyDraft,
            system_instruction=HARMONY_SYSTEM_PROMPT,
            temperature=0.0,
        )


class TheoryAuditAgent(BaseMusicAgent):
    def audit(
        self,
        *,
        structure: StructureDraft,
        harmony: HarmonyDraft,
        dsp: DspSummary | None,
        source_type: str,
    ) -> AIAnalysisDraft:
        return self._generate_typed(
            contents=[build_audit_prompt(structure, harmony, dsp, source_type)],
            schema=AIAnalysisDraft,
            system_instruction=AUDIT_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=24_576,
        )


class QuestionRouterAgent(BaseMusicAgent):
    def route(self, question: str, analysis: AnalysisResult) -> QuestionRoute:
        compact = {
            "question": question,
            "track": analysis.track.model_dump(),
            "sectionNames": [section.name for section in analysis.sections],
        }
        return self._generate_typed(
            contents=[json.dumps(compact, ensure_ascii=False)],
            schema=QuestionRoute,
            system_instruction=QUESTION_ROUTER_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=512,
        )


class AnalysisQuestionAgent(BaseMusicAgent):
    def answer(self, question: str, analysis: AnalysisResult) -> str:
        payload = {"analysis": analysis.model_dump(), "question": question}
        response = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            config=types.GenerateContentConfig(
                system_instruction=ANALYSIS_QA_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=2_048,
            ),
        )
        if not response.text:
            raise RuntimeError("質問への回答が空でした。")
        return response.text.strip()


class MusicResearchAgent(BaseMusicAgent):
    def answer(self, question: str, analysis: AnalysisResult) -> QuestionResponse:
        research_context = {
            "track": analysis.track.model_dump(),
            "sections": [
                {
                    "name": section.name,
                    "key": section.key,
                    "startSeconds": section.startSeconds,
                    "endSeconds": section.endSeconds,
                    "chords": [
                        chord.symbol
                        for measure in section.measures
                        for chord in measure.chords
                    ],
                }
                for section in analysis.sections
            ],
            "question": question,
        }
        response = self.client.models.generate_content(
            model=self.model,
            contents=json.dumps(research_context, ensure_ascii=False, separators=(",", ":")),
            config=types.GenerateContentConfig(
                system_instruction=RESEARCH_SYSTEM_PROMPT,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0,
                max_output_tokens=3_072,
            ),
        )
        if not response.text:
            raise RuntimeError("Web調査の回答が空でした。")
        sources: list[SourceLink] = []
        search_queries: list[str] = []
        search_entry_point_html = ""
        candidate = response.candidates[0] if response.candidates else None
        grounding = getattr(candidate, "grounding_metadata", None) if candidate else None
        if grounding:
            for chunk in getattr(grounding, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                url = getattr(web, "uri", None) if web else None
                title = getattr(web, "title", None) if web else None
                if url and not any(item.url == url for item in sources):
                    sources.append(SourceLink(title=title or url, url=url))
            search_queries = list(getattr(grounding, "web_search_queries", None) or [])
            search_entry = getattr(grounding, "search_entry_point", None)
            search_entry_point_html = (
                getattr(search_entry, "rendered_content", "") if search_entry else ""
            ) or ""
        return QuestionResponse(
            answer=response.text.strip(),
            usedWebSearch=True,
            routeReason="質問に外部楽曲または最新情報の調査が必要なためです。",
            sources=sources,
            searchQueries=search_queries,
            searchEntryPointHtml=search_entry_point_html,
        )


class MusicCoordinatorAgent:
    def __init__(self) -> None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        resolver_model = os.environ.get("GEMINI_RESOLVER_MODEL", model)
        if not project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT が設定されていません。")
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self.model = model
        self.pipeline_mode = os.environ.get("ANALYSIS_PIPELINE", "v2").lower()
        self.accuracy_v2 = AccuracyPipelineV2(
            client=self.client,
            model=model,
            resolver_model=resolver_model,
            max_parallel_calls=int(os.environ.get("MODEL_MAX_PARALLEL_CALLS", "4")),
        )
        self.structure_agent = StructureAnalysisAgent(self.client, model)
        self.harmony_agent = HarmonyAnalysisAgent(self.client, model)
        self.audit_agent = TheoryAuditAgent(self.client, model)
        self.question_router = QuestionRouterAgent(self.client, model)
        self.analysis_qa_agent = AnalysisQuestionAgent(self.client, model)
        self.research_agent = MusicResearchAgent(self.client, model)

    @staticmethod
    def _media_from_bytes(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> types.Part:
        return types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

    @staticmethod
    def _media_from_youtube(url: str) -> types.Part:
        return types.Part.from_uri(file_uri=url, mime_type="video/mp4")

    def analyze_upload(
        self,
        *,
        dsp_path: str | Path,
        model_audio_path: str | Path,
        source_label: str,
        source_type: str = "upload",
    ) -> AnalysisResult:
        if self.pipeline_mode != "legacy":
            return self.accuracy_v2.analyze_upload(
                dsp_path=str(dsp_path),
                model_audio_path=str(model_audio_path),
                source_label=source_label,
                source_type=source_type,
            )
        dsp = analyze_audio_file(dsp_path)
        media = self._media_from_bytes(Path(model_audio_path).read_bytes())
        return self._run_legacy(
            media=media,
            source_label=source_label,
            source_type=source_type,
            known_duration=dsp.durationSeconds,
            dsp=dsp,
        )

    def analyze_youtube(self, *, url: str) -> AnalysisResult:
        if self.pipeline_mode != "legacy":
            return self.accuracy_v2.analyze_youtube(url=url)
        return self._run_legacy(
            media=self._media_from_youtube(url),
            source_label=url,
            source_type="youtube",
            known_duration=None,
            dsp=None,
        )

    def _run_legacy(
        self,
        *,
        media: types.Part,
        source_label: str,
        source_type: str,
        known_duration: float | None,
        dsp: DspSummary | None,
    ) -> AnalysisResult:
        structure = self.structure_agent.analyze(
            media_part=media,
            source_label=source_label,
            known_duration=known_duration,
            dsp=dsp,
        )
        harmony = self.harmony_agent.analyze(
            media_part=media,
            structure=structure,
            dsp=dsp,
        )
        try:
            draft = self.audit_agent.audit(
                structure=structure,
                harmony=harmony,
                dsp=dsp,
                source_type=source_type,
            )
        except Exception:
            draft = self._legacy_fallback(structure, harmony, dsp)
        duration = dsp.durationSeconds if dsp else draft.track.durationSeconds
        duration = max(0.1, float(duration))
        sections = self._legacy_normalise_sections(draft, harmony, duration, dsp)
        return AnalysisResult(
            track=draft.track.model_copy(update={"durationSeconds": duration}),
            sections=sections,
            musicalSummary=draft.musicalSummary,
            warnings=draft.warnings,
            analysisVersion="1.0",
            sourceType=source_type,
            sourceLabel=source_label,
            analysisMethod="hybrid" if dsp else "ai_only",
            waveform=dsp.waveform if dsp else [],
            limitations=["legacy pipeline"],
        )

    def _legacy_fallback(
        self,
        structure: StructureDraft,
        harmony: HarmonyDraft,
        dsp: DspSummary | None,
    ) -> AIAnalysisDraft:
        harmony_by_id = {section.sectionId: section for section in harmony.sections}
        bpm = harmony.bpm or structure.track.bpm or (dsp.bpm if dsp else None)
        signature = harmony.timeSignature or structure.track.timeSignature or "4/4"
        global_key = harmony.globalKey or structure.track.globalKey
        sections: list[SectionResult] = []
        for section in structure.sections:
            h_section = harmony_by_id.get(section.id)
            chords = []
            if h_section:
                chords = [
                    ChordEvent(
                        symbol=item.symbol,
                        startSeconds=max(section.startSeconds, item.startSeconds),
                        endSeconds=min(section.endSeconds, item.endSeconds),
                        confidence=item.confidence,
                        source="hybrid" if dsp else "ai",
                        alternatives=item.alternatives,
                    )
                    for item in h_section.chords
                    if min(section.endSeconds, item.endSeconds)
                    > max(section.startSeconds, item.startSeconds)
                ]
            measures = self._legacy_build_measures(
                chords, section.startSeconds, section.endSeconds, bpm, signature
            )
            sections.append(
                SectionResult(
                    id=section.id,
                    name=section.name,
                    type=section.type,
                    startSeconds=section.startSeconds,
                    endSeconds=section.endSeconds,
                    key=(h_section.key if h_section else None) or section.key or global_key,
                    mode=(h_section.mode if h_section else None) or section.mode,
                    confidence=section.confidence,
                    summary=section.notes,
                    measures=measures,
                )
            )
        return AIAnalysisDraft(
            track=TrackResult(
                title=structure.track.title,
                artist=structure.track.artist,
                durationSeconds=structure.track.durationSeconds,
                bpm=bpm,
                timeSignature=signature,
                globalKey=global_key,
                globalMode=harmony.globalMode or structure.track.globalMode,
                confidence=structure.track.confidence,
            ),
            sections=sections,
            musicalSummary="従来パイプラインの統合結果です。",
            warnings=["理論監査に失敗したため保守的統合を使用しました。"],
        )

    def _legacy_normalise_sections(
        self,
        draft: AIAnalysisDraft,
        harmony: HarmonyDraft,
        duration: float,
        dsp: DspSummary | None,
    ) -> list[SectionResult]:
        bpm = draft.track.bpm or harmony.bpm or (dsp.bpm if dsp else None)
        signature = draft.track.timeSignature or harmony.timeSignature or "4/4"
        output: list[SectionResult] = []
        for index, section in enumerate(sorted(draft.sections, key=lambda item: item.startSeconds)):
            start = max(0.0, min(duration, section.startSeconds))
            end = min(duration, max(start + 0.05, section.endSeconds))
            chords = [chord for measure in section.measures for chord in measure.chords]
            measures = self._legacy_build_measures(chords, start, end, bpm, signature)
            output.append(
                section.model_copy(
                    update={
                        "id": section.id or f"section-{index + 1}",
                        "startSeconds": start,
                        "endSeconds": end,
                        "measures": measures,
                    }
                )
            )
        return output

    @staticmethod
    def _legacy_build_measures(
        chords: list[ChordEvent],
        section_start: float,
        section_end: float,
        bpm: float | None,
        time_signature: str | None,
    ) -> list[Measure]:
        if not chords:
            return []
        match = re.match(r"\s*(\d+)\s*/", time_signature or "")
        beats = max(1, min(12, int(match.group(1)))) if match else 4
        safe_bpm = float(bpm) if bpm and math.isfinite(float(bpm)) and bpm > 0 else 120.0
        beat_duration = 60.0 / safe_bpm
        bar_duration = beat_duration * beats
        grouped: dict[int, list[ChordEvent]] = defaultdict(list)
        for chord in chords:
            start = max(section_start, min(section_end, chord.startSeconds))
            end = min(section_end, max(start + 0.03, chord.endSeconds))
            if end <= start:
                continue
            bar = max(1, int(math.floor(start / bar_duration)) + 1)
            bar_start = (bar - 1) * bar_duration
            beat = max(1.0, min(float(beats), ((start - bar_start) / beat_duration) + 1.0))
            grouped[bar].append(
                chord.model_copy(
                    update={"startSeconds": start, "endSeconds": end, "bar": bar, "beat": beat}
                )
            )
        return [
            Measure(
                bar=bar,
                startSeconds=max(section_start, (bar - 1) * bar_duration),
                endSeconds=min(section_end, bar * bar_duration),
                chords=sorted(grouped[bar], key=lambda item: item.startSeconds),
            )
            for bar in sorted(grouped)
        ]

    def answer_question(self, question: str, analysis: AnalysisResult) -> QuestionResponse:
        try:
            route = self.question_router.route(question, analysis)
        except Exception:
            external_pattern = re.compile(
                r"(同じ|似た|類似|有名曲|他の曲|最近|発売|アーティスト|原曲|検索|調べ)",
                re.IGNORECASE,
            )
            route = QuestionRoute(
                route="web_research" if external_pattern.search(question) else "analysis_only",
                reason="ルーター障害時の保守的なキーワード判定です。",
            )
        if route.route == "web_research":
            try:
                response = self.research_agent.answer(question, analysis)
                response.routeReason = route.reason
                return response
            except Exception as exc:
                answer = self.analysis_qa_agent.answer(question, analysis)
                return QuestionResponse(
                    answer=answer + "\n\n外部検索を利用できなかったため分析済みデータだけで回答しています。",
                    usedWebSearch=False,
                    routeReason=f"{route.reason} / Web検索失敗: {type(exc).__name__}",
                )
        return QuestionResponse(
            answer=self.analysis_qa_agent.answer(question, analysis),
            usedWebSearch=False,
            routeReason=route.reason,
        )
