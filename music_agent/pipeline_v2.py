from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from google import genai

from .beat_grid import BeatGrid, build_beat_grid
from .consensus import ConsensusOutput, consensus_section, reconcile_repeated_sections
from .diagnostics import DiagnosticsRecorder
from .dsp import analyze_audio_file
from .media_source import AnalysisMediaSource, LocalAudioMediaSource, YouTubeMediaSource
from .model_gateway import ModelGateway
from .prompts import (
    FINAL_EXPLANATION_SYSTEM_PROMPT,
    RESOLUTION_SYSTEM_PROMPT,
    RHYTHM_SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPTS,
    STRUCTURE_SYSTEM_PROMPT,
    build_final_explanation_prompt,
    build_resolution_prompt,
    build_rhythm_prompt,
    build_specialist_prompt,
    build_structure_prompt,
)
from .reference_research import ReferenceResearchResult, research_references
from .schemas import (
    AnalysisResult,
    DspSummary,
    FinalExplanationDraft,
    RhythmDraft,
    SectionResult,
    SectionStructureDraft,
    SpecialistSectionDraft,
    StructureDraft,
    TempoCandidate,
    TrackResult,
)
from .validators import (
    apply_resolution,
    build_section_result,
    normalise_sections,
    validate_invariants,
)
from .youtube_metadata import YouTubeMetadata, resolve_youtube_metadata


SPECIALIST_ROLES = ("root_quality", "bass_extension", "rhythm_pattern")


class AccuracyPipelineV2:
    def __init__(
        self,
        *,
        client: genai.Client,
        model: str,
        resolver_model: str | None = None,
        max_parallel_calls: int = 4,
    ) -> None:
        self.client = client
        self.model = model
        self.resolver_model = resolver_model or model
        self.max_parallel_calls = max(1, min(8, max_parallel_calls))
        self.youtube_api_key = os.environ.get("YOUTUBE_API_KEY")
        self.enable_reference_research = os.environ.get("ENABLE_REFERENCE_RESEARCH", "1") == "1"
        self.max_resolver_calls = max(0, int(os.environ.get("MAX_RESOLVER_CALLS", "4")))

    def analyze_youtube(self, *, url: str) -> AnalysisResult:
        metadata = resolve_youtube_metadata(url, api_key=self.youtube_api_key)
        with YouTubeMediaSource(metadata.canonical_url) as media:
            return self._run(
                media=media,
                source_label=url,
                source_type="youtube",
                known_duration=metadata.duration_seconds,
                dsp=None,
                youtube_metadata=metadata,
            )

    def analyze_upload(
        self,
        *,
        dsp_path: str,
        model_audio_path: str,
        source_label: str,
        source_type: str = "upload",
    ) -> AnalysisResult:
        dsp = analyze_audio_file(dsp_path)
        with LocalAudioMediaSource(model_audio_path) as media:
            return self._run(
                media=media,
                source_label=source_label,
                source_type=source_type,
                known_duration=dsp.durationSeconds,
                dsp=dsp,
                youtube_metadata=None,
            )

    @staticmethod
    def _metadata_dict(metadata: YouTubeMetadata | None) -> dict[str, Any]:
        return metadata.to_prompt_dict() if metadata else {}

    @staticmethod
    def _fallback_structure(
        *,
        source_label: str,
        duration: float,
        metadata: YouTubeMetadata | None,
        dsp: DspSummary | None,
    ) -> StructureDraft:
        from .schemas import SectionStructureDraft, TrackStructureDraft

        title = metadata.title if metadata and metadata.title else source_label
        artist = metadata.channel_title if metadata and metadata.channel_title else "不明"
        key = dsp.keyCandidates[0].key if dsp and dsp.keyCandidates else None
        return StructureDraft(
            track=TrackStructureDraft(
                title=title,
                artist=artist,
                durationSeconds=max(0.001, duration),
                bpm=dsp.bpm if dsp else None,
                globalKey=key,
                confidence=0.2,
            ),
            sections=[
                SectionStructureDraft(
                    id="full-track",
                    name="全体",
                    type="other",
                    startSeconds=0.0,
                    endSeconds=max(0.001, duration),
                    key=key,
                    confidence=0.2,
                    notes="構造分析が失敗したため全体区間へフォールバックしました。",
                )
            ],
            observations=["structure_fallback"],
        )

    @staticmethod
    def _fallback_rhythm(duration: float, dsp: DspSummary | None) -> RhythmDraft:
        bpm = dsp.bpm if dsp and dsp.bpm else 120.0
        key = dsp.keyCandidates[0].key if dsp and dsp.keyCandidates else None
        return RhythmDraft(
            durationSeconds=max(0.001, duration),
            bpmCandidates=[TempoCandidate(bpm=bpm, confidence=0.25, interpretation="fallback")],
            selectedBpm=bpm,
            timeSignature="4/4",
            downbeatOffsetSeconds=(dsp.beatTimes[0] if dsp and dsp.beatTimes else 0.0),
            globalKey=key,
            confidence=0.2,
            observations=["rhythm_fallback"],
        )

    @staticmethod
    def _duration(
        known_duration: float | None,
        structure: StructureDraft | None,
        rhythm: RhythmDraft | None,
    ) -> float:
        values = [
            float(value)
            for value in (
                known_duration,
                structure.track.durationSeconds if structure else None,
                rhythm.durationSeconds if rhythm else None,
            )
            if value is not None and float(value) > 0
        ]
        return max(0.001, values[0] if known_duration else max(values, default=0.001))

    def _run(
        self,
        *,
        media: AnalysisMediaSource,
        source_label: str,
        source_type: str,
        known_duration: float | None,
        dsp: DspSummary | None,
        youtube_metadata: YouTubeMetadata | None,
    ) -> AnalysisResult:
        diagnostics = DiagnosticsRecorder()
        gateway = ModelGateway(
            client=self.client,
            model=self.model,
            diagnostics=diagnostics,
            max_parallel_calls=self.max_parallel_calls,
        )
        resolver_gateway = (
            gateway
            if self.resolver_model == self.model
            else ModelGateway(
                client=self.client,
                model=self.resolver_model,
                diagnostics=diagnostics,
                max_parallel_calls=self.max_parallel_calls,
            )
        )
        metadata_dict = self._metadata_dict(youtube_metadata)
        duration_hint = known_duration or (dsp.durationSeconds if dsp else None) or 360.0
        global_part = media.part()
        structure: StructureDraft | None = None
        rhythm: RhythmDraft | None = None
        references = ReferenceResearchResult()
        global_warnings: list[str] = []

        with diagnostics.stage("global-analysis"):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures: dict[str, Future[Any]] = {
                    "structure": executor.submit(
                        gateway.generate_typed,
                        contents=[
                            global_part,
                            build_structure_prompt(
                                source_label,
                                known_duration,
                                dsp,
                                metadata_dict,
                            ),
                        ],
                        schema=StructureDraft,
                        system_instruction=STRUCTURE_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=12_288,
                    ),
                    "rhythm": executor.submit(
                        gateway.generate_typed,
                        contents=[
                            global_part,
                            build_rhythm_prompt(
                                source_label,
                                known_duration,
                                dsp,
                                metadata_dict,
                            ),
                        ],
                        schema=RhythmDraft,
                        system_instruction=RHYTHM_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=8_192,
                    ),
                }
                if source_type == "youtube" and self.enable_reference_research:
                    futures["references"] = executor.submit(
                        research_references,
                        gateway,
                        metadata_dict or {"sourceLabel": source_label},
                    )
                for name, future in futures.items():
                    try:
                        value = future.result()
                    except Exception as exc:  # noqa: BLE001 - guarded model boundary
                        global_warnings.append(f"{name} failed: {type(exc).__name__}")
                        continue
                    if name == "structure":
                        structure = value
                    elif name == "rhythm":
                        rhythm = value
                    else:
                        references = value

        duration = self._duration(known_duration, structure, rhythm)
        if structure is None:
            structure = self._fallback_structure(
                source_label=source_label,
                duration=duration,
                metadata=youtube_metadata,
                dsp=dsp,
            )
        if rhythm is None:
            rhythm = self._fallback_rhythm(duration, dsp)

        # Authoritative source metadata always wins over model guesses.
        track_updates: dict[str, Any] = {"durationSeconds": duration}
        if youtube_metadata:
            if youtube_metadata.title:
                track_updates["title"] = youtube_metadata.title
            if youtube_metadata.channel_title:
                track_updates["artist"] = youtube_metadata.channel_title
        structure.track = structure.track.model_copy(update=track_updates)
        rhythm.durationSeconds = duration
        for bpm in references.bpm_candidates:
            if not any(abs(item.bpm - bpm) < 0.1 for item in rhythm.bpmCandidates):
                rhythm.bpmCandidates.append(
                    TempoCandidate(bpm=bpm, confidence=0.35, interpretation="grounded-reference")
                )

        sections = normalise_sections(structure.sections, duration=duration, max_sections=8)
        global_key = rhythm.globalKey or structure.track.globalKey
        global_mode = rhythm.globalMode or structure.track.globalMode
        sections = [
            section.model_copy(
                update={
                    "key": section.key or global_key,
                    "mode": section.mode or global_mode,
                }
            )
            for section in sections
        ]

        with diagnostics.stage("beat-grid"):
            grid = build_beat_grid(
                rhythm=rhythm,
                sections=sections,
                duration=duration,
                dsp=dsp,
            )
            diagnostics.grid_score = grid.score

        section_parts: dict[str, Any] = {}
        section_clip_bounds: dict[str, tuple[float, float]] = {}
        with diagnostics.stage("prepare-section-clips"):
            for section in sections:
                context = min(2.0, max(1.25, grid.beat_duration * 2.0))
                clip_start = max(0.0, section.startSeconds - context)
                clip_end = min(duration, section.endSeconds + context)
                section_parts[section.id] = media.part(clip_start, clip_end)
                section_clip_bounds[section.id] = (clip_start, clip_end)

        specialist_results: dict[str, list[SpecialistSectionDraft]] = {
            section.id: [] for section in sections
        }
        with diagnostics.stage("section-specialists"):
            with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
                future_map: dict[Future[SpecialistSectionDraft], tuple[SectionStructureDraft, str]] = {}
                for section in sections:
                    clip_start, clip_end = section_clip_bounds[section.id]
                    for role in SPECIALIST_ROLES:
                        prompt = build_specialist_prompt(
                            role=role,
                            section=section,
                            rhythm=rhythm,
                            grid=grid,
                            clip_start=clip_start,
                            clip_end=clip_end,
                            dsp=dsp,
                        )
                        future = executor.submit(
                            gateway.generate_typed,
                            contents=[section_parts[section.id], prompt],
                            schema=SpecialistSectionDraft,
                            system_instruction=SPECIALIST_SYSTEM_PROMPTS[role],
                            temperature=0.0,
                            max_output_tokens=8_192,
                        )
                        future_map[future] = (section, role)
                for future in as_completed(future_map):
                    section, role = future_map[future]
                    diagnostics.section_specialist_calls += 1
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        global_warnings.append(
                            f"specialist {section.id}/{role} failed: {type(exc).__name__}"
                        )
                        result = SpecialistSectionDraft(
                            sectionId=section.id,
                            role=role,
                            key=section.key,
                            mode=section.mode,
                            observations=["specialist_failed"],
                        )
                    specialist_results[section.id].append(result)

        consensus_pairs: list[tuple[SectionStructureDraft, ConsensusOutput]] = []
        with diagnostics.stage("consensus"):
            for section in sections:
                dsp_runs = None
                if dsp:
                    dsp_runs = [
                        item
                        for item in dsp.chordRuns
                        if item.endSeconds > section.startSeconds
                        and item.startSeconds < section.endSeconds
                    ]
                output = consensus_section(
                    section=section,
                    specialists=specialist_results[section.id],
                    grid=grid,
                    dsp_runs=dsp_runs,
                )
                consensus_pairs.append((section, output))
            reconcile_repeated_sections(consensus_pairs)

        all_uncertain = [
            item
            for _, output in consensus_pairs
            for item in output.uncertain_ranges
        ]
        resolver_targets = sorted(
            all_uncertain,
            key=lambda item: (-(item.endSeconds - item.startSeconds), item.startSeconds),
        )[: self.max_resolver_calls]
        with diagnostics.stage("targeted-resolution"):
            for target in resolver_targets:
                pair = next((pair for pair in consensus_pairs if pair[0].id == target.sectionId), None)
                if pair is None:
                    continue
                section, output = pair
                previous_symbol = None
                next_symbol = None
                for chord in output.chords:
                    if chord.endSeconds <= target.startSeconds + 1e-3:
                        previous_symbol = chord.symbol
                    elif chord.startSeconds >= target.endSeconds - 1e-3:
                        next_symbol = chord.symbol
                        break
                clip_start = max(0.0, target.startSeconds - grid.beat_duration * grid.beats_per_bar)
                clip_end = min(duration, target.endSeconds + grid.beat_duration * grid.beats_per_bar)
                try:
                    resolution = resolver_gateway.generate_typed(
                        contents=[
                            media.part(clip_start, clip_end),
                            build_resolution_prompt(
                                section=section,
                                start=target.startSeconds,
                                end=target.endSeconds,
                                candidates=target.candidates,
                                previous_symbol=previous_symbol,
                                next_symbol=next_symbol,
                                key=section.key,
                                grid=grid,
                            ),
                        ],
                        schema=__import__(
                            "music_agent.schemas", fromlist=["ResolutionDraft"]
                        ).ResolutionDraft,
                        system_instruction=RESOLUTION_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=2_048,
                    )
                    diagnostics.resolver_calls += 1
                    apply_resolution(output.chords, resolution)
                    if resolution.choices:
                        target.resolved = True
                except Exception as exc:  # noqa: BLE001
                    global_warnings.append(
                        f"resolver {target.sectionId} failed: {type(exc).__name__}"
                    )

        preliminary_sections: list[SectionResult] = [
            build_section_result(
                section=section,
                chords=output.chords,
                grid=grid,
                agreement=output.agreement,
            )
            for section, output in consensus_pairs
        ]

        explanation = FinalExplanationDraft()
        with diagnostics.stage("final-explanation"):
            compact = {
                "track": {
                    "title": structure.track.title,
                    "artist": structure.track.artist,
                    "durationSeconds": duration,
                    "bpm": grid.bpm,
                    "timeSignature": grid.time_signature,
                    "globalKey": global_key,
                    "globalMode": global_mode,
                },
                "sections": [
                    {
                        "id": section.id,
                        "name": section.name,
                        "type": section.type,
                        "key": section.key,
                        "startSeconds": section.startSeconds,
                        "endSeconds": section.endSeconds,
                        "agreement": section.agreement,
                        "chords": [
                            chord.symbol
                            for measure in section.measures
                            for chord in measure.chords
                        ],
                    }
                    for section in preliminary_sections
                ],
                "unresolvedRanges": [
                    item.model_dump() for item in all_uncertain if not item.resolved
                ],
                "referenceSummary": references.summary[:3000],
            }
            try:
                explanation = gateway.generate_typed(
                    contents=[build_final_explanation_prompt(compact)],
                    schema=FinalExplanationDraft,
                    system_instruction=FINAL_EXPLANATION_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=3_072,
                )
            except Exception as exc:  # noqa: BLE001
                global_warnings.append(f"final explanation failed: {type(exc).__name__}")

        for section in preliminary_sections:
            section.summary = explanation.sectionSummaries.get(section.id, section.summary)

        track_confidences = [structure.track.confidence, rhythm.confidence]
        track_confidences.extend(
            output.agreement for _, output in consensus_pairs if output.agreement > 0
        )
        result = AnalysisResult(
            track=TrackResult(
                title=structure.track.title,
                artist=structure.track.artist,
                durationSeconds=round(duration, 3),
                bpm=round(grid.bpm, 2),
                timeSignature=grid.time_signature,
                globalKey=global_key,
                globalMode=global_mode,
                confidence=round(mean(track_confidences), 4),
            ),
            sections=preliminary_sections,
            musicalSummary=explanation.musicalSummary
            or "複数の専門分析と決定的な合議処理による解析結果です。",
            warnings=list(
                dict.fromkeys(
                    global_warnings
                    + explanation.warnings
                    + [
                        f"{item.sectionId} {item.startSeconds:.2f}〜{item.endSeconds:.2f}秒は低信頼です。"
                        for item in all_uncertain
                        if not item.resolved
                    ]
                )
            ),
            analysisVersion="2.0",
            sourceType=source_type,
            sourceLabel=source_label,
            analysisMethod="hybrid" if dsp else "ai_only",
            waveform=dsp.waveform if dsp else [],
            limitations=[
                "自動採譜のため、密集したボイシング、意図的な曖昧和音、ライブ版では誤差が残る場合があります。",
                "低一致度の範囲は uncertainRanges と warnings に明示しています。",
            ],
            tempoSegments=list(grid.tempo_segments),
            downbeatOffsetSeconds=grid.downbeat_offset,
            referenceSources=references.sources,
            uncertainRanges=all_uncertain,
        )
        invariant_errors = validate_invariants(result)
        result.diagnostics = diagnostics.build(invariant_errors)
        if invariant_errors:
            raise RuntimeError(
                "Accuracy v2 invariant violation: " + ", ".join(invariant_errors)
            )
        return result
