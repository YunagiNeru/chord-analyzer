from __future__ import annotations

import math
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from statistics import mean
from typing import Any

from google import genai

from .beat_grid import BeatGrid, build_beat_grid
from .chord_symbol import canonicalize_symbol
from .consensus import ConsensusOutput, consensus_section
from .diagnostics import DiagnosticsRecorder
from .dsp import analyze_audio_file
from .media_source import AnalysisMediaSource, LocalAudioMediaSource, YouTubeMediaSource
from .model_gateway import ModelGateway
from .motif_reconcile import reconcile_repeated_sections_enhanced
from .prompts import (
    FINAL_EXPLANATION_SYSTEM_PROMPT,
    RESOLUTION_SYSTEM_PROMPT,
    RHYTHM_SYSTEM_PROMPT,
    SPECIALIST_SYSTEM_PROMPTS,
    STRUCTURE_REFINEMENT_SYSTEM_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    build_final_explanation_prompt,
    build_resolution_prompt,
    build_rhythm_prompt,
    build_specialist_prompt,
    build_structure_prompt,
    build_structure_refinement_prompt,
)
from .reference_research import ReferenceResearchResult, research_references
from .schemas import (
    AnalysisResult,
    CompactResolutionDraft,
    CompactSpecialistDraft,
    DspSummary,
    FinalExplanationDraft,
    ResolutionChoice,
    ResolutionDraft,
    RhythmDraft,
    SectionResult,
    SectionStructureDraft,
    SpecialistSectionDraft,
    StructureDraft,
    StructureRefinementDraft,
    TempoCandidate,
    TrackResult,
    TrackStructureDraft,
    UncertainRange,
)
from .structure_refine import (
    AnalysisSlice,
    align_sections_to_grid,
    build_analysis_slices,
    oversized_sections,
    refine_sections,
)
from .validators import (
    apply_resolution,
    build_section_result,
    compact_to_specialist,
    normalise_sections,
    normalise_specialist_result,
    validate_invariants,
    validate_quality,
)
from .youtube_metadata import YouTubeMetadata, resolve_youtube_metadata


SPECIALIST_ROLES = ("root_quality", "bass_extension", "rhythm_pattern")


def _finite(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _confidence(value: object, default: float = 0.5) -> float:
    return max(0.0, min(1.0, _finite(value, default)))


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _compact_progression(output: ConsensusOutput, limit: int = 40) -> list[str]:
    progression: list[str] = []
    for chord in sorted(output.chords, key=lambda item: (item.startSeconds, item.endSeconds)):
        symbol = canonicalize_symbol(chord.symbol)
        if not progression or progression[-1] != symbol:
            progression.append(symbol)
        if len(progression) >= limit:
            break
    return progression


def _compact_failure_label(value: str) -> str:
    return value.split(":", 1)[0][:160]


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
        self.max_resolver_calls = max(0, int(os.environ.get("MAX_RESOLVER_CALLS", "12")))

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
        known = _finite(known_duration, 0.0) if known_duration is not None else 0.0
        if known > 0:
            return known
        values: list[float] = []
        if structure:
            values.append(_finite(structure.track.durationSeconds, 0.0))
        if rhythm:
            values.append(_finite(rhythm.durationSeconds, 0.0))
        return max(0.001, max(values, default=0.001))

    @staticmethod
    def _minimum_sections(duration: float) -> int:
        if duration < 45.0:
            return 1
        if duration < 90.0:
            return 2
        return 3

    @classmethod
    def _usable_sections(
        cls,
        structure: StructureDraft | None,
        *,
        duration: float,
    ) -> list[SectionStructureDraft]:
        if structure is None:
            return []
        sections = normalise_sections(structure.sections, duration=duration, max_sections=20)
        if len(sections) < cls._minimum_sections(duration):
            return []
        if duration >= 90.0 and any(
            section.endSeconds - section.startSeconds > duration * 0.82
            for section in sections
        ):
            return []
        return sections

    def _windowed_structure(
        self,
        *,
        media: AnalysisMediaSource,
        gateway: ModelGateway,
        source_label: str,
        duration: float,
        dsp: DspSummary | None,
        metadata: dict[str, Any],
        diagnostics: DiagnosticsRecorder,
    ) -> StructureDraft:
        window_seconds = 55.0
        overlap_seconds = 5.0
        starts: list[float] = []
        current = 0.0
        while current < duration - 0.1:
            starts.append(current)
            next_value = current + window_seconds - overlap_seconds
            if next_value <= current:
                break
            current = next_value

        sections: list[SectionStructureDraft] = []
        track = TrackStructureDraft(durationSeconds=duration)
        with diagnostics.stage("windowed-structure-recovery"):
            with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
                futures: dict[Future[StructureDraft], tuple[int, float, float]] = {}
                for index, start in enumerate(starts):
                    end = min(duration, start + window_seconds)
                    window_metadata = dict(metadata)
                    window_metadata["analysisWindow"] = {
                        "startSeconds": start,
                        "endSeconds": end,
                        "timestampsMayBeClipRelative": True,
                    }
                    future = executor.submit(
                        gateway.generate_typed,
                        contents=[
                            media.part(start, end),
                            build_structure_prompt(
                                f"{source_label} [{start:.3f}-{end:.3f}]",
                                duration,
                                dsp,
                                window_metadata,
                            ),
                        ],
                        schema=StructureDraft,
                        system_instruction=STRUCTURE_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=6_144,
                        diagnostic_label=f"structure-window-{index + 1}",
                    )
                    futures[future] = (index, start, end)

                for future in as_completed(futures):
                    index, start, end = futures[future]
                    try:
                        draft = future.result()
                    except Exception as exc:  # noqa: BLE001
                        diagnostics.record_error(f"structure-window-{index + 1}", exc)
                        continue
                    if draft.track.title and draft.track.title != "不明な楽曲":
                        track = draft.track
                    clip_duration = end - start
                    times = [
                        _finite(value, -1.0)
                        for section in draft.sections
                        for value in (section.startSeconds, section.endSeconds)
                    ]
                    valid_times = [value for value in times if value >= 0.0]
                    relative = bool(valid_times) and max(valid_times) <= clip_duration + 1.0 and start > 0.0
                    offset = start if relative else 0.0
                    for local_index, section in enumerate(draft.sections, start=1):
                        section_start = _finite(section.startSeconds, -1.0) + offset
                        section_end = _finite(section.endSeconds, -1.0) + offset
                        if section_start < 0.0 or section_end <= section_start:
                            continue
                        sections.append(
                            section.model_copy(
                                update={
                                    "id": f"window-{index + 1}-{section.id or local_index}",
                                    "startSeconds": section_start,
                                    "endSeconds": section_end,
                                }
                            )
                        )

        return StructureDraft(
            track=track.model_copy(update={"durationSeconds": duration}),
            sections=sections,
            observations=["windowed_structure_recovery"],
        )

    def _refine_structure(
        self,
        *,
        media: AnalysisMediaSource,
        gateway: ModelGateway,
        sections: list[SectionStructureDraft],
        grid: BeatGrid,
        duration: float,
        diagnostics: DiagnosticsRecorder,
    ) -> list[SectionStructureDraft]:
        aligned, adjustments = align_sections_to_grid(
            sections,
            grid=grid,
            duration=duration,
        )
        diagnostics.boundary_adjustments += adjustments
        targets = oversized_sections(aligned, grid=grid)
        if not targets:
            return aligned

        refinements: dict[str, StructureRefinementDraft] = {}
        bar_duration = grid.beat_duration * grid.beats_per_bar
        with diagnostics.stage("structure-refinement"):
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_calls, len(targets))) as executor:
                future_map: dict[Future[StructureRefinementDraft], SectionStructureDraft] = {}
                for section in targets:
                    clip_start = max(0.0, section.startSeconds - bar_duration)
                    clip_end = min(duration, section.endSeconds + bar_duration)
                    future = executor.submit(
                        gateway.generate_typed,
                        contents=[
                            media.part(clip_start, clip_end),
                            build_structure_refinement_prompt(
                                section=section,
                                grid=grid,
                                clip_start=clip_start,
                                clip_end=clip_end,
                            ),
                        ],
                        schema=StructureRefinementDraft,
                        system_instruction=STRUCTURE_REFINEMENT_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=2_048,
                        retries=2,
                        diagnostic_label=f"structure-refinement-{section.id}",
                    )
                    future_map[future] = section
                    diagnostics.structure_refinement_calls += 1

                for future in as_completed(future_map):
                    section = future_map[future]
                    try:
                        refinements[section.id] = future.result()
                    except Exception as exc:  # noqa: BLE001
                        diagnostics.record_error(f"structure-refinement-{section.id}", exc)

        refined, final_adjustments = refine_sections(
            aligned,
            grid=grid,
            duration=duration,
            model_refinements=refinements,
        )
        diagnostics.boundary_adjustments += final_adjustments
        return refined

    @staticmethod
    def _merge_slice_results(
        sections: list[SectionStructureDraft],
        accumulated: dict[str, dict[str, list[SpecialistSectionDraft]]],
    ) -> dict[str, list[SpecialistSectionDraft]]:
        output: dict[str, list[SpecialistSectionDraft]] = {section.id: [] for section in sections}
        section_by_id = {section.id: section for section in sections}
        for parent_id, role_map in accumulated.items():
            parent = section_by_id[parent_id]
            for role in SPECIALIST_ROLES:
                parts = role_map.get(role, [])
                chords = sorted(
                    [chord for part in parts for chord in part.chords],
                    key=lambda item: (item.startSeconds, item.endSeconds, item.symbol),
                )
                repeated = list(
                    dict.fromkeys(
                        item
                        for part in parts
                        for item in part.repeatedPattern
                        if item
                    )
                )[:16]
                output[parent_id].append(
                    SpecialistSectionDraft(
                        sectionId=parent_id,
                        role=role,
                        key=parent.key,
                        mode=parent.mode,
                        chords=chords,
                        repeatedPattern=repeated,
                    )
                )
        return output

    def _run_specialists(
        self,
        *,
        media: AnalysisMediaSource,
        gateway: ModelGateway,
        sections: list[SectionStructureDraft],
        rhythm: RhythmDraft,
        grid: BeatGrid,
        duration: float,
        dsp: DspSummary | None,
        diagnostics: DiagnosticsRecorder,
        global_warnings: list[str],
    ) -> dict[str, list[SpecialistSectionDraft]]:
        slices = build_analysis_slices(sections, grid=grid, max_bars=16)
        diagnostics.analysis_slice_count = len(slices)
        accumulated: dict[str, dict[str, list[SpecialistSectionDraft]]] = {
            section.id: {role: [] for role in SPECIALIST_ROLES}
            for section in sections
        }

        with diagnostics.stage("section-specialists"):
            with ThreadPoolExecutor(max_workers=self.max_parallel_calls) as executor:
                future_map: dict[
                    Future[CompactSpecialistDraft],
                    tuple[AnalysisSlice, str, str, float, float, Any],
                ] = {}
                for analysis_slice in slices:
                    slice_section = analysis_slice.section
                    context = min(2.0, max(1.0, grid.beat_duration * 2.0))
                    clip_start = max(0.0, slice_section.startSeconds - context)
                    clip_end = min(duration, slice_section.endSeconds + context)
                    part = media.part(clip_start, clip_end)
                    for role in SPECIALIST_ROLES:
                        prompt = build_specialist_prompt(
                            role=role,
                            section=slice_section,
                            rhythm=rhythm,
                            grid=grid,
                            clip_start=clip_start,
                            clip_end=clip_end,
                            dsp=dsp,
                        )
                        future = executor.submit(
                            gateway.generate_typed,
                            contents=[part, prompt],
                            schema=CompactSpecialistDraft,
                            system_instruction=SPECIALIST_SYSTEM_PROMPTS[role],
                            temperature=0.0,
                            max_output_tokens=3_072,
                            retries=3,
                            diagnostic_label=f"specialist-{slice_section.id}-{role}",
                        )
                        future_map[future] = (
                            analysis_slice,
                            role,
                            prompt,
                            clip_start,
                            clip_end,
                            part,
                        )
                        diagnostics.section_specialist_calls += 1

                for future in as_completed(future_map):
                    analysis_slice, role, prompt, clip_start, clip_end, part = future_map[future]
                    slice_section = analysis_slice.section
                    label = f"specialist-{slice_section.id}-{role}"
                    try:
                        compact = future.result()
                    except Exception as exc:  # noqa: BLE001
                        diagnostics.record_error(label, exc)
                        diagnostics.record_required_failure(label, exc)
                        global_warnings.append(f"{label} failed: {type(exc).__name__}")
                        continue

                    raw_result = compact_to_specialist(
                        compact,
                        section=slice_section,
                        role=role,
                    )
                    result = normalise_specialist_result(
                        raw_result,
                        section=slice_section,
                        role=role,
                        clip_start=clip_start,
                        clip_end=clip_end,
                        beat_duration=grid.beat_duration,
                    )
                    if not result.chords:
                        try:
                            retry_compact = gateway.generate_typed(
                                contents=[
                                    part,
                                    prompt,
                                    "chordsが空です。担当区間を覆う最小限のコード状態を返し、聴取不能部分だけXにしてください。",
                                ],
                                schema=CompactSpecialistDraft,
                                system_instruction=SPECIALIST_SYSTEM_PROMPTS[role],
                                temperature=0.0,
                                max_output_tokens=3_072,
                                retries=2,
                                diagnostic_label=f"{label}-empty-retry",
                            )
                            result = normalise_specialist_result(
                                compact_to_specialist(
                                    retry_compact,
                                    section=slice_section,
                                    role=role,
                                ),
                                section=slice_section,
                                role=role,
                                clip_start=clip_start,
                                clip_end=clip_end,
                                beat_duration=grid.beat_duration,
                            )
                        except Exception as exc:  # noqa: BLE001
                            diagnostics.record_error(f"{label}-empty-retry", exc)
                            diagnostics.record_required_failure(label, exc)
                    if not result.chords:
                        diagnostics.record_required_failure(label)
                        global_warnings.append(f"{label} returned no usable chord events")
                        continue
                    accumulated[analysis_slice.parent_id][role].append(result)

        merged = self._merge_slice_results(sections, accumulated)
        for section in sections:
            for specialist in merged[section.id]:
                if not specialist.chords:
                    diagnostics.record_required_failure(
                        f"specialist-{section.id}-{specialist.role}-missing"
                    )
        return merged

    @staticmethod
    def _apply_compact_resolution(
        output: ConsensusOutput,
        target: UncertainRange,
        decision: CompactResolutionDraft,
    ) -> bool:
        chosen = canonicalize_symbol(decision.chosenSymbol)
        allowed = {
            canonicalize_symbol(item)
            for item in target.candidates + ["N", "X"]
        }
        if chosen not in allowed:
            raise ValueError(f"resolver returned a candidate outside the allow-list: {chosen}")
        if chosen == "X":
            return False

        resolution = ResolutionDraft(
            choices=[
                ResolutionChoice(
                    startSeconds=target.startSeconds,
                    endSeconds=target.endSeconds,
                    chosenSymbol=chosen,
                    confidence=_confidence(decision.confidence),
                    reason="",
                )
            ]
        )
        for chord in output.chords:
            if _overlap(
                target.startSeconds,
                target.endSeconds,
                chord.startSeconds,
                chord.endSeconds,
            ) > 0 and chosen not in chord.alternatives and chosen != chord.symbol:
                chord.alternatives = [chosen] + chord.alternatives[:2]
        before = [chord.symbol for chord in output.chords]
        apply_resolution(output.chords, resolution)
        after = [chord.symbol for chord in output.chords]
        return before != after or any(
            _overlap(
                target.startSeconds,
                target.endSeconds,
                chord.startSeconds,
                chord.endSeconds,
            ) > 0 and chord.symbol == chosen
            for chord in output.chords
        )

    def _run_resolvers(
        self,
        *,
        media: AnalysisMediaSource,
        resolver_gateway: ModelGateway,
        consensus_pairs: list[tuple[SectionStructureDraft, ConsensusOutput]],
        grid: BeatGrid,
        duration: float,
        diagnostics: DiagnosticsRecorder,
        global_warnings: list[str],
    ) -> list[UncertainRange]:
        all_uncertain = [
            item
            for _, output in consensus_pairs
            for item in output.uncertain_ranges
        ]
        resolver_targets = sorted(
            [item for item in all_uncertain if not item.resolved],
            key=lambda item: (-(item.endSeconds - item.startSeconds), item.startSeconds),
        )[: self.max_resolver_calls]

        with diagnostics.stage("targeted-resolution"):
            for target in resolver_targets:
                pair = next(
                    (pair for pair in consensus_pairs if pair[0].id == target.sectionId),
                    None,
                )
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
                bar_duration = grid.beat_duration * grid.beats_per_bar
                clip_start = max(0.0, target.startSeconds - bar_duration)
                clip_end = min(duration, target.endSeconds + bar_duration)
                label = f"resolver-{target.sectionId}-{target.startSeconds:.3f}"
                try:
                    decision = resolver_gateway.generate_typed(
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
                        schema=CompactResolutionDraft,
                        system_instruction=RESOLUTION_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=512,
                        retries=3,
                        diagnostic_label=label,
                    )
                    diagnostics.resolver_calls += 1
                    if self._apply_compact_resolution(output, target, decision):
                        target.resolved = True
                except Exception as exc:  # noqa: BLE001
                    diagnostics.record_error(label, exc)
                    diagnostics.record_required_failure(label, exc)
                    global_warnings.append(f"{label} failed: {type(exc).__name__}")
        return all_uncertain

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
                        diagnostic_label="global-structure",
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
                        diagnostic_label="global-rhythm",
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
                    except Exception as exc:  # noqa: BLE001
                        diagnostics.record_error(name, exc)
                        global_warnings.append(f"{name} failed: {type(exc).__name__}: {str(exc)[:240]}")
                        continue
                    if name == "structure":
                        structure = value
                    elif name == "rhythm":
                        rhythm = value
                    else:
                        references = value

        duration = self._duration(known_duration, structure, rhythm)
        if rhythm is None:
            rhythm = self._fallback_rhythm(duration, dsp)
            global_warnings.append("リズム分析が失敗したため、保守的なテンポ候補を使用しました。")

        sections = self._usable_sections(structure, duration=duration)
        if not sections:
            global_warnings.append("全曲構造分析が品質基準を満たさなかったため、時間窓で再解析しました。")
            structure = self._windowed_structure(
                media=media,
                gateway=gateway,
                source_label=source_label,
                duration=duration,
                dsp=dsp,
                metadata=metadata_dict,
                diagnostics=diagnostics,
            )
            sections = self._usable_sections(structure, duration=duration)

        if structure is None or not sections:
            raise RuntimeError(
                "Accuracy v2 structure analysis failed quality gates; "
                "no result was returned instead of fabricating a full-track section."
            )

        track_updates: dict[str, Any] = {"durationSeconds": duration}
        if youtube_metadata:
            if youtube_metadata.title:
                track_updates["title"] = youtube_metadata.title
            if youtube_metadata.channel_title:
                track_updates["artist"] = youtube_metadata.channel_title
        structure.track = structure.track.model_copy(update=track_updates)
        rhythm.durationSeconds = duration
        for bpm in references.bpm_candidates:
            if 20.0 <= bpm <= 320.0 and not any(
                abs(_finite(item.bpm, 0.0) - bpm) < 0.1
                for item in rhythm.bpmCandidates
            ):
                rhythm.bpmCandidates.append(
                    TempoCandidate(bpm=bpm, confidence=0.35, interpretation="grounded-reference")
                )

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

        sections = self._refine_structure(
            media=media,
            gateway=gateway,
            sections=sections,
            grid=grid,
            duration=duration,
            diagnostics=diagnostics,
        )
        sections = [
            section.model_copy(
                update={
                    "key": section.key or global_key,
                    "mode": section.mode or global_mode,
                }
            )
            for section in sections
        ]

        specialist_results = self._run_specialists(
            media=media,
            gateway=gateway,
            sections=sections,
            rhythm=rhythm,
            grid=grid,
            duration=duration,
            dsp=dsp,
            diagnostics=diagnostics,
            global_warnings=global_warnings,
        )

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
            reconcile_repeated_sections_enhanced(consensus_pairs)

        all_uncertain = self._run_resolvers(
            media=media,
            resolver_gateway=resolver_gateway,
            consensus_pairs=consensus_pairs,
            grid=grid,
            duration=duration,
            diagnostics=diagnostics,
            global_warnings=global_warnings,
        )

        diagnostics.chord_state_count = sum(
            len(output.chords)
            for _, output in consensus_pairs
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
        diagnostics.display_chord_event_count = sum(
            len(measure.chords)
            for section in preliminary_sections
            for measure in section.measures
        )

        explanation = FinalExplanationDraft()
        with diagnostics.stage("final-explanation"):
            unresolved = [item for item in all_uncertain if not item.resolved]
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
                        "startSeconds": section.startSeconds,
                        "endSeconds": section.endSeconds,
                        "agreement": output.agreement,
                        "progression": _compact_progression(output),
                    }
                    for section, output in consensus_pairs
                ],
                "unresolvedSummary": {
                    "count": len(unresolved),
                    "seconds": round(
                        sum(item.endSeconds - item.startSeconds for item in unresolved),
                        3,
                    ),
                    "reasons": sorted({item.reason for item in unresolved}),
                },
                "referenceSummary": references.summary[:1_200],
            }
            try:
                explanation = gateway.generate_typed(
                    contents=[build_final_explanation_prompt(compact)],
                    schema=FinalExplanationDraft,
                    system_instruction=FINAL_EXPLANATION_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=2_048,
                    retries=3,
                    diagnostic_label="final-explanation",
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics.record_error("final-explanation", exc)
                global_warnings.append(
                    f"final explanation failed: {type(exc).__name__}: {str(exc)[:200]}"
                )

        for section in preliminary_sections:
            section.summary = explanation.sectionSummaries.get(section.id, section.summary)

        track_confidences = [
            _confidence(structure.track.confidence),
            _confidence(rhythm.confidence),
        ]
        track_confidences.extend(
            _confidence(output.agreement, 0.0)
            for _, output in consensus_pairs
            if output.agreement > 0
        )
        result = AnalysisResult(
            track=TrackResult(
                title=structure.track.title or source_label,
                artist=structure.track.artist or "不明",
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
        quality_errors = validate_quality(result)
        quality_errors.extend(
            f"required_model_failure:{_compact_failure_label(item)}"
            for item in diagnostics.required_failures
        )
        quality_errors = sorted(set(quality_errors))
        combined_errors = invariant_errors + [f"quality:{item}" for item in quality_errors]
        result.diagnostics = diagnostics.build(combined_errors)
        if invariant_errors:
            raise RuntimeError(
                "Accuracy v2 invariant violation: " + ", ".join(invariant_errors)
            )
        if quality_errors:
            raise RuntimeError(
                "Accuracy v2 quality gate violation: " + ", ".join(quality_errors)
            )
        return result
