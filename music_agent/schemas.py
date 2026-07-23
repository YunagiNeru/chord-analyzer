from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class KeyCandidate(ApiModel):
    key: str
    score: float = Field(ge=0.0, le=1.0)


class ChordAlternative(ApiModel):
    symbol: str
    score: float = Field(ge=0.0, le=1.0)


class DspChordRun(ApiModel):
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    symbol: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[ChordAlternative] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> "DspChordRun":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class DspSummary(ApiModel):
    durationSeconds: float = Field(ge=0.001)
    bpm: float | None = Field(default=None, ge=0.001)
    keyCandidates: list[KeyCandidate] = Field(default_factory=list)
    beatTimes: list[float] = Field(default_factory=list)
    sectionBoundaryCandidates: list[float] = Field(default_factory=list)
    chordRuns: list[DspChordRun] = Field(default_factory=list)
    waveform: list[float] = Field(default_factory=list)


class TrackStructureDraft(ApiModel):
    title: str = "不明な楽曲"
    artist: str = "不明"
    durationSeconds: float = Field(ge=0.001)
    bpm: float | None = Field(default=None, ge=0.001)
    timeSignature: str | None = None
    globalKey: str | None = None
    globalMode: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SectionStructureDraft(ApiModel):
    id: str
    name: str
    type: Literal[
        "intro",
        "verse",
        "pre_chorus",
        "chorus",
        "post_chorus",
        "bridge",
        "interlude",
        "solo",
        "breakdown",
        "outro",
        "other",
    ] = "other"
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    key: str | None = None
    mode: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    notes: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "SectionStructureDraft":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class StructureDraft(ApiModel):
    track: TrackStructureDraft
    sections: list[SectionStructureDraft] = Field(min_length=1)
    observations: list[str] = Field(default_factory=list)


class TempoCandidate(ApiModel):
    bpm: float = Field(ge=20.0, le=320.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    interpretation: str = ""


class TempoSegment(ApiModel):
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    bpm: float = Field(ge=20.0, le=320.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> "TempoSegment":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class RhythmDraft(ApiModel):
    durationSeconds: float = Field(ge=0.001)
    bpmCandidates: list[TempoCandidate] = Field(default_factory=list)
    selectedBpm: float | None = Field(default=None, ge=20.0, le=320.0)
    timeSignature: str = "4/4"
    downbeatOffsetSeconds: float = Field(default=0.0, ge=0.0)
    globalKey: str | None = None
    globalMode: str | None = None
    tempoSegments: list[TempoSegment] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    observations: list[str] = Field(default_factory=list)


class HarmonyChordDraft(ApiModel):
    symbol: str
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    alternatives: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> "HarmonyChordDraft":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class HarmonySectionDraft(ApiModel):
    sectionId: str
    key: str | None = None
    mode: str | None = None
    chords: list[HarmonyChordDraft] = Field(default_factory=list)


class HarmonyDraft(ApiModel):
    bpm: float | None = Field(default=None, ge=0.001)
    timeSignature: str | None = None
    globalKey: str | None = None
    globalMode: str | None = None
    sections: list[HarmonySectionDraft] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class ChordComponentsDraft(ApiModel):
    root: str | None = None
    quality: Literal[
        "major",
        "minor",
        "diminished",
        "augmented",
        "sus2",
        "sus4",
        "power",
        "no_chord",
        "unknown",
    ] = "unknown"
    seventh: Literal[
        "none",
        "minor7",
        "major7",
        "diminished7",
        "unknown",
    ] = "none"
    extensions: list[str] = Field(default_factory=list)
    alterations: list[str] = Field(default_factory=list)
    bass: str | None = None


class SpecialistChordDraft(ApiModel):
    symbol: str
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    components: ChordComponentsDraft = Field(default_factory=ChordComponentsDraft)
    alternatives: list[str] = Field(default_factory=list)
    evidence: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "SpecialistChordDraft":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class SpecialistSectionDraft(ApiModel):
    sectionId: str
    role: Literal["root_quality", "bass_extension", "rhythm_pattern"]
    key: str | None = None
    mode: str | None = None
    chords: list[SpecialistChordDraft] = Field(default_factory=list)
    repeatedPattern: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class ResolutionChoice(ApiModel):
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    chosenSymbol: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "ResolutionChoice":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class ResolutionDraft(ApiModel):
    choices: list[ResolutionChoice] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class FinalExplanationDraft(ApiModel):
    musicalSummary: str = ""
    warnings: list[str] = Field(default_factory=list)
    sectionSummaries: dict[str, str] = Field(default_factory=dict)


class ChordEvent(ApiModel):
    symbol: str
    roman: str | None = None
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    bar: int | None = Field(default=None, ge=1)
    beat: float | None = Field(default=None, ge=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source: Literal["hybrid", "ai", "dsp"] = "hybrid"
    alternatives: list[str] = Field(default_factory=list)
    agreement: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> "ChordEvent":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class Measure(ApiModel):
    bar: int = Field(ge=1)
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    chords: list[ChordEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_range(self) -> "Measure":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class SectionResult(ApiModel):
    id: str
    name: str
    type: Literal[
        "intro",
        "verse",
        "pre_chorus",
        "chorus",
        "post_chorus",
        "bridge",
        "interlude",
        "solo",
        "breakdown",
        "outro",
        "other",
    ] = "other"
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    key: str | None = None
    mode: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""
    measures: list[Measure] = Field(default_factory=list)
    agreement: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_range(self) -> "SectionResult":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class TrackResult(ApiModel):
    title: str = "不明な楽曲"
    artist: str = "不明"
    durationSeconds: float = Field(ge=0.001)
    bpm: float | None = Field(default=None, ge=0.001)
    timeSignature: str | None = None
    globalKey: str | None = None
    globalMode: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AIAnalysisDraft(ApiModel):
    track: TrackResult
    sections: list[SectionResult] = Field(min_length=1)
    musicalSummary: str = ""
    warnings: list[str] = Field(default_factory=list)


class ReferenceSource(ApiModel):
    title: str
    url: str
    sourceType: str = "web"
    facts: list[str] = Field(default_factory=list)


class UncertainRange(ApiModel):
    sectionId: str
    startSeconds: float = Field(ge=0.0)
    endSeconds: float = Field(ge=0.001)
    reason: str
    candidates: list[str] = Field(default_factory=list)
    resolved: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> "UncertainRange":
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class StageTiming(ApiModel):
    stage: str
    seconds: float = Field(ge=0.0)


class ModelUsage(ApiModel):
    model: str
    calls: int = Field(default=0, ge=0)
    inputTokens: int = Field(default=0, ge=0)
    outputTokens: int = Field(default=0, ge=0)


class AnalysisDiagnostics(ApiModel):
    pipeline: str = "accuracy-v2"
    stageTimings: list[StageTiming] = Field(default_factory=list)
    modelUsage: list[ModelUsage] = Field(default_factory=list)
    invariantErrors: list[str] = Field(default_factory=list)
    sectionSpecialistCalls: int = Field(default=0, ge=0)
    resolverCalls: int = Field(default=0, ge=0)
    gridScore: float | None = None


class AnalysisResult(AIAnalysisDraft):
    analysisVersion: str = "2.0"
    sourceType: Literal["upload", "youtube", "direct_audio_url"]
    sourceLabel: str
    analysisMethod: Literal["hybrid", "ai_only"]
    waveform: list[float] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    tempoSegments: list[TempoSegment] = Field(default_factory=list)
    downbeatOffsetSeconds: float | None = Field(default=None, ge=0.0)
    referenceSources: list[ReferenceSource] = Field(default_factory=list)
    uncertainRanges: list[UncertainRange] = Field(default_factory=list)
    diagnostics: AnalysisDiagnostics | None = None


class QuestionRoute(ApiModel):
    route: Literal["analysis_only", "web_research"]
    reason: str


class QuestionRequest(ApiModel):
    question: str = Field(min_length=1, max_length=1000)
    analysis: AnalysisResult

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question is empty")
        return value


class SourceLink(ApiModel):
    title: str
    url: str


class QuestionResponse(ApiModel):
    answer: str
    usedWebSearch: bool = False
    routeReason: str = ""
    sources: list[SourceLink] = Field(default_factory=list)
    searchQueries: list[str] = Field(default_factory=list)
    searchEntryPointHtml: str = ""
