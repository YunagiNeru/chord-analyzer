from __future__ import annotations

from typing import Annotated, Literal

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


# Model-facing contracts are deliberately permissive. Semantic validation,
# canonicalisation, range clipping, and coverage checks happen in deterministic
# Python so one malformed optional field cannot discard an otherwise useful run.
class TrackStructureDraft(ApiModel):
    title: str = "不明な楽曲"
    artist: str = "不明"
    durationSeconds: float = 0.001
    bpm: float | None = None
    timeSignature: str | None = None
    globalKey: str | None = None
    globalMode: str | None = None
    confidence: float = 0.5


class SectionStructureDraft(ApiModel):
    id: str = ""
    name: str = ""
    type: str = "other"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    key: str | None = None
    mode: str | None = None
    confidence: float = 0.5
    notes: str = ""


class StructureDraft(ApiModel):
    track: TrackStructureDraft = Field(default_factory=TrackStructureDraft)
    sections: list[SectionStructureDraft] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class RefinedSectionDraft(ApiModel):
    name: str = ""
    type: str = "other"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5


class StructureRefinementDraft(ApiModel):
    sections: list[RefinedSectionDraft] = Field(default_factory=list, max_length=8)


class TempoCandidate(ApiModel):
    bpm: float = 120.0
    confidence: float = 0.5
    interpretation: str = ""


class TempoSegment(ApiModel):
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    bpm: float = 120.0
    confidence: float = 0.5


class RhythmDraft(ApiModel):
    durationSeconds: float = 0.001
    bpmCandidates: list[TempoCandidate] = Field(default_factory=list)
    selectedBpm: float | None = None
    timeSignature: str = "4/4"
    downbeatOffsetSeconds: float = 0.0
    globalKey: str | None = None
    globalMode: str | None = None
    tempoSegments: list[TempoSegment] = Field(default_factory=list)
    confidence: float = 0.5
    observations: list[str] = Field(default_factory=list)


class HarmonyChordDraft(ApiModel):
    symbol: str = "X"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5
    alternatives: list[str] = Field(default_factory=list)


class HarmonySectionDraft(ApiModel):
    sectionId: str = ""
    key: str | None = None
    mode: str | None = None
    chords: list[HarmonyChordDraft] = Field(default_factory=list)


class HarmonyDraft(ApiModel):
    bpm: float | None = None
    timeSignature: str | None = None
    globalKey: str | None = None
    globalMode: str | None = None
    sections: list[HarmonySectionDraft] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class ChordComponentsDraft(ApiModel):
    root: str | None = None
    quality: str = "unknown"
    seventh: str = "none"
    extensions: list[str] = Field(default_factory=list)
    alterations: list[str] = Field(default_factory=list)
    bass: str | None = None


class SpecialistChordDraft(ApiModel):
    symbol: str = "X"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5
    components: ChordComponentsDraft = Field(default_factory=ChordComponentsDraft)
    alternatives: list[str] = Field(default_factory=list)
    evidence: str = ""


class SpecialistSectionDraft(ApiModel):
    sectionId: str = ""
    role: str = ""
    key: str | None = None
    mode: str | None = None
    chords: list[SpecialistChordDraft] = Field(default_factory=list)
    repeatedPattern: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class CompactChordDraft(ApiModel):
    """Small role-neutral chord state returned by section specialists."""

    symbol: str = "X"
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    confidence: float = 0.5
    alternatives: list[str] = Field(default_factory=list, max_length=3)


class CompactSpecialistDraft(ApiModel):
    chords: list[CompactChordDraft] = Field(default_factory=list, max_length=96)
    repeatedPattern: list[str] = Field(default_factory=list, max_length=16)


ShortResolutionReason = Annotated[str, Field(max_length=60)]
ShortObservation = Annotated[str, Field(max_length=100)]
ShortWarning = Annotated[str, Field(max_length=180)]
ShortSectionSummary = Annotated[str, Field(max_length=100)]


class ResolutionChoice(ApiModel):
    startSeconds: float = 0.0
    endSeconds: float = 0.0
    chosenSymbol: str = "X"
    confidence: float = 0.5
    reason: ShortResolutionReason = ""


class ResolutionDraft(ApiModel):
    choices: list[ResolutionChoice] = Field(default_factory=list, max_length=4)
    observations: list[ShortObservation] = Field(default_factory=list, max_length=2)


class CompactResolutionDraft(ApiModel):
    """Exactly one explicit resolver decision; no free-form reason text."""

    chosenSymbol: str
    confidence: float


class FinalExplanationDraft(ApiModel):
    musicalSummary: str = Field(default="", max_length=350)
    warnings: list[ShortWarning] = Field(default_factory=list, max_length=8)
    sectionSummaries: dict[str, ShortSectionSummary] = Field(default_factory=dict)


# Final API models remain strict. Only deterministic Python is allowed to create
# these objects.
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
    modelErrors: list[str] = Field(default_factory=list)
    requiredModelFailures: list[str] = Field(default_factory=list)
    sectionSpecialistCalls: int = Field(default=0, ge=0)
    resolverCalls: int = Field(default=0, ge=0)
    structureRefinementCalls: int = Field(default=0, ge=0)
    analysisSliceCount: int = Field(default=0, ge=0)
    chordStateCount: int = Field(default=0, ge=0)
    displayChordEventCount: int = Field(default=0, ge=0)
    boundaryAdjustments: int = Field(default=0, ge=0)
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
