from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputKind(str, Enum):
    PPTX = "pptx"
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    DOCX = "docx"


class ContentKind(str, Enum):
    SLIDE = "slide"
    PAGE = "page"
    TRANSCRIPT_SEGMENT = "transcript_segment"
    TEXT_SECTION = "text_section"
    DOCUMENT_SECTION = "document_section"


class SourceTiming(BaseModel):
    """Timing present in the source or created by ASR, never future alignment."""

    model_config = ConfigDict(extra="forbid")
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    provenance: Literal["none", "container", "asr"] = "none"


class ContentStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = None
    body: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)
    speaker_notes: str | None = None


class InputRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: InputKind
    filename: str
    sha256: str
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: ContentKind
    index: int = Field(ge=1)
    text: str = ""
    structure: ContentStructure = Field(default_factory=ContentStructure)
    source_timing: SourceTiming = Field(default_factory=SourceTiming)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NarrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    content_item_ids: list[str]
    text: str
    origin: Literal["generated", "user_script", "source_transcript"]
    generation_status: Literal["generated", "not_requested", "skipped_existing_speech"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class InstructionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_text: str | None = None
    parser: str | None = None


class ControlTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_item_id: str | None = None
    selector_type: Literal["section", "exact_text", "keyword", "unresolved"]
    selector_value: str
    occurrence: int | None = Field(default=None, ge=1)


class ControlAttributes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    emotion: str | None = None
    energy: float | None = Field(default=None, ge=0, le=1)
    pitch: float | None = Field(default=None, ge=0, le=1)
    rate: float | None = Field(default=None, ge=0, le=1)
    emphasis: float | None = Field(default=None, ge=0, le=1)
    pause_before: bool | None = None
    pause_after: bool | None = None


class SemanticDirective(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    target: ControlTarget
    attributes: ControlAttributes
    confidence: float = Field(ge=0, le=1)
    timing_status: Literal["semantic_pending_alignment"] = "semantic_pending_alignment"


class SemanticControlPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = "1.0"
    status: Literal["semantic_pending_alignment", "no_directives"]
    directives: list[SemanticDirective] = Field(default_factory=list)
    parser: str
    warnings: list[str] = Field(default_factory=list)


class PipelineArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeakerProfile(BaseModel):
    """Identity/profile data for a speaker, when available."""

    model_config = ConfigDict(extra="forbid")
    speaker_id: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeechRegion(BaseModel):
    """A temporally bounded region of speech for Task 2 alignment."""

    model_config = ConfigDict(extra="forbid")
    id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    speaker_id: str | None = None
    source: Literal["source_audio", "source_video", "generated_tts", "aligned_audio", "unknown"] = "unknown"
    confidence: float | None = Field(default=None, ge=0, le=1)
    content_item_ids: list[str] = Field(default_factory=list)
    narration_item_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_order(self) -> SpeechRegion:
        if self.start_seconds > self.end_seconds:
            raise ValueError("start_seconds must be less than or equal to end_seconds")
        return self


class WordAlignment(BaseModel):
    """Timing and references for an individual spoken word."""

    model_config = ConfigDict(extra="forbid")
    id: str
    word: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    speaker_id: str | None = None
    speech_region_id: str | None = None
    content_item_id: str | None = None
    narration_item_id: str | None = None
    segment_index: int | None = Field(default=None, ge=1)
    alignment_method: Literal["asr", "forced_alignment", "unknown"] = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_order(self) -> WordAlignment:
        if self.start_seconds > self.end_seconds:
            raise ValueError("start_seconds must be less than or equal to end_seconds")
        return self


class PhonemeAlignment(BaseModel):
    """Optional future phoneme-level timing information."""

    model_config = ConfigDict(extra="forbid")
    id: str
    phoneme: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    word_alignment_id: str | None = None
    speaker_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    alignment_method: Literal["forced_alignment", "unknown"] = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_order(self) -> PhonemeAlignment:
        if self.start_seconds > self.end_seconds:
            raise ValueError("start_seconds must be less than or equal to end_seconds")
        return self


class ProsodyFrame(BaseModel):
    """Frame-level baseline prosody measurements for later control."""

    model_config = ConfigDict(extra="forbid")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    speaker_id: str | None = None
    f0_hz: float | None = Field(default=None, ge=0)
    energy: float | None = Field(default=None, ge=0)
    speaking_rate_wpm: float | None = Field(default=None, ge=0)
    voiced: bool | None = None
    speech_region_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_order(self) -> ProsodyFrame:
        if self.start_seconds > self.end_seconds:
            raise ValueError("start_seconds must be less than or equal to end_seconds")
        return self


class SpeechAnalysis(BaseModel):
    """Task 2 speech identity, alignment, and baseline prosody data."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["not_started", "partial", "complete", "unavailable", "failed"] = "not_started"
    speakers: list[SpeakerProfile] = Field(default_factory=list)
    speech_regions: list[SpeechRegion] = Field(default_factory=list)
    word_alignments: list[WordAlignment] = Field(default_factory=list)
    phoneme_alignments: list[PhonemeAlignment] = Field(default_factory=list)
    prosody: list[ProsodyFrame] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class NormalizedRepresentation(BaseModel):
    """The Review 1 + Review 2 normalized contract."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.1"
    job_id: str
    input: InputRecord
    content_items: list[ContentItem] = Field(default_factory=list)
    narration: list[NarrationItem] = Field(default_factory=list)
    instructions: InstructionRecord = Field(default_factory=InstructionRecord)
    semantic_control_plan: SemanticControlPlan | None = None
    speech_analysis: SpeechAnalysis = Field(default_factory=SpeechAnalysis)
    artifacts: list[PipelineArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
