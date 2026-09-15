from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class NormalizedRepresentation(BaseModel):
    """The Review 1 contract for all input modalities."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    job_id: str
    input: InputRecord
    content_items: list[ContentItem] = Field(default_factory=list)
    narration: list[NarrationItem] = Field(default_factory=list)
    instructions: InstructionRecord = Field(default_factory=InstructionRecord)
    semantic_control_plan: SemanticControlPlan | None = None
    artifacts: list[PipelineArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
