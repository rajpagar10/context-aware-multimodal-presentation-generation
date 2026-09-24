from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .alignment import FasterWhisperWordAligner, WordAligner
from .detection import detect_input
from .errors import PipelineError
from .extractors import build_extractor
from .instructions import InstructionParser, RuleBasedInstructionParser
from .media import MediaTools
from .models import (
    ContentKind,
    InputKind,
    InputRecord,
    InstructionRecord,
    NarrationItem,
    NormalizedRepresentation,
    PipelineArtifact,
    SpeakerProfile,
    SpeechAnalysis,
    SpeechRegion,
)
from .narration import NarrationGenerator, TemplateNarrationGenerator, generated_narration
from .prosody import NumpyProsodyExtractor, ProsodyExtractor
from .speaker import SpeakerEncoder
from .transcription import FasterWhisperTranscriber, TranscriptSegment, Transcriber, TranscriptionResult
from .tts import TTSEngine
from .utils import json_dump, sha256_file, stable_id

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: Path
    generate_narration: bool = False
    rewrite_existing_speech: bool = False
    enable_speech_analysis: bool = False
    max_input_bytes: int = 1024 * 1024 * 1024


class IngestionPipeline:
    def __init__(self, config: PipelineConfig, *, transcriber: Transcriber | None = None,
                 narrator: NarrationGenerator | None = None, instruction_parser: InstructionParser | None = None,
                 media_tools: MediaTools | None = None, speaker_encoder: SpeakerEncoder | None = None,
                 word_aligner: WordAligner | None = None, prosody_extractor: ProsodyExtractor | None = None,
                 tts_engine: TTSEngine | None = None):
        self.config = config
        self.transcriber = transcriber or FasterWhisperTranscriber()
        self.narrator = narrator or TemplateNarrationGenerator()
        self.instruction_parser = instruction_parser or RuleBasedInstructionParser()
        self.media_tools = media_tools or MediaTools()
        self.speaker_encoder = speaker_encoder
        self.word_aligner = word_aligner or FasterWhisperWordAligner()
        self.prosody_extractor = prosody_extractor or NumpyProsodyExtractor()
        self.tts_engine = tts_engine

    def run(self, input_path: str | Path, *, instruction: str | None = None) -> tuple[NormalizedRepresentation, Path]:
        detected = detect_input(input_path, max_bytes=self.config.max_input_bytes)
        LOGGER.info("Detected %s input: %s", detected.kind.value, detected.path.name)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        extractor = build_extractor(detected.kind, transcriber=self.transcriber, media_tools=self.media_tools)
        extracted = extractor.extract(detected, output_dir=self.config.output_dir)
        digest = sha256_file(detected.path)
        representation = NormalizedRepresentation(
            job_id=stable_id("job", digest),
            input=InputRecord(id=stable_id("input", digest), kind=detected.kind, filename=detected.path.name,
                              sha256=digest, mime_type=detected.mime_type,
                              metadata={"size_bytes": detected.size_bytes, **extracted.metadata}),
            content_items=extracted.items, artifacts=extracted.artifacts, warnings=extracted.warnings,
            instructions=InstructionRecord(raw_text=instruction, parser=self.instruction_parser.name if instruction else None),
        )
        representation.narration = self._narration_for(representation)
        if instruction:
            representation.semantic_control_plan = self.instruction_parser.parse(instruction, representation.content_items)
        if self.config.enable_speech_analysis:
            self._populate_speech_analysis(representation, detected.path, detected.kind)
        result_path = self.config.output_dir / "normalized.json"
        json_dump(result_path, representation.model_dump(mode="json"))
        LOGGER.info("Wrote normalized output: %s", result_path)
        return representation, result_path

    def _narration_for(self, representation: NormalizedRepresentation) -> list[NarrationItem]:
        kind = representation.input.kind
        if kind == InputKind.TEXT:
            return [NarrationItem(id=stable_id("narration", item.id, "user_script"), content_item_ids=[item.id], text=item.text,
                                  origin="user_script", generation_status="not_requested", metadata={"preserved": True})
                    for item in representation.content_items]
        if kind in {InputKind.AUDIO, InputKind.VIDEO} and not self.config.rewrite_existing_speech:
            return [NarrationItem(id=stable_id("narration", item.id, "source_transcript"), content_item_ids=[item.id], text=item.text,
                                  origin="source_transcript", generation_status="skipped_existing_speech", metadata={"preserved": True})
                    for item in representation.content_items]
        if self.config.generate_narration or self.config.rewrite_existing_speech:
            return generated_narration(representation.content_items, self.narrator)
        return []

    def _populate_speech_analysis(self, representation: NormalizedRepresentation, input_path: Path, input_kind: InputKind) -> None:
        """Attach optional Task 2 evidence without invalidating a Task 1 result."""
        analysis = SpeechAnalysis()
        attempted_sources = 0

        source_audio = self._source_audio_path(representation, input_path, input_kind)
        source_transcript = self._source_transcription(representation)
        if source_audio is not None and source_transcript is not None:
            attempted_sources += 1
            self._analyze_audio(
                representation,
                analysis,
                source_audio,
                source_transcript,
                source="source_video" if input_kind == InputKind.VIDEO else "source_audio",
                content_item_ids=[item.id for item in representation.content_items if item.kind == ContentKind.TRANSCRIPT_SEGMENT],
                narration_item_ids=[item.id for item in representation.narration if item.origin == "source_transcript"],
            )

        if self.tts_engine is not None:
            for narration in representation.narration:
                attempted_sources += 1
                tts_path = self.config.output_dir / "derived" / f"{narration.id}.wav"
                try:
                    result = self.tts_engine.synthesize(narration.text, tts_path)
                    transcript = self.transcriber.transcribe(result.output_path)
                    representation.artifacts.append(PipelineArtifact(
                        kind="generated_tts",
                        path=str(result.output_path),
                        metadata=result.metadata,
                    ))
                    self._analyze_audio(
                        representation,
                        analysis,
                        result.output_path,
                        transcript,
                        source="generated_tts",
                        content_item_ids=narration.content_item_ids,
                        narration_item_ids=[narration.id],
                    )
                except PipelineError as exc:
                    self._record_analysis_error(representation, analysis, exc)

        if attempted_sources == 0:
            analysis.status = "unavailable"
            analysis.warnings.append("Speech analysis was enabled, but no source audio or configured TTS narration was available.")
        elif representation.errors and not (analysis.speech_regions or analysis.word_alignments or analysis.prosody):
            analysis.status = "failed"
        elif analysis.speakers and analysis.speech_regions and analysis.word_alignments and analysis.prosody and not analysis.warnings:
            analysis.status = "complete"
        else:
            analysis.status = "partial"
        representation.speech_analysis = analysis

    def _analyze_audio(
        self,
        representation: NormalizedRepresentation,
        analysis: SpeechAnalysis,
        audio_path: Path,
        transcript: TranscriptionResult,
        *,
        source: str,
        content_item_ids: list[str],
        narration_item_ids: list[str],
    ) -> None:
        speaker_id = self._speaker_id_for(representation, analysis, audio_path)
        region_ids_by_segment: dict[int, str] = {}
        source_regions: list[SpeechRegion] = []
        for segment in transcript.segments:
            region_id = f"speech_region_{len(analysis.speech_regions) + 1:06d}"
            region_ids_by_segment[segment.index] = region_id
            region = SpeechRegion(
                id=region_id,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                speaker_id=speaker_id,
                source=source,
                content_item_ids=content_item_ids,
                narration_item_ids=narration_item_ids,
                metadata={"transcription_engine": transcript.engine, "segment_index": segment.index},
            )
            analysis.speech_regions.append(region)
            source_regions.append(region)

        source_word_alignments = []
        try:
            aligned_words = self.word_aligner.align(transcript, speaker_id=speaker_id)
            for word in aligned_words:
                word.id = f"word_{len(analysis.word_alignments) + 1:06d}"
                word.speech_region_id = region_ids_by_segment.get(word.segment_index)
                analysis.word_alignments.append(word)
                source_word_alignments.append(word)
        except PipelineError as exc:
            self._record_analysis_error(representation, analysis, exc)

        if audio_path.suffix.lower() != ".wav":
            analysis.warnings.append(f"Prosody analysis skipped for '{audio_path.name}': the baseline extractor requires PCM WAV input.")
            return
        try:
            prosody_frames = self.prosody_extractor.extract(
                audio_path,
                word_alignments=source_word_alignments,
                speaker_id=speaker_id,
            )
            for frame in prosody_frames:
                region = next(
                    (candidate for candidate in source_regions
                     if frame.start_seconds < candidate.end_seconds and frame.end_seconds > candidate.start_seconds),
                    None,
                )
                frame.speech_region_id = region.id if region is not None else None
            analysis.prosody.extend(prosody_frames)
        except PipelineError as exc:
            self._record_analysis_error(representation, analysis, exc)

    def _speaker_id_for(self, representation: NormalizedRepresentation, analysis: SpeechAnalysis, audio_path: Path) -> str | None:
        if self.speaker_encoder is None:
            analysis.warnings.append("Speaker analysis skipped: no SpeakerEncoder is configured.")
            return None
        try:
            embedding = self.speaker_encoder.encode(audio_path)
        except PipelineError as exc:
            self._record_analysis_error(representation, analysis, exc)
            return None
        speaker_id = stable_id("speaker", audio_path, embedding.model_name)
        analysis.speakers.append(SpeakerProfile(
            speaker_id=speaker_id,
            embedding=embedding.vector,
            embedding_model=embedding.model_name,
            embedding_dimension=embedding.dimension,
            metadata=embedding.metadata,
        ))
        return speaker_id

    @staticmethod
    def _source_audio_path(representation: NormalizedRepresentation, input_path: Path, input_kind: InputKind) -> Path | None:
        if input_kind == InputKind.AUDIO:
            return input_path
        if input_kind == InputKind.VIDEO:
            for artifact in representation.artifacts:
                if artifact.kind == "extracted_audio":
                    return Path(artifact.path)
        return None

    @staticmethod
    def _source_transcription(representation: NormalizedRepresentation) -> TranscriptionResult | None:
        segments: list[TranscriptSegment] = []
        for item in representation.content_items:
            timing = item.source_timing
            if item.kind != ContentKind.TRANSCRIPT_SEGMENT or timing.start_seconds is None or timing.end_seconds is None:
                continue
            words = item.metadata.get("word_timestamps", [])
            if not isinstance(words, list):
                continue
            segments.append(TranscriptSegment(item.index, item.text, timing.start_seconds, timing.end_seconds, words))
        if not segments:
            return None
        transcription_metadata = representation.input.metadata.get("transcription", {})
        if not isinstance(transcription_metadata, dict):
            transcription_metadata = {}
        return TranscriptionResult(
            language=transcription_metadata.get("language"),
            duration_seconds=transcription_metadata.get("duration_seconds"),
            segments=segments,
            engine=str(transcription_metadata.get("engine", "unknown")),
        )

    @staticmethod
    def _record_analysis_error(representation: NormalizedRepresentation, analysis: SpeechAnalysis, error: PipelineError) -> None:
        message = f"Speech analysis {error.stage}: {error}"
        analysis.warnings.append(message)
        representation.warnings.append(message)
        representation.errors.append(error.as_dict())
