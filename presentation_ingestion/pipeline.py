from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .detection import detect_input
from .extractors import build_extractor
from .instructions import InstructionParser, RuleBasedInstructionParser
from .media import MediaTools
from .models import InputKind, InputRecord, InstructionRecord, NarrationItem, NormalizedRepresentation
from .narration import NarrationGenerator, TemplateNarrationGenerator, generated_narration
from .transcription import FasterWhisperTranscriber, Transcriber
from .utils import json_dump, sha256_file, stable_id

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: Path
    generate_narration: bool = False
    rewrite_existing_speech: bool = False
    max_input_bytes: int = 1024 * 1024 * 1024


class IngestionPipeline:
    def __init__(self, config: PipelineConfig, *, transcriber: Transcriber | None = None,
                 narrator: NarrationGenerator | None = None, instruction_parser: InstructionParser | None = None,
                 media_tools: MediaTools | None = None):
        self.config = config
        self.transcriber = transcriber or FasterWhisperTranscriber()
        self.narrator = narrator or TemplateNarrationGenerator()
        self.instruction_parser = instruction_parser or RuleBasedInstructionParser()
        self.media_tools = media_tools or MediaTools()

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
