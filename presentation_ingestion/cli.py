from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .errors import PipelineError
from .instructions import ApiInstructionParser, RuleBasedInstructionParser
from .llm import OpenAICompatibleBackend
from .narration import ApiNarrationGenerator, TemplateNarrationGenerator
from .pipeline import IngestionPipeline, PipelineConfig
from .transcription import FasterWhisperTranscriber
from .utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review 1 multimodal ingestion and semantic-instruction pipeline")
    parser.add_argument("--input", required=True, help="PPTX, PDF, video, audio, TXT, or DOCX input")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized.json and derived media")
    parser.add_argument("--instruction", help="Natural-language semantic delivery instruction")
    parser.add_argument("--generate-narration", action="store_true", help="Generate narration for document/presentation content")
    parser.add_argument("--rewrite-existing-speech", action="store_true", help="Explicitly allow narration generation for audio/video transcripts")
    parser.add_argument("--narration-mode", choices=("template", "api"), default="template")
    parser.add_argument("--instruction-mode", choices=("rule", "api"), default="rule")
    parser.add_argument("--asr-model", default="base", help="faster-whisper model name")
    parser.add_argument("--asr-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--asr-compute-type", default="int8")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        backend = OpenAICompatibleBackend() if "api" in {args.narration_mode, args.instruction_mode} else None
        narrator = ApiNarrationGenerator(backend) if args.narration_mode == "api" and backend else TemplateNarrationGenerator()
        instruction_parser = ApiInstructionParser(backend) if args.instruction_mode == "api" and backend else RuleBasedInstructionParser()
        pipeline = IngestionPipeline(
            PipelineConfig(Path(args.output_dir), generate_narration=args.generate_narration, rewrite_existing_speech=args.rewrite_existing_speech),
            transcriber=FasterWhisperTranscriber(args.asr_model, args.asr_device, args.asr_compute_type), narrator=narrator,
            instruction_parser=instruction_parser,
        )
        _, output = pipeline.run(args.input, instruction=args.instruction)
    except PipelineError as exc:
        logging.error("Pipeline failed at %s for %s (%s): %s", exc.stage, exc.path or args.input, exc.modality or "unknown modality", exc)
        return 2
    except ValueError as exc:
        logging.error("Pipeline returned invalid structured data: %s", exc)
        return 2
    print(output)
    return 0
