from __future__ import annotations

import json
from math import pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from presentation_ingestion.errors import PipelineError
from presentation_ingestion.models import NormalizedRepresentation
from presentation_ingestion.pipeline import IngestionPipeline, PipelineConfig
from presentation_ingestion.speaker import SpeakerEmbedding, SpeakerEncoder
from presentation_ingestion.transcription import TranscriptSegment, TranscriptionResult, Transcriber
from presentation_ingestion.tts import TTSEngine, TTSResult


class StubTranscriber(Transcriber):
    def transcribe(self, media_path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            language="en",
            duration_seconds=1.0,
            engine="stub_asr",
            segments=[TranscriptSegment(
                1,
                "Hello world",
                0.0,
                1.0,
                [
                    {"word": "Hello", "start_seconds": 0.0, "end_seconds": 0.4, "probability": 0.98},
                    {"word": "world", "start_seconds": 0.55, "end_seconds": 1.0, "probability": 0.96},
                ],
            )],
        )


class StubSpeakerEncoder(SpeakerEncoder):
    def encode(self, audio: object) -> SpeakerEmbedding:
        return SpeakerEmbedding([0.1, 0.2, 0.3], "stub-speaker", 3, {"backend": "stub"})


class FailingSpeakerEncoder(SpeakerEncoder):
    def encode(self, audio: object) -> SpeakerEmbedding:
        raise PipelineError("speaker model is unavailable", stage="speaker_embedding", modality="audio")


class StubTTSEngine(TTSEngine):
    def synthesize(self, text: str, output_path: str | Path) -> TTSResult:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 16000
        samples = [round(12000 * sin(2 * pi * 200 * index / sample_rate)) for index in range(sample_rate)]
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples))
        return TTSResult(destination, sample_rate, 1.0, {"backend": "stub-tts"})


class PipelineTaskTwoIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.script = self.root / "script.txt"
        self.script.write_text("Hello world", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_task_two_services_enrich_the_normalized_representation(self) -> None:
        pipeline = IngestionPipeline(
            PipelineConfig(self.root / "integrated", enable_speech_analysis=True),
            transcriber=StubTranscriber(),
            speaker_encoder=StubSpeakerEncoder(),
            tts_engine=StubTTSEngine(),
        )

        representation, _ = pipeline.run(self.script)
        analysis = representation.speech_analysis

        self.assertEqual(representation.content_items[0].text, "Hello world")
        self.assertEqual(representation.narration[0].text, "Hello world")
        self.assertEqual(analysis.status, "complete")
        self.assertEqual(len(analysis.speakers), 1)
        self.assertEqual(len(analysis.speech_regions), 1)
        self.assertEqual([word.word for word in analysis.word_alignments], ["Hello", "world"])
        self.assertEqual(analysis.speech_regions[0].start_seconds, 0.0)
        self.assertEqual(analysis.speech_regions[0].end_seconds, 1.0)
        self.assertEqual(analysis.word_alignments[0].word, "Hello")
        self.assertEqual(analysis.word_alignments[0].start_seconds, 0.0)
        self.assertEqual(analysis.word_alignments[0].end_seconds, 0.4)
        self.assertEqual(analysis.word_alignments[0].confidence, 0.98)
        self.assertEqual(analysis.word_alignments[0].alignment_method, "asr")
        self.assertEqual(analysis.word_alignments[1].word, "world")
        self.assertEqual(analysis.word_alignments[1].start_seconds, 0.55)
        self.assertEqual(analysis.word_alignments[1].end_seconds, 1.0)
        self.assertEqual(analysis.word_alignments[1].confidence, 0.96)
        self.assertEqual(analysis.word_alignments[1].alignment_method, "asr")
        self.assertTrue(analysis.prosody)
        self.assertEqual(analysis.speech_regions[0].source, "generated_tts")
        self.assertEqual(analysis.word_alignments[0].speech_region_id, analysis.speech_regions[0].id)
        self.assertEqual(analysis.word_alignments[0].speaker_id, analysis.speakers[0].speaker_id)
        self.assertTrue(all(frame.speech_region_id == analysis.speech_regions[0].id for frame in analysis.prosody))
        self.assertTrue(any(artifact.kind == "generated_tts" for artifact in representation.artifacts))

        restored = NormalizedRepresentation.model_validate_json(json.dumps(representation.model_dump(mode="json")))
        self.assertEqual(restored.speech_analysis.speakers[0].embedding_dimension, 3)
        self.assertEqual(restored.speech_analysis.word_alignments[1].word, "world")

    def test_task_one_behavior_is_unchanged_when_speech_analysis_is_disabled(self) -> None:
        pipeline = IngestionPipeline(PipelineConfig(self.root / "task_one"), transcriber=StubTranscriber())

        representation, _ = pipeline.run(self.script)

        self.assertEqual(representation.content_items[0].text, "Hello world")
        self.assertEqual(representation.narration[0].origin, "user_script")
        self.assertEqual(representation.speech_analysis.status, "not_started")

    def test_source_audio_reuses_existing_transcription_evidence(self) -> None:
        audio_path = self.root / "source.wav"
        StubTTSEngine().synthesize("source", audio_path)
        pipeline = IngestionPipeline(
            PipelineConfig(self.root / "source", enable_speech_analysis=True),
            transcriber=StubTranscriber(),
            speaker_encoder=StubSpeakerEncoder(),
        )

        representation, _ = pipeline.run(audio_path)
        analysis = representation.speech_analysis

        self.assertEqual(analysis.status, "complete")
        self.assertEqual(analysis.speech_regions[0].source, "source_audio")
        self.assertEqual([word.word for word in analysis.word_alignments], ["Hello", "world"])
        self.assertTrue(analysis.prosody)

    def test_optional_speaker_failure_keeps_task_one_and_other_task_two_evidence(self) -> None:
        pipeline = IngestionPipeline(
            PipelineConfig(self.root / "partial", enable_speech_analysis=True),
            transcriber=StubTranscriber(),
            speaker_encoder=FailingSpeakerEncoder(),
            tts_engine=StubTTSEngine(),
        )

        representation, _ = pipeline.run(self.script)

        self.assertEqual(representation.content_items[0].text, "Hello world")
        self.assertEqual(representation.speech_analysis.status, "partial")
        self.assertFalse(representation.speech_analysis.speakers)
        self.assertTrue(representation.speech_analysis.word_alignments)
        self.assertTrue(representation.speech_analysis.prosody)
        self.assertTrue(representation.errors)


if __name__ == "__main__":
    unittest.main()
