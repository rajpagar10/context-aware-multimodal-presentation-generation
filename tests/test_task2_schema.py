from __future__ import annotations

import unittest

from pydantic import ValidationError

from presentation_ingestion.models import (
    InputKind,
    InputRecord,
    NormalizedRepresentation,
    PhonemeAlignment,
    ProsodyFrame,
    SpeakerProfile,
    SpeechAnalysis,
    SpeechRegion,
    WordAlignment,
)


def minimal_representation(**kwargs: object) -> NormalizedRepresentation:
    return NormalizedRepresentation(
        job_id="job_1",
        input=InputRecord(id="input_1", kind=InputKind.TEXT, filename="script.txt", sha256="abc123"),
        **kwargs,
    )


class TaskTwoSchemaTests(unittest.TestCase):
    def test_default_speech_analysis(self) -> None:
        representation = minimal_representation()

        self.assertEqual(representation.schema_version, "1.1")
        self.assertEqual(representation.speech_analysis.status, "not_started")
        self.assertEqual(representation.speech_analysis.speakers, [])
        self.assertEqual(representation.speech_analysis.speech_regions, [])
        self.assertEqual(representation.speech_analysis.word_alignments, [])
        self.assertEqual(representation.speech_analysis.phoneme_alignments, [])
        self.assertEqual(representation.speech_analysis.prosody, [])

    def test_nested_task_two_models_are_constructed(self) -> None:
        representation = minimal_representation(
            speech_analysis=SpeechAnalysis(
                status="complete",
                speakers=[SpeakerProfile(speaker_id="speaker_1", embedding_dimension=192)],
                speech_regions=[SpeechRegion(
                    id="region_1", start_seconds=0.5, end_seconds=1.2, speaker_id="speaker_1",
                    source="source_audio", content_item_ids=["content_1"], narration_item_ids=["narration_1"],
                )],
                word_alignments=[WordAlignment(
                    id="word_1", word="hello", start_seconds=0.5, end_seconds=0.8, confidence=0.98,
                    speaker_id="speaker_1", speech_region_id="region_1", content_item_id="content_1",
                    narration_item_id="narration_1", segment_index=1, alignment_method="asr",
                )],
                phoneme_alignments=[PhonemeAlignment(
                    id="phoneme_1", phoneme="HH", start_seconds=0.5, end_seconds=0.55,
                    word_alignment_id="word_1", speaker_id="speaker_1", alignment_method="forced_alignment",
                )],
                prosody=[ProsodyFrame(
                    start_seconds=0.5, end_seconds=0.6, speaker_id="speaker_1", f0_hz=180.0,
                    energy=0.65, speaking_rate_wpm=145.0, speech_region_id="region_1",
                )],
            )
        )

        analysis = representation.speech_analysis
        self.assertIsInstance(analysis.speakers[0], SpeakerProfile)
        self.assertIsInstance(analysis.speech_regions[0], SpeechRegion)
        self.assertIsInstance(analysis.word_alignments[0], WordAlignment)
        self.assertIsInstance(analysis.phoneme_alignments[0], PhonemeAlignment)
        self.assertIsInstance(analysis.prosody[0], ProsodyFrame)
        self.assertEqual(analysis.word_alignments[0].speech_region_id, "region_1")
        self.assertEqual(analysis.word_alignments[0].speaker_id, "speaker_1")
        self.assertEqual(analysis.phoneme_alignments[0].word_alignment_id, "word_1")
        self.assertEqual(analysis.prosody[0].speech_region_id, "region_1")

    def test_validation_constraints(self) -> None:
        with self.assertRaises(ValidationError):
            SpeechRegion(id="region_1", start_seconds=-1, end_seconds=1.0)
        with self.assertRaises(ValidationError):
            SpeechRegion(id="region_1", start_seconds=1.0, end_seconds=0.5)
        with self.assertRaises(ValidationError):
            WordAlignment(id="word_1", word="hello", start_seconds=1.0, end_seconds=0.5)
        with self.assertRaises(ValidationError):
            PhonemeAlignment(id="phoneme_1", phoneme="HH", start_seconds=1.0, end_seconds=0.5)
        with self.assertRaises(ValidationError):
            ProsodyFrame(start_seconds=1.0, end_seconds=0.5)
        with self.assertRaises(ValidationError):
            SpeechRegion(id="region_1", start_seconds=0.0, end_seconds=1.0, confidence=-0.1)
        with self.assertRaises(ValidationError):
            WordAlignment(id="word_1", word="hello", start_seconds=0.0, end_seconds=1.0, confidence=1.1)
        with self.assertRaises(ValidationError):
            SpeakerProfile(speaker_id="speaker_1", embedding_dimension=0)
        with self.assertRaises(ValidationError):
            SpeechRegion(id="region_1", start_seconds=0.0, end_seconds=1.0, source="microphone")
        with self.assertRaises(ValidationError):
            WordAlignment(
                id="word_1", word="hello", start_seconds=0.0, end_seconds=1.0, alignment_method="manual"
            )

    def test_json_round_trip_preserves_speech_analysis(self) -> None:
        representation = minimal_representation(
            speech_analysis=SpeechAnalysis(
                speakers=[SpeakerProfile(speaker_id="speaker_1")],
                speech_regions=[SpeechRegion(id="region_1", start_seconds=0.5, end_seconds=1.2)],
                word_alignments=[WordAlignment(
                    id="word_1", word="hello", start_seconds=0.5, end_seconds=0.8, speech_region_id="region_1"
                )],
            )
        )

        restored = NormalizedRepresentation.model_validate_json(representation.model_dump_json())

        self.assertEqual(restored.schema_version, "1.1")
        self.assertEqual(restored.speech_analysis.speakers[0].speaker_id, "speaker_1")
        self.assertEqual(restored.speech_analysis.word_alignments[0].word, "hello")


if __name__ == "__main__":
    unittest.main()
