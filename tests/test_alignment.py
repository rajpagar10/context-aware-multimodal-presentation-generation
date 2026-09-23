from __future__ import annotations

import unittest

from presentation_ingestion.alignment import FasterWhisperWordAligner, WordAligner
from presentation_ingestion.errors import PipelineError
from presentation_ingestion.transcription import TranscriptSegment, TranscriptionResult


def transcription_with_words() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        duration_seconds=2.0,
        engine="faster-whisper",
        segments=[
            TranscriptSegment(
                index=1,
                text="Artificial intelligence",
                start_seconds=0.0,
                end_seconds=1.0,
                words=[
                    {"word": "Artificial", "start_seconds": 0.0, "end_seconds": 0.4, "probability": 0.98},
                    {"word": "intelligence", "start_seconds": 0.4, "end_seconds": 1.0},
                ],
            ),
            TranscriptSegment(
                index=2,
                text="is useful",
                start_seconds=1.0,
                end_seconds=2.0,
                words=[
                    {"word": "is", "start_seconds": 1.0, "end_seconds": 1.2, "probability": 0.95},
                    {"word": "useful", "start_seconds": 1.2, "end_seconds": 2.0, "probability": 0.91},
                ],
            ),
        ],
    )


class AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = FasterWhisperWordAligner()

    def test_basic_word_conversion(self) -> None:
        alignments = self.aligner.align(transcription_with_words())

        self.assertIsInstance(self.aligner, WordAligner)
        self.assertEqual([alignment.word for alignment in alignments], ["Artificial", "intelligence", "is", "useful"])
        self.assertEqual((alignments[0].start_seconds, alignments[0].end_seconds), (0.0, 0.4))
        self.assertTrue(all(alignment.alignment_method == "asr" for alignment in alignments))

    def test_ids_are_unique_and_deterministic(self) -> None:
        first = self.aligner.align(transcription_with_words())
        second = self.aligner.align(transcription_with_words())

        self.assertEqual([alignment.id for alignment in first], ["word_000001", "word_000002", "word_000003", "word_000004"])
        self.assertEqual([alignment.id for alignment in first], [alignment.id for alignment in second])

    def test_segment_indexes_and_optional_confidence_are_preserved(self) -> None:
        alignments = self.aligner.align(transcription_with_words())

        self.assertEqual([alignment.segment_index for alignment in alignments], [1, 1, 2, 2])
        self.assertEqual(alignments[0].confidence, 0.98)
        self.assertIsNone(alignments[1].confidence)

    def test_empty_transcription_and_segments_without_words_are_empty(self) -> None:
        empty = TranscriptionResult(language="en", duration_seconds=0.0, segments=[], engine="faster-whisper")
        no_words = TranscriptionResult(
            language="en",
            duration_seconds=1.0,
            engine="faster-whisper",
            segments=[TranscriptSegment(1, "speech", 0.0, 1.0)],
        )

        self.assertEqual(self.aligner.align(empty), [])
        self.assertEqual(self.aligner.align(no_words), [])

    def test_no_speaker_is_fabricated(self) -> None:
        self.assertTrue(all(alignment.speaker_id is None for alignment in self.aligner.align(transcription_with_words())))

    def test_invalid_timing_is_rejected(self) -> None:
        for word in (
            {"word": "bad", "start_seconds": -0.1, "end_seconds": 0.2},
            {"word": "bad", "start_seconds": 0.3, "end_seconds": 0.2},
        ):
            transcription = TranscriptionResult(
                language="en",
                duration_seconds=1.0,
                engine="faster-whisper",
                segments=[TranscriptSegment(1, "bad", 0.0, 1.0, [word])],
            )
            with self.assertRaises(PipelineError):
                self.aligner.align(transcription)


if __name__ == "__main__":
    unittest.main()
