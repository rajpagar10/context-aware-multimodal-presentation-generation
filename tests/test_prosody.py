from __future__ import annotations

from array import array
from math import pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import wave

from presentation_ingestion.errors import PipelineError
from presentation_ingestion.models import WordAlignment
from presentation_ingestion.prosody import NumpyProsodyExtractor, ProsodyExtractor


def write_pcm_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    pcm = array("h", (max(-32768, min(32767, round(sample * 32767))) for sample in samples))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def sine_wave(frequency_hz: float, duration_seconds: float, sample_rate: int = 16000) -> list[float]:
    return [0.5 * sin(2 * pi * frequency_hz * index / sample_rate) for index in range(round(duration_seconds * sample_rate))]


class ProsodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = NumpyProsodyExtractor()

    def test_basic_extraction_f0_energy_and_temporal_order(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "tone.wav"
            write_pcm_wav(audio_path, sine_wave(200.0, 0.3))
            frames = self.extractor.extract(audio_path)

        self.assertIsInstance(self.extractor, ProsodyExtractor)
        self.assertTrue(frames)
        self.assertTrue(all(frame.start_seconds <= frame.end_seconds for frame in frames))
        self.assertEqual([frame.start_seconds for frame in frames], sorted(frame.start_seconds for frame in frames))
        self.assertTrue(all(frame.energy is not None and frame.energy >= 0 for frame in frames))
        voiced_f0 = [frame.f0_hz for frame in frames if frame.voiced]
        self.assertTrue(voiced_f0)
        self.assertTrue(all(180 <= f0 <= 220 for f0 in voiced_f0 if f0 is not None))

    def test_silence_has_energy_but_no_fabricated_f0(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "silence.wav"
            write_pcm_wav(audio_path, [0.0] * 1600)
            frames = self.extractor.extract(audio_path)

        self.assertTrue(frames)
        self.assertTrue(all(frame.energy == 0.0 for frame in frames))
        self.assertTrue(all(frame.f0_hz is None and frame.voiced is False for frame in frames))

    def test_speaking_rate_and_alignment_pauses(self) -> None:
        words = [
            WordAlignment(id="word_1", word="hello", start_seconds=0.0, end_seconds=0.5),
            WordAlignment(id="word_2", word="world", start_seconds=1.0, end_seconds=1.5),
        ]
        with TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "tone.wav"
            write_pcm_wav(audio_path, sine_wave(200.0, 1.6))
            frames = self.extractor.extract(audio_path, word_alignments=words)

        self.assertTrue(all(frame.speaking_rate_wpm == 120.0 for frame in frames))
        self.assertTrue(any(frame.metadata.get("aligned_pause") for frame in frames))

    def test_missing_alignments_do_not_fabricate_wpm_or_short_pauses(self) -> None:
        short_gap_words = [
            WordAlignment(id="word_1", word="hello", start_seconds=0.0, end_seconds=0.5),
            WordAlignment(id="word_2", word="world", start_seconds=0.55, end_seconds=1.0),
        ]
        with TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "tone.wav"
            write_pcm_wav(audio_path, sine_wave(200.0, 1.1))
            without_words = self.extractor.extract(audio_path)
            with_short_gap = self.extractor.extract(audio_path, word_alignments=short_gap_words)

        self.assertTrue(all(frame.speaking_rate_wpm is None for frame in without_words))
        self.assertFalse(any(frame.metadata.get("aligned_pause") for frame in with_short_gap))

    def test_missing_input_is_rejected(self) -> None:
        with self.assertRaises(PipelineError):
            self.extractor.extract("missing.wav")


if __name__ == "__main__":
    unittest.main()
