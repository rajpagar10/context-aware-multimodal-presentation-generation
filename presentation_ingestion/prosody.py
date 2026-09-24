"""Frame-level baseline prosody extraction from local PCM WAV audio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import wave

from .errors import DependencyUnavailableError, PipelineError
from .models import ProsodyFrame, WordAlignment


class ProsodyExtractor(ABC):
    """Replaceable interface for deriving temporal prosody evidence from audio."""

    @abstractmethod
    def extract(
        self,
        audio: str | Path,
        *,
        word_alignments: list[WordAlignment] | None = None,
        speaker_id: str | None = None,
        speech_region_id: str | None = None,
    ) -> list[ProsodyFrame]:
        """Return frame-level prosody measurements from audio."""


class NumpyProsodyExtractor(ProsodyExtractor):
    """CPU baseline using PCM RMS energy and normalized autocorrelation F0.

    Frame timestamps are defined by the source-sample offsets; overlapping
    frames use ``frame_duration_seconds`` and advance by
    ``hop_duration_seconds``. ``energy`` is RMS amplitude after PCM conversion
    to approximately [-1, 1]. F0 is reported in Hz only for frames whose RMS
    and autocorrelation meet the configured voicing thresholds; unvoiced frames
    have ``f0_hz=None`` and ``voiced=False``. Supplied word alignments produce a
    WPM value from their union of spoken intervals. Gaps above
    ``pause_threshold_seconds`` are annotated on overlapping frames as inferred
    alignment pauses; no pause is inserted or synthesized.
    """

    def __init__(
        self,
        *,
        frame_duration_seconds: float = 0.03,
        hop_duration_seconds: float = 0.01,
        f0_min_hz: float = 50.0,
        f0_max_hz: float = 400.0,
        minimum_rms: float = 0.001,
        minimum_correlation: float = 0.3,
        pause_threshold_seconds: float = 0.25,
    ):
        if frame_duration_seconds <= 0 or hop_duration_seconds <= 0:
            raise ValueError("Frame duration and hop duration must be positive.")
        if f0_min_hz <= 0 or f0_max_hz <= f0_min_hz:
            raise ValueError("F0 bounds must be positive and f0_max_hz must exceed f0_min_hz.")
        if minimum_rms < 0 or not 0 <= minimum_correlation <= 1 or pause_threshold_seconds < 0:
            raise ValueError("Prosody thresholds must be non-negative; correlation must be within [0, 1].")
        self.frame_duration_seconds = frame_duration_seconds
        self.hop_duration_seconds = hop_duration_seconds
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.minimum_rms = minimum_rms
        self.minimum_correlation = minimum_correlation
        self.pause_threshold_seconds = pause_threshold_seconds

    def extract(
        self,
        audio: str | Path,
        *,
        word_alignments: list[WordAlignment] | None = None,
        speaker_id: str | None = None,
        speech_region_id: str | None = None,
    ) -> list[ProsodyFrame]:
        """Extract baseline prosody from a readable PCM WAV file."""
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise PipelineError(
                f"Prosody input is not a readable audio file: '{audio_path}'.",
                stage="prosody",
                path=str(audio_path),
                modality="audio",
            )

        numpy = self._numpy()
        samples, sample_rate = self._read_pcm_wav(audio_path, numpy)
        if not len(samples):
            return []

        frame_samples = max(1, round(self.frame_duration_seconds * sample_rate))
        hop_samples = max(1, round(self.hop_duration_seconds * sample_rate))
        speaking_rate_wpm = self._speaking_rate(word_alignments)
        pauses = self._pause_intervals(word_alignments)
        frames: list[ProsodyFrame] = []

        for start_sample in range(0, len(samples), hop_samples):
            frame = samples[start_sample:start_sample + frame_samples]
            if not len(frame):
                continue
            end_sample = start_sample + len(frame)
            start_seconds = start_sample / sample_rate
            end_seconds = end_sample / sample_rate
            energy = float(numpy.sqrt(numpy.mean(numpy.square(frame))))
            f0_hz = self._f0(frame, sample_rate, energy, numpy)
            metadata: dict[str, Any] = {
                "energy_representation": "rms_pcm_normalized",
                "f0_method": "normalized_autocorrelation",
            }
            overlapping_pauses = [pause for pause in pauses if start_seconds < pause[1] and end_seconds > pause[0]]
            if overlapping_pauses:
                metadata["aligned_pause"] = True
                metadata["pause_source"] = "word_alignment_gap"

            frames.append(ProsodyFrame(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                speaker_id=speaker_id,
                f0_hz=f0_hz,
                energy=energy,
                speaking_rate_wpm=speaking_rate_wpm,
                voiced=f0_hz is not None,
                speech_region_id=speech_region_id,
                metadata=metadata,
            ))
        return frames

    @staticmethod
    def _numpy():
        try:
            import numpy
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Prosody extraction requires NumPy. Install requirements-prosody.txt and retry.",
                stage="prosody",
                modality="audio",
            ) from exc
        return numpy

    @staticmethod
    def _read_pcm_wav(audio_path: Path, numpy):
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                compression = wav_file.getcomptype()
                raw_audio = wav_file.readframes(frame_count)
        except (wave.Error, OSError) as exc:
            raise PipelineError(
                f"Prosody extraction could not decode WAV audio: {exc}",
                stage="prosody",
                path=str(audio_path),
                modality="audio",
            ) from exc

        if channels < 1 or sample_rate <= 0 or sample_width not in {1, 2, 3, 4} or compression != "NONE":
            raise PipelineError(
                "Prosody extraction requires uncompressed PCM WAV audio with a valid sample rate.",
                stage="prosody",
                path=str(audio_path),
                modality="audio",
            )

        if sample_width == 1:
            samples = (numpy.frombuffer(raw_audio, dtype=numpy.uint8).astype(numpy.float64) - 128.0) / 128.0
        elif sample_width == 2:
            samples = numpy.frombuffer(raw_audio, dtype="<i2").astype(numpy.float64) / 32768.0
        elif sample_width == 4:
            samples = numpy.frombuffer(raw_audio, dtype="<i4").astype(numpy.float64) / 2147483648.0
        else:
            encoded = numpy.frombuffer(raw_audio, dtype=numpy.uint8).reshape(-1, 3)
            signed = encoded[:, 0].astype(numpy.int32) | (encoded[:, 1].astype(numpy.int32) << 8) | (encoded[:, 2].astype(numpy.int32) << 16)
            signed[signed >= 0x800000] -= 0x1000000
            samples = signed.astype(numpy.float64) / 8388608.0

        if len(samples) % channels:
            raise PipelineError("PCM WAV samples do not match its channel count.", stage="prosody", path=str(audio_path), modality="audio")
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return samples, sample_rate

    def _f0(self, frame, sample_rate: int, energy: float, numpy) -> float | None:
        if energy < self.minimum_rms or len(frame) < 3:
            return None
        centered = frame - numpy.mean(frame)
        autocorrelation = numpy.correlate(centered, centered, mode="full")[len(centered) - 1:]
        zero_lag = autocorrelation[0]
        minimum_lag = max(1, round(sample_rate / self.f0_max_hz))
        maximum_lag = min(len(centered) - 1, round(sample_rate / self.f0_min_hz))
        if zero_lag <= 0 or maximum_lag < minimum_lag:
            return None
        candidates = autocorrelation[minimum_lag:maximum_lag + 1]
        lag = minimum_lag + int(numpy.argmax(candidates))
        if autocorrelation[lag] / zero_lag < self.minimum_correlation:
            return None
        return float(sample_rate / lag)

    @staticmethod
    def _speaking_rate(word_alignments: list[WordAlignment] | None) -> float | None:
        if not word_alignments:
            return None
        intervals = sorted((word.start_seconds, word.end_seconds) for word in word_alignments)
        merged: list[list[float]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        spoken_duration_seconds = sum(end - start for start, end in merged)
        return None if spoken_duration_seconds == 0 else len(word_alignments) / (spoken_duration_seconds / 60)

    def _pause_intervals(self, word_alignments: list[WordAlignment] | None) -> list[tuple[float, float]]:
        if not word_alignments:
            return []
        ordered_words = sorted(word_alignments, key=lambda word: (word.start_seconds, word.end_seconds))
        pauses: list[tuple[float, float]] = []
        previous_end = ordered_words[0].end_seconds
        for word in ordered_words[1:]:
            if word.start_seconds - previous_end >= self.pause_threshold_seconds:
                pauses.append((previous_end, word.start_seconds))
            previous_end = max(previous_end, word.end_seconds)
        return pauses
