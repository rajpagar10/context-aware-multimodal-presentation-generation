from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .errors import TranscriptionUnavailableError


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    text: str
    start_seconds: float
    end_seconds: float
    words: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str | None
    duration_seconds: float | None
    segments: list[TranscriptSegment]
    engine: str


class Transcriber(Protocol):
    def transcribe(self, media_path: Path) -> TranscriptionResult: ...


class FasterWhisperTranscriber:
    """Lazy adapter: importing the package/model is deferred until ASR is used."""

    def __init__(self, model_name: str = "base", device: str = "auto", compute_type: str = "int8"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionUnavailableError(
                "Local ASR requires faster-whisper. Install requirements-asr.txt and retry.",
                stage="transcription",
            ) from exc
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, media_path: Path) -> TranscriptionResult:
        model = self._load_model()
        try:
            segments_iter, info = model.transcribe(str(media_path), word_timestamps=True, vad_filter=True)
            segments: list[TranscriptSegment] = []
            for index, segment in enumerate(segments_iter, start=1):
                words = [
                    {"word": word.word, "start_seconds": word.start, "end_seconds": word.end, "probability": word.probability}
                    for word in (segment.words or [])
                ]
                segments.append(TranscriptSegment(index, segment.text.strip(), float(segment.start), float(segment.end), words))
            return TranscriptionResult(getattr(info, "language", None), getattr(info, "duration", None), segments, "faster-whisper")
        except TranscriptionUnavailableError:
            raise
        except Exception as exc:
            raise TranscriptionUnavailableError(
                f"ASR failed for '{media_path.name}'. Verify the media is decodable and the model runtime is available: {exc}",
                stage="transcription",
                path=str(media_path),
            ) from exc
