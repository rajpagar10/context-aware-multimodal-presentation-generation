"""Alignment abstractions that turn existing ASR evidence into schema models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .errors import PipelineError
from .models import WordAlignment
from .transcription import TranscriptionResult


class WordAligner(ABC):
    """Replaceable interface for converting transcription evidence to words."""

    @abstractmethod
    def align(self, transcription: TranscriptionResult, *, speaker_id: str | None = None) -> list[WordAlignment]:
        """Return word-level timing derived from an existing transcription result."""


class FasterWhisperWordAligner(WordAligner):
    """Map Faster-Whisper word timestamps into Task 2 ``WordAlignment`` objects.

    This adapter does not perform ASR or forced alignment. It preserves the
    timestamps and optional ``probability`` values already present in
    ``TranscriptSegment.words``.
    """

    def align(self, transcription: TranscriptionResult, *, speaker_id: str | None = None) -> list[WordAlignment]:
        alignments: list[WordAlignment] = []

        for segment in transcription.segments:
            if segment.index < 1:
                raise PipelineError(
                    f"ASR segment index must be at least 1; received {segment.index}.",
                    stage="alignment",
                    modality="audio",
                )
            for word_data in segment.words:
                word, start_seconds, end_seconds = self._required_word_fields(word_data, segment.index)
                confidence = word_data.get("probability")
                if confidence is not None and not self._is_number(confidence):
                    raise PipelineError(
                        f"ASR word '{word}' in segment {segment.index} has a non-numeric probability.",
                        stage="alignment",
                        modality="audio",
                    )
                self._validate_timing(word, start_seconds, end_seconds, segment.index)

                alignments.append(WordAlignment(
                    id=f"word_{len(alignments) + 1:06d}",
                    word=word,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    confidence=confidence,
                    speaker_id=speaker_id,
                    segment_index=segment.index,
                    alignment_method="asr",
                ))

        return alignments

    @staticmethod
    def _required_word_fields(word_data: dict[str, object], segment_index: int) -> tuple[str, float | int, float | int]:
        word = word_data.get("word")
        start_seconds = word_data.get("start_seconds")
        end_seconds = word_data.get("end_seconds")
        if not isinstance(word, str) or not word.strip():
            raise PipelineError(
                f"ASR segment {segment_index} contains a word timestamp without word text.",
                stage="alignment",
                modality="audio",
            )
        if not FasterWhisperWordAligner._is_number(start_seconds) or not FasterWhisperWordAligner._is_number(end_seconds):
            raise PipelineError(
                f"ASR word '{word}' in segment {segment_index} has missing or non-numeric timing.",
                stage="alignment",
                modality="audio",
            )
        return word, start_seconds, end_seconds

    @staticmethod
    def _validate_timing(word: str, start_seconds: float | int, end_seconds: float | int, segment_index: int) -> None:
        if start_seconds < 0 or end_seconds < 0 or start_seconds > end_seconds:
            raise PipelineError(
                f"ASR word '{word}' in segment {segment_index} has invalid timing: "
                f"start={start_seconds}, end={end_seconds}.",
                stage="alignment",
                modality="audio",
            )

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
