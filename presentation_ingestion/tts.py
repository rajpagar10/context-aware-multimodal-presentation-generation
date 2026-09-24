"""Replaceable local text-to-speech engines for the Task 2 baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import wave

from .errors import DependencyUnavailableError, PipelineError


@dataclass(frozen=True)
class TTSResult:
    """A validated WAV artifact created by a TTS engine."""

    output_path: Path
    sample_rate: int
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TTSEngine(ABC):
    """Replaceable interface for text-to-WAV synthesis."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str | Path) -> TTSResult:
        """Synthesize non-empty text to a valid WAV file."""


class PiperTTSEngine(TTSEngine):
    """Lazy Piper TTS adapter for one locally available voice model.

    ``model_path`` must point to a Piper ``.onnx`` voice model (with its
    corresponding ``.onnx.json`` configuration available to Piper). Piper
    writes PCM WAV directly; its model and runtime are never loaded on module
    import or engine construction. CUDA is used only when an ONNX Runtime CUDA
    provider is available; otherwise the backend uses CPU.
    """

    backend_name = "piper-tts"

    def __init__(
        self,
        model_path: str | Path,
        *,
        config_path: str | Path | None = None,
        use_cuda: bool | None = None,
        voice_loader: Callable[[Path, Path | None, bool], object] | None = None,
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path is not None else None
        self.use_cuda = use_cuda
        self._voice_loader = voice_loader or self._load_piper_voice
        self._voice: object | None = None
        self._active_use_cuda: bool | None = None

    def synthesize(self, text: str, output_path: str | Path) -> TTSResult:
        """Synthesize text through Piper and validate its non-empty WAV output."""
        if not isinstance(text, str) or not text.strip():
            raise PipelineError("TTS requires non-empty text.", stage="tts", modality="audio")

        destination = Path(output_path)
        if destination.suffix.lower() != ".wav":
            raise PipelineError("TTS output_path must use the .wav extension.", stage="tts", path=str(destination), modality="audio")
        if destination.exists() and destination.is_dir():
            raise PipelineError("TTS output_path must be a file, not a directory.", stage="tts", path=str(destination), modality="audio")
        destination.parent.mkdir(parents=True, exist_ok=True)

        voice = self._get_voice()
        try:
            with wave.open(str(destination), "wb") as wav_file:
                voice.synthesize(text.strip(), wav_file)  # type: ignore[attr-defined]
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Piper could not synthesize speech: {exc}",
                stage="tts",
                path=str(destination),
                modality="audio",
            ) from exc

        try:
            with wave.open(str(destination), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                if sample_rate <= 0 or frame_count <= 0:
                    raise ValueError("WAV has no valid audio frames")
                duration_seconds = frame_count / sample_rate
        except (wave.Error, OSError, ValueError) as exc:
            raise PipelineError(
                f"Piper did not produce a valid non-empty WAV file: {exc}",
                stage="tts",
                path=str(destination),
                modality="audio",
            ) from exc

        return TTSResult(
            output_path=destination,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            metadata={
                "backend": self.backend_name,
                "model": self.model_path.name,
                "device": "cuda" if self._active_use_cuda else "cpu",
            },
        )

    def _get_voice(self) -> object:
        if self._voice is not None:
            return self._voice
        if not self.model_path.is_file():
            raise PipelineError(
                f"Piper voice model was not found: '{self.model_path}'.",
                stage="tts",
                path=str(self.model_path),
                modality="audio",
            )

        self._active_use_cuda = self.use_cuda if self.use_cuda is not None else self._default_use_cuda()
        try:
            self._voice = self._voice_loader(self.model_path, self.config_path, self._active_use_cuda)
        except DependencyUnavailableError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Could not load Piper voice model '{self.model_path.name}': {exc}",
                stage="tts",
                path=str(self.model_path),
                modality="audio",
            ) from exc
        return self._voice

    @staticmethod
    def _load_piper_voice(model_path: Path, config_path: Path | None, use_cuda: bool) -> object:
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Baseline TTS requires Piper. Install requirements-tts.txt and provide a local Piper voice model.",
                stage="tts",
                modality="audio",
            ) from exc
        return PiperVoice.load(model_path, config_path=config_path, use_cuda=use_cuda)

    @staticmethod
    def _default_use_cuda() -> bool:
        try:
            import onnxruntime
        except ImportError:
            return False
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
