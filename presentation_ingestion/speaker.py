"""Model-agnostic speaker-embedding abstractions for future Task 2 work."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DependencyUnavailableError, PipelineError


@dataclass(frozen=True)
class SpeakerEmbedding:
    """The numerical output produced by a concrete speaker encoder."""

    vector: list[float]
    model_name: str
    dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeakerEncoder(ABC):
    """Replaceable interface for future model-specific speaker encoders."""

    @abstractmethod
    def encode(self, audio: object) -> SpeakerEmbedding:
        """Create a speaker embedding from an encoder-specific audio input."""


class SpeechBrainECAPASpeakerEncoder(SpeakerEncoder):
    """Lazy local ECAPA-TDNN speaker encoder using a readable audio-file path.

    The SpeechBrain ``spkrec-ecapa-voxceleb`` model produces 192-dimensional
    embeddings. SpeechBrain performs model-specific audio loading; the model is
    loaded only on the first ``encode`` call. CUDA is selected when PyTorch
    reports it as available, otherwise CPU is used.
    """

    model_name = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, device: str | None = None, *, model_loader: Callable[[str], object] | None = None):
        self.device = device
        self._model_loader = model_loader or self._load_speechbrain_model
        self._model: object | None = None
        self._active_device: str | None = None

    def encode(self, audio: object) -> SpeakerEmbedding:
        """Encode a readable local audio-file path into a stable Python vector."""
        if not isinstance(audio, (str, Path)):
            raise TypeError("SpeechBrainECAPASpeakerEncoder expects an audio file path.")

        audio_path = Path(audio)
        if not audio_path.is_file():
            raise PipelineError(
                f"Speaker embedding input is not a readable audio file: '{audio_path}'.",
                stage="speaker_embedding",
                path=str(audio_path),
                modality="audio",
            )

        model = self._get_model()
        try:
            output = model.encode_file(str(audio_path))  # type: ignore[attr-defined]
            vector = self._to_vector(output)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"SpeechBrain could not encode '{audio_path.name}': {exc}",
                stage="speaker_embedding",
                path=str(audio_path),
                modality="audio",
            ) from exc

        if not vector:
            raise PipelineError(
                "SpeechBrain returned an empty speaker embedding.",
                stage="speaker_embedding",
                path=str(audio_path),
                modality="audio",
            )

        return SpeakerEmbedding(
            vector=vector,
            model_name=self.model_name,
            dimension=len(vector),
            metadata={
                "backend": "speechbrain",
                "model_source": self.model_name,
                "device": self._active_device,
            },
        )

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model

        self._active_device = self.device or self._default_device()
        try:
            self._model = self._model_loader(self._active_device)
        except DependencyUnavailableError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Could not load SpeechBrain speaker model '{self.model_name}': {exc}",
                stage="speaker_embedding",
                modality="audio",
            ) from exc
        return self._model

    def _load_speechbrain_model(self, device: str) -> object:
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Speaker embedding requires SpeechBrain. Install requirements-speaker.txt and retry.",
                stage="speaker_embedding",
                modality="audio",
            ) from exc
        return EncoderClassifier.from_hparams(source=self.model_name, run_opts={"device": device})

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _to_vector(output: object) -> list[float]:
        tensor = output.detach().cpu() if hasattr(output, "detach") else output  # type: ignore[attr-defined]
        values = tensor.flatten().tolist() if hasattr(tensor, "flatten") else tensor  # type: ignore[attr-defined]
        return [float(value) for value in SpeechBrainECAPASpeakerEncoder._flatten(values)]

    @staticmethod
    def _flatten(values: object) -> list[object]:
        if isinstance(values, (list, tuple)):
            return [item for value in values for item in SpeechBrainECAPASpeakerEncoder._flatten(value)]
        return [values]
