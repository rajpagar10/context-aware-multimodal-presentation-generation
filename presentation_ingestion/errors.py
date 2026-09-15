from __future__ import annotations


class PipelineError(Exception):
    """An expected, user-actionable pipeline failure."""

    def __init__(self, message: str, *, stage: str, path: str | None = None, modality: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.path = path
        self.modality = modality

    def as_dict(self) -> dict[str, str | None]:
        return {"message": str(self), "stage": self.stage, "path": self.path, "modality": self.modality}


class UnsupportedInputError(PipelineError):
    pass


class DependencyUnavailableError(PipelineError):
    pass


class TranscriptionUnavailableError(PipelineError):
    pass
