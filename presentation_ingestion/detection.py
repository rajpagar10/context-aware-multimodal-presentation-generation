from __future__ import annotations

import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .errors import PipelineError, UnsupportedInputError
from .models import InputKind

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
SIMPLE_SUFFIXES = {".pptx": InputKind.PPTX, ".pdf": InputKind.PDF, ".txt": InputKind.TEXT, ".docx": InputKind.DOCX}


@dataclass(frozen=True)
class DetectedInput:
    path: Path
    kind: InputKind
    mime_type: str | None
    size_bytes: int


def _validate_signature(path: Path, kind: InputKind) -> None:
    try:
        header = path.read_bytes()[:8]
    except OSError as exc:
        raise PipelineError(f"Cannot read input file: {exc}", stage="validation", path=str(path), modality=kind.value) from exc

    if kind == InputKind.PDF and not header.startswith(b"%PDF-"):
        raise PipelineError("File extension is .pdf but the PDF signature is missing.", stage="validation", path=str(path), modality=kind.value)
    if kind in {InputKind.PPTX, InputKind.DOCX}:
        if not zipfile.is_zipfile(path):
            raise PipelineError("Office Open XML file is not a valid ZIP package.", stage="validation", path=str(path), modality=kind.value)
        expected = "ppt/presentation.xml" if kind == InputKind.PPTX else "word/document.xml"
        try:
            with zipfile.ZipFile(path) as package:
                if expected not in package.namelist():
                    raise PipelineError("Office file has the wrong internal document type.", stage="validation", path=str(path), modality=kind.value)
        except zipfile.BadZipFile as exc:
            raise PipelineError("Office Open XML package is corrupt.", stage="validation", path=str(path), modality=kind.value) from exc
    if kind == InputKind.TEXT:
        try:
            path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PipelineError("TXT input must be UTF-8 encoded.", stage="validation", path=str(path), modality=kind.value) from exc


def detect_input(path: str | Path, *, max_bytes: int = 1024 * 1024 * 1024) -> DetectedInput:
    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise PipelineError("Input file does not exist or is not a regular file.", stage="detection", path=str(source))
    size = source.stat().st_size
    if size == 0:
        raise PipelineError("Input file is empty.", stage="validation", path=str(source))
    if size > max_bytes:
        raise PipelineError(f"Input exceeds configured limit of {max_bytes} bytes.", stage="validation", path=str(source))

    suffix = source.suffix.lower()
    if suffix in SIMPLE_SUFFIXES:
        kind = SIMPLE_SUFFIXES[suffix]
    elif suffix in VIDEO_SUFFIXES:
        kind = InputKind.VIDEO
    elif suffix in AUDIO_SUFFIXES:
        kind = InputKind.AUDIO
    else:
        supported = ", ".join(sorted(set(SIMPLE_SUFFIXES) | VIDEO_SUFFIXES | AUDIO_SUFFIXES))
        raise UnsupportedInputError(f"Unsupported input format '{suffix or '(no extension)'}'. Supported extensions: {supported}", stage="detection", path=str(source))

    _validate_signature(source, kind)
    mime_type = mimetypes.guess_type(str(source))[0]
    return DetectedInput(path=source, kind=kind, mime_type=mime_type, size_bytes=size)
