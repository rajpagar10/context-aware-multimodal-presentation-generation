from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

from .errors import DependencyUnavailableError, PipelineError


class MediaTools:
    def __init__(self, ffmpeg_bin: str | None = None, ffprobe_bin: str | None = None):
        self.ffmpeg_bin = ffmpeg_bin or shutil.which("ffmpeg")
        self.ffprobe_bin = ffprobe_bin or shutil.which("ffprobe")

    def _require_ffprobe(self) -> str:
        if not self.ffprobe_bin:
            raise DependencyUnavailableError("ffprobe was not found on PATH. Install FFmpeg to inspect compressed audio/video files.", stage="media_metadata")
        return self.ffprobe_bin

    def probe(self, path: Path) -> dict[str, Any]:
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as source:
                    frames = source.getnframes()
                    rate = source.getframerate()
                    return {
                        "format": {"format_name": "wav", "duration": frames / rate if rate else None, "size": str(path.stat().st_size)},
                        "streams": [{"codec_type": "audio", "codec_name": "pcm", "sample_rate": str(rate), "channels": source.getnchannels(), "bits_per_sample": source.getsampwidth() * 8}],
                    }
            except wave.Error as exc:
                raise PipelineError(f"Invalid WAV file: {exc}", stage="media_metadata", path=str(path), modality="audio") from exc

        command = [self._require_ffprobe(), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise PipelineError(f"ffprobe could not read this media file: {result.stderr.strip() or 'unknown media error'}", stage="media_metadata", path=str(path))
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise PipelineError("ffprobe returned invalid metadata JSON.", stage="media_metadata", path=str(path)) from exc

    def extract_audio(self, video_path: Path, destination: Path) -> Path:
        if not self.ffmpeg_bin:
            raise DependencyUnavailableError("ffmpeg was not found on PATH. Install FFmpeg to extract a video's audio track.", stage="audio_extraction", path=str(video_path), modality="video")
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [self.ffmpeg_bin, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
            raise PipelineError(f"FFmpeg could not extract an audio stream: {result.stderr.strip() or 'no audio stream found'}", stage="audio_extraction", path=str(video_path), modality="video")
        return destination
