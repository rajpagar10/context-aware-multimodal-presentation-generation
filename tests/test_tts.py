from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import wave

from presentation_ingestion.errors import PipelineError
from presentation_ingestion.tts import PiperTTSEngine, TTSEngine, TTSResult


class FakeVoice:
    def synthesize(self, text: str, wav_file: wave.Wave_write) -> None:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


class FakeTTSEngine(TTSEngine):
    def synthesize(self, text: str, output_path: str | Path) -> TTSResult:
        return TTSResult(Path(output_path), 16000, 0.01, {"backend": "fake"})


class TTSTests(unittest.TestCase):
    def _engine(self, model_path: Path, loaded_devices: list[bool] | None = None) -> PiperTTSEngine:
        def load_voice(_: Path, __: Path | None, use_cuda: bool) -> FakeVoice:
            if loaded_devices is not None:
                loaded_devices.append(use_cuda)
            return FakeVoice()

        return PiperTTSEngine(model_path, voice_loader=load_voice)

    def test_abstraction_is_replaceable(self) -> None:
        result = FakeTTSEngine().synthesize("Hello", "output.wav")

        self.assertIsInstance(FakeTTSEngine(), TTSEngine)
        self.assertEqual(result.output_path, Path("output.wav"))
        self.assertEqual(result.metadata["backend"], "fake")

    def test_piper_writes_and_validates_wav_with_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "voice.onnx"
            model_path.touch()
            output_path = root / "nested" / "baseline.wav"

            result = self._engine(model_path).synthesize("Hello, world.", output_path)

            self.assertEqual(result.output_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(result.sample_rate, 16000)
            self.assertGreater(result.duration_seconds, 0)
            self.assertEqual(result.metadata["backend"], "piper-tts")
            self.assertEqual(result.metadata["model"], "voice.onnx")
            self.assertEqual(result.metadata["device"], "cpu")
            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16000)
                self.assertGreater(wav_file.getnframes(), 0)

    def test_empty_text_and_non_wav_output_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "voice.onnx"
            model_path.touch()
            engine = self._engine(model_path)

            with self.assertRaises(PipelineError):
                engine.synthesize("   ", root / "baseline.wav")
            with self.assertRaises(PipelineError):
                engine.synthesize("Hello", root / "baseline.mp3")

    def test_cpu_fallback_and_lazy_model_loading(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_path = root / "voice.onnx"
            model_path.touch()
            loaded_devices: list[bool] = []
            with patch.object(PiperTTSEngine, "_default_use_cuda", return_value=False):
                engine = self._engine(model_path, loaded_devices)
                self.assertEqual(loaded_devices, [])
                engine.synthesize("Hello", root / "baseline.wav")

            self.assertEqual(loaded_devices, [False])


if __name__ == "__main__":
    unittest.main()
