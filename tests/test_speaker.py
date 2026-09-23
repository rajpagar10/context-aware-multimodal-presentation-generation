from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from presentation_ingestion.speaker import SpeakerEmbedding, SpeakerEncoder, SpeechBrainECAPASpeakerEncoder


class FakeSpeakerEncoder(SpeakerEncoder):
    def encode(self, audio: object) -> SpeakerEmbedding:
        return SpeakerEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_name="test-encoder",
            dimension=3,
            metadata={"audio_reference": audio},
        )


class FakeTensor:
    def __init__(self, values: list[list[list[float]]]):
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def flatten(self) -> FakeTensor:
        return self

    def tolist(self) -> list[list[list[float]]]:
        return self.values


class FakeSpeechBrainModel:
    def encode_file(self, audio_path: str) -> FakeTensor:
        return FakeTensor([[[0.1, 0.2, 0.3]]])


class SpeakerAbstractionTests(unittest.TestCase):
    def test_speaker_embedding_preserves_values(self) -> None:
        embedding = SpeakerEmbedding(
            vector=[0.1, 0.2, 0.3],
            model_name="test-encoder",
            dimension=3,
            metadata={"sample_rate": 16000},
        )

        self.assertEqual(embedding.vector, [0.1, 0.2, 0.3])
        self.assertEqual(embedding.model_name, "test-encoder")
        self.assertEqual(embedding.dimension, 3)
        self.assertEqual(embedding.metadata, {"sample_rate": 16000})

    def test_speaker_encoder_can_be_replaced(self) -> None:
        embedding = FakeSpeakerEncoder().encode("audio-fixture")

        self.assertIsInstance(embedding, SpeakerEmbedding)
        self.assertEqual(embedding.model_name, "test-encoder")
        self.assertEqual(embedding.metadata["audio_reference"], "audio-fixture")

    def test_base_encoder_cannot_fake_extraction(self) -> None:
        with self.assertRaises(TypeError):
            SpeakerEncoder()

    def test_speechbrain_encoder_implements_speaker_encoder(self) -> None:
        self.assertIsInstance(SpeechBrainECAPASpeakerEncoder(device="cpu", model_loader=lambda _: FakeSpeechBrainModel()), SpeakerEncoder)

    def test_speechbrain_output_contract_and_dimension(self) -> None:
        loaded_devices: list[str] = []

        def load_model(device: str) -> FakeSpeechBrainModel:
            loaded_devices.append(device)
            return FakeSpeechBrainModel()

        with TemporaryDirectory() as temporary_directory:
            audio_path = Path(temporary_directory) / "fixture.wav"
            audio_path.touch()
            embedding = SpeechBrainECAPASpeakerEncoder(device="cpu", model_loader=load_model).encode(audio_path)

        self.assertEqual(loaded_devices, ["cpu"])
        self.assertEqual(embedding.vector, [0.1, 0.2, 0.3])
        self.assertTrue(all(isinstance(value, float) for value in embedding.vector))
        self.assertEqual(embedding.model_name, "speechbrain/spkrec-ecapa-voxceleb")
        self.assertEqual(embedding.dimension, len(embedding.vector))
        self.assertEqual(embedding.metadata["backend"], "speechbrain")
        self.assertEqual(embedding.metadata["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
