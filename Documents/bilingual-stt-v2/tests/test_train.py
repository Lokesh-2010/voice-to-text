import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.dataset import SpeechSample, collate_speech
from src.model import CNNBiLSTM
from src.train import batch_loss, check_gradients, save_checkpoint, validate_batch
from src.vocabulary import Vocabulary


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CNNBiLSTM(num_classes=5)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.samples = [
            SpeechSample(torch.randn(80, 8), torch.tensor([2, 2, 3]), "short.wav"),
            SpeechSample(torch.randn(80, 13), torch.tensor([3, 4]), "long.wav"),
        ]

    def test_loss_ignores_padding_and_backward_is_finite(self):
        batch = collate_speech(self.samples)
        loss = batch_loss(self.model, batch, self.criterion)
        single_losses = [
            batch_loss(self.model, collate_speech([sample]), self.criterion)
            for sample in self.samples
        ]
        torch.testing.assert_close(loss, torch.stack(single_losses).mean())
        loss.backward()
        check_gradients(self.model)

    def test_rejects_bad_ids_and_insufficient_frames(self):
        batch = collate_speech(self.samples)
        with self.assertRaisesRegex(ValueError, "Invalid CTC target ID"):
            validate_batch(batch._replace(targets=torch.tensor([5, 2, 3, 3, 4])), 5)
        with self.assertRaisesRegex(ValueError, "requires 4 CTC frames"):
            validate_batch(batch._replace(feature_lengths=torch.tensor([3, 13])), 5)

    def test_checkpoint_creation_does_not_overwrite(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        vocabulary = Vocabulary(["a", "b", "c"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, 1, vocabulary)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, self.model, optimizer, 1, vocabulary)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epochs"], 1)
            self.assertEqual(checkpoint["num_classes"], 5)
            self.assertEqual(checkpoint["vocabulary_tokens"], vocabulary.id_to_char)
            with self.assertRaisesRegex(ValueError, "class count"):
                save_checkpoint(Path(directory) / "bad.pt", self.model, optimizer, 1, Vocabulary(["a"]))


if __name__ == "__main__":
    unittest.main()
