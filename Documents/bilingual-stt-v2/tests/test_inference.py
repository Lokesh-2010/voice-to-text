import csv
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional

import src.inference as inference
from src.dataset import DEFAULT_CHECKPOINT_PATH, DEFAULT_VOCAB_PATH, SpeechDataset
from src.inference import character_error_rate, greedy_ctc_ids, load_model, test_recordings
from src.model import CNNBiLSTM
from src.train import save_checkpoint
from src.vocabulary import BLANK_ID, Vocabulary


class InferenceTests(unittest.TestCase):
    def test_v3_configuration_and_synthetic_forward(self):
        vocabulary = Vocabulary.load(DEFAULT_VOCAB_PATH)
        self.assertEqual(len(vocabulary), 75)
        self.assertEqual(vocabulary.char_to_id["<blank>"], BLANK_ID)
        self.assertEqual(BLANK_ID, 0)
        self.assertEqual(SpeechDataset.__init__.__defaults__[1], DEFAULT_VOCAB_PATH)
        self.assertEqual(inference.DEFAULT_VOCAB_PATH, DEFAULT_VOCAB_PATH)
        self.assertEqual(inference.DEFAULT_CHECKPOINT_PATH, DEFAULT_CHECKPOINT_PATH)
        self.assertEqual(DEFAULT_CHECKPOINT_PATH.name, "cnn_bilstm_ctc_v3.pt")
        model = CNNBiLSTM(num_classes=len(vocabulary)).eval()
        with torch.no_grad():
            output = model(torch.randn(2, 80, 17))
        self.assertEqual(tuple(output.shape), (2, 17, len(vocabulary)))
        self.assertTrue(torch.isfinite(output).all())

    def test_checkpoint_compatibility(self):
        vocabulary = Vocabulary(["a", "b", "c"])
        model = CNNBiLSTM(num_classes=len(vocabulary))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, 1, vocabulary)
            self.assertEqual(load_model(path, vocabulary).classifier.out_features, len(vocabulary))
            with self.assertRaisesRegex(ValueError, "class count"):
                load_model(path, Vocabulary(["a"]))
            with self.assertRaisesRegex(ValueError, "vocabulary mapping missing or different"):
                load_model(path, Vocabulary(["a", "b", "d"]))
            legacy = Path(directory) / "legacy.pt"
            torch.save({"num_classes": len(vocabulary), "model_state_dict": model.state_dict()}, legacy)
            with self.assertRaisesRegex(ValueError, "vocabulary mapping missing or different"):
                load_model(legacy, vocabulary)
            old_classes = Path(directory) / "old_classes.pt"
            torch.save({"num_classes": 60}, old_classes)
            with self.assertRaisesRegex(ValueError, "class count 60 does not match vocabulary size 75"):
                load_model(old_classes, Vocabulary.load(DEFAULT_VOCAB_PATH))

    def test_ctc_repeat_collapse_before_blank_removal(self):
        ids = torch.tensor([0, 2, 2, 0, 2, 3, 3, 0, 0])
        logits = functional.one_hot(ids, num_classes=4).float()
        self.assertEqual(greedy_ctc_ids(logits), [2, 2, 3])
        self.assertEqual(greedy_ctc_ids(functional.one_hot(torch.zeros(3, dtype=torch.long), 4).float()), [])

    def test_character_error_rate(self):
        self.assertEqual(character_error_rate("abc", "abc"), 0)
        self.assertEqual(character_error_rate("abc", ""), 1)
        self.assertAlmostEqual(character_error_rate("abc", "axcd"), 2 / 3)
        self.assertAlmostEqual(character_error_rate("నేను", "నేను"), 0)
        with self.assertRaises(ValueError):
            character_error_rate("", "x")

    def test_selects_only_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcripts.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["audio_path", "language", "transcript", "split"])
                writer.writerows([
                    ("train.wav", "english", "training", "train"),
                    ("english.wav", "english", "English test", "test"),
                    ("telugu.wav", "telugu", "తెలుగు", "test"),
                ])
            self.assertEqual(test_recordings(path), [
                ("english", "english.wav", "English test"),
                ("telugu", "telugu.wav", "తెలుగు"),
            ])


if __name__ == "__main__":
    unittest.main()
