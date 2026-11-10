import csv
import tempfile
import unittest
from pathlib import Path

import librosa
import numpy as np
import torch

from src.audio_preprocessing import extract_log_mel
from src.dataset import SpeechDataset, SpeechSample, collate_speech, make_loader, minimum_ctc_frames
from src.vocabulary import UNKNOWN_ID, Vocabulary


class DatasetTests(unittest.TestCase):
    def test_preprocessing_matches_existing_librosa_calculation(self):
        waveform = np.random.default_rng(42).standard_normal(4096).astype(np.float32)
        expected = librosa.power_to_db(
            librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=80)
        )
        np.testing.assert_array_equal(extract_log_mel(waveform), expected)

    def test_padding_lengths_and_concatenated_targets(self):
        samples = [
            SpeechSample(torch.ones(80, 5), torch.tensor([2, 2, 3]), "first.wav"),
            SpeechSample(torch.full((80, 7), 2.0), torch.tensor([3, 4]), "second.wav"),
        ]
        batch = collate_speech(samples)
        self.assertEqual(tuple(batch.features.shape), (2, 80, 7))
        self.assertEqual(batch.feature_lengths.tolist(), [5, 7])
        self.assertEqual(batch.target_lengths.tolist(), [3, 2])
        self.assertEqual(batch.targets.tolist(), [2, 2, 3, 3, 4])
        self.assertTrue(torch.all(batch.features[0, :, 5:] == 0))
        self.assertEqual(minimum_ctc_frames(samples[0].targets), 4)
        self.assertEqual(batch.audio_paths, ("first.wav", "second.wav"))

    def test_repeated_characters_require_extra_frames(self):
        target = torch.tensor([2, 2, 2])
        self.assertEqual(minimum_ctc_frames(target), 5)
        with self.assertRaisesRegex(ValueError, "requires 5 frames; got 4"):
            collate_speech([SpeechSample(torch.ones(80, 4), target, "short.wav")])
        batch = collate_speech([SpeechSample(torch.ones(80, 5), target, "enough.wav")])
        self.assertEqual(batch.feature_lengths.tolist(), [5])
        with self.assertRaisesRegex(ValueError, "Invalid CTC target"):
            collate_speech([SpeechSample(torch.ones(80, 5), torch.tensor([0]), "blank.wav")])

    def test_real_dataset_splits_and_all_ctc_lengths(self):
        expected_counts = {"train": 28, "val": 2, "test": 2}
        for split, count in expected_counts.items():
            dataset = SpeechDataset(split)
            self.assertEqual(len(dataset), count)
            total = 0
            for batch in make_loader(dataset):
                self.assertEqual(batch.features.shape[0], len(batch.feature_lengths))
                self.assertEqual(batch.targets.numel(), batch.target_lengths.sum().item())
                offset = 0
                for length, target_length in zip(batch.feature_lengths, batch.target_lengths):
                    target = batch.targets[offset:offset + int(target_length)]
                    self.assertGreaterEqual(int(length), minimum_ctc_frames(target))
                    offset += int(target_length)
                total += len(batch.audio_paths)
            self.assertEqual(total, count)

    def test_split_selection_and_unknown_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab_path = root / "vocab.json"
            Vocabulary(["a"]).save(vocab_path)
            csv_path = root / "transcripts.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["audio_path", "transcript", "split"])
                writer.writerows([("train.wav", "a", "train"), ("val.wav", "z", "val")])
            dataset = SpeechDataset("val", csv_path, vocab_path, root)
            self.assertEqual(dataset.rows, [("val.wav", "z")])
            self.assertEqual(dataset.vocabulary.encode(dataset.rows[0][1]), [UNKNOWN_ID])
            with self.assertRaises(ValueError):
                SpeechDataset("test", csv_path, vocab_path, root)


if __name__ == "__main__":
    unittest.main()
