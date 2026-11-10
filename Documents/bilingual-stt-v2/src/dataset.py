"""Prepare padded speech batches and concatenated CTC targets."""

import csv
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .audio_preprocessing import N_MELS, extract_log_mel, load_audio
from .vocabulary import BLANK_ID, Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "models" / "vocab_v3.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "cnn_bilstm_ctc_v3.pt"
SPLITS = ("train", "val", "test")


class SpeechSample(NamedTuple):
    features: Tensor
    targets: Tensor
    audio_path: str


class SpeechBatch(NamedTuple):
    features: Tensor
    targets: Tensor
    feature_lengths: Tensor
    target_lengths: Tensor
    audio_paths: tuple[str, ...]


def minimum_ctc_frames(targets: Tensor) -> int:
    """CTC needs a blank between each pair of identical adjacent IDs."""
    return targets.numel() + int((targets[1:] == targets[:-1]).sum().item())


class SpeechDataset(Dataset[SpeechSample]):
    def __init__(
        self,
        split: str,
        csv_path: Path = PROJECT_ROOT / "data" / "transcripts.csv",
        vocab_path: Path = DEFAULT_VOCAB_PATH,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        self.project_root = project_root
        self.vocabulary = Vocabulary.load(vocab_path)
        self.rows: list[tuple[str, str]] = []
        with csv_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not {"audio_path", "transcript", "split"} <= set(reader.fieldnames):
                raise ValueError("CSV must contain audio_path, transcript, and split columns")
            for row in reader:
                if row["split"] == split:
                    audio_path = row["audio_path"]
                    transcript = row["transcript"]
                    if not audio_path or not transcript or not transcript.strip():
                        raise ValueError(f"Missing audio path or transcript in {split} split")
                    self.rows.append((audio_path, transcript))
        if not self.rows:
            raise ValueError(f"No records found for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> SpeechSample:
        audio_path, transcript = self.rows[index]
        features = torch.from_numpy(extract_log_mel(load_audio(self.project_root / audio_path))).float()
        targets = torch.tensor(self.vocabulary.encode(transcript), dtype=torch.long)
        return SpeechSample(features, targets, audio_path)


def collate_speech(samples: list[SpeechSample]) -> SpeechBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    for sample in samples:
        if sample.features.ndim != 2 or sample.features.shape[0] != N_MELS or sample.features.shape[1] == 0:
            raise ValueError(f"Invalid log-Mel features for {sample.audio_path}")
        if sample.targets.ndim != 1 or sample.targets.numel() == 0 or bool((sample.targets == BLANK_ID).any()):
            raise ValueError(f"Invalid CTC target for {sample.audio_path}")
        needed = minimum_ctc_frames(sample.targets)
        if sample.features.shape[1] < needed:
            raise ValueError(
                f"CTC target for {sample.audio_path} requires {needed} frames; "
                f"got {sample.features.shape[1]}"
            )

    feature_lengths = torch.tensor([sample.features.shape[1] for sample in samples], dtype=torch.long)
    target_lengths = torch.tensor([sample.targets.numel() for sample in samples], dtype=torch.long)
    features = samples[0].features.new_zeros((len(samples), N_MELS, int(feature_lengths.max())))
    for index, sample in enumerate(samples):
        features[index, :, :sample.features.shape[1]] = sample.features
    return SpeechBatch(
        features,
        torch.cat([sample.targets for sample in samples]),
        feature_lengths,
        target_lengths,
        tuple(sample.audio_path for sample in samples),
    )


def make_loader(dataset: SpeechDataset, batch_size: int = 2, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_speech)


if __name__ == "__main__":
    training = SpeechDataset("train")
    batch = next(iter(make_loader(training)))
    print(f"Features: {tuple(batch.features.shape)}")
    print(f"Concatenated targets: {tuple(batch.targets.shape)}")
    print(f"Feature lengths: {batch.feature_lengths.tolist()}")
    print(f"Target lengths: {batch.target_lengths.tolist()}")
    for split in SPLITS:
        dataset = SpeechDataset(split)
        for checked_batch in make_loader(dataset):
            print(f"PASS: CTC frame check: {', '.join(checked_batch.audio_paths)}")
