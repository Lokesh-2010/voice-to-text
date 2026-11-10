"""CTC training and a one-batch forward/backward smoke test."""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .dataset import DEFAULT_CHECKPOINT_PATH, SPLITS, SpeechBatch, SpeechDataset, make_loader, minimum_ctc_frames
from .model import CNNBiLSTM
from .vocabulary import BLANK_ID, UNKNOWN_ID, Vocabulary

SEED = 42
BATCH_SIZE = 2
LEARNING_RATE = 0.001
MAX_EPOCHS = 5


def validate_batch(batch: SpeechBatch, num_classes: int) -> None:
    count = batch.features.shape[0]
    if (
        batch.features.ndim != 3
        or batch.feature_lengths.shape != (count,)
        or batch.target_lengths.shape != (count,)
        or len(batch.audio_paths) != count
        or batch.targets.ndim != 1
        or int(batch.target_lengths.sum()) != batch.targets.numel()
    ):
        raise ValueError("Invalid batch dimensions or target lengths")
    if not torch.isfinite(batch.features).all():
        raise ValueError("Batch features contain non-finite values")
    offset = 0
    for path, feature_length, target_length in zip(
        batch.audio_paths, batch.feature_lengths.tolist(), batch.target_lengths.tolist()
    ):
        if not 0 < feature_length <= batch.features.shape[2] or target_length <= 0:
            raise ValueError(f"Invalid feature or target length for {path}")
        targets = batch.targets[offset:offset + target_length]
        if bool(((targets <= BLANK_ID) | (targets >= num_classes)).any()):
            raise ValueError(f"Invalid CTC target ID for {path}")
        required = minimum_ctc_frames(targets)
        if feature_length < required:
            raise ValueError(f"{path} requires {required} CTC frames, got {feature_length}")
        offset += target_length


def batch_loss(model: CNNBiLSTM, batch: SpeechBatch, criterion: nn.CTCLoss) -> Tensor:
    """Run each sample without padding, then pad logits for batched CTC."""
    validate_batch(batch, model.classifier.out_features)
    sequences = []
    for index, length in enumerate(batch.feature_lengths.tolist()):
        logits = model(batch.features[index:index + 1, :, :length])
        if logits.shape[1] != length:
            raise ValueError("Model changed the time dimension; update CTC output lengths")
        sequences.append(logits.squeeze(0))
    logits = pad_sequence(sequences, batch_first=False)
    if not torch.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits")
    loss = criterion(
        logits.log_softmax(dim=2),
        batch.targets,
        batch.feature_lengths,
        batch.target_lengths,
    )
    if not torch.isfinite(loss):
        raise ValueError("CTC loss is not finite")
    return loss


def validate_datasets(datasets: dict[str, SpeechDataset]) -> None:
    for split, dataset in datasets.items():
        unknown_count = 0
        for batch in make_loader(dataset, batch_size=BATCH_SIZE):
            validate_batch(batch, len(dataset.vocabulary))
            unknown_count += int((batch.targets == UNKNOWN_ID).sum())
        if split == "train" and unknown_count:
            raise ValueError(f"Training transcripts contain {unknown_count} unknown characters")
        print(f"PASS: {split} CTC targets ({len(dataset)} samples, {unknown_count} unknown characters)")


def check_gradients(model: CNNBiLSTM) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing or non-finite gradient: {name}")


def run_epoch(
    model: CNNBiLSTM,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, criterion)
            if optimizer is not None:
                loss.backward()
                check_gradients(model)
                optimizer.step()
            count = len(batch.audio_paths)
            total_loss += loss.item() * count
            total_samples += count
    if total_samples == 0:
        raise ValueError("Cannot calculate loss for an empty loader")
    return total_loss / total_samples


def save_checkpoint(
    path: Path, model: CNNBiLSTM, optimizer: torch.optim.Optimizer, epochs: int, vocabulary: Vocabulary
) -> None:
    """Atomically publish a new checkpoint without replacing an existing file."""
    if model.classifier.out_features != len(vocabulary):
        raise ValueError("Model class count does not match vocabulary")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as file:
            temporary_path = Path(file.name)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_classes": model.classifier.out_features,
                    "vocabulary_tokens": vocabulary.id_to_char,
                    "epochs": epochs,
                    "seed": SEED,
                },
                file,
            )
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Run training instead of the default smoke test")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, choices=range(1, MAX_EPOCHS + 1))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    datasets = {split: SpeechDataset(split) for split in SPLITS}
    validate_datasets(datasets)
    model = CNNBiLSTM(num_classes=len(datasets["train"].vocabulary))
    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    if not args.train:
        batch = next(iter(make_loader(datasets["train"], batch_size=BATCH_SIZE)))
        loss = batch_loss(model, batch, criterion)
        loss.backward()
        check_gradients(model)
        print(f"PASS: one-batch forward/backward smoke test; CTC loss={loss.item():.4f}")
        print("No optimizer step or checkpoint was created.")
        return

    if args.checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {args.checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(datasets["val"], batch_size=BATCH_SIZE)
    for epoch in range(1, args.epochs + 1):
        training_loss = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{args.epochs}: train loss={training_loss:.4f}, validation loss={validation_loss:.4f}")
    save_checkpoint(args.checkpoint, model, optimizer, args.epochs, datasets["train"].vocabulary)
    print(f"Saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
