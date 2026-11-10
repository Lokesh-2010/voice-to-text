"""Run greedy CTC inference on one test recording per language."""

import csv
from pathlib import Path

import torch
from torch import Tensor

from .audio_preprocessing import extract_log_mel, load_audio
from .dataset import DEFAULT_CHECKPOINT_PATH, DEFAULT_VOCAB_PATH, PROJECT_ROOT
from .model import CNNBiLSTM
from .vocabulary import BLANK_ID, UNKNOWN_ID, Vocabulary


def greedy_ctc_ids(logits: Tensor) -> list[int]:
    """Collapse consecutive IDs before removing blanks from (time, classes) logits."""
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("Logits must have shape (time, classes) with nonzero dimensions")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits contain non-finite values")
    result = []
    previous = None
    for token_id in logits.argmax(dim=1).tolist():
        if token_id != previous and token_id != BLANK_ID:
            result.append(token_id)
        previous = token_id
    return result


def character_error_rate(reference: str, prediction: str) -> float:
    """Compute case-sensitive Unicode code-point edit distance / reference length."""
    if not reference:
        raise ValueError("CER is undefined for an empty reference")
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(prediction, start=1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1] / len(reference)


def test_recordings(csv_path: Path) -> list[tuple[str, str, str]]:
    selections = {}
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"split", "language", "audio_path", "transcript"} <= set(reader.fieldnames):
            raise ValueError("CSV is missing required columns")
        for row in reader:
            if row["split"] == "test" and row["language"] in ("english", "telugu"):
                if not row["audio_path"] or not row["transcript"] or not row["transcript"].strip():
                    raise ValueError("Test recording is missing a path or transcript")
                selections.setdefault(row["language"], []).append((row["audio_path"], row["transcript"]))
    results = []
    for language in ("english", "telugu"):
        if language not in selections:
            raise ValueError(f"No {language} test recording found")
        audio_path, transcript = sorted(selections[language])[0]
        results.append((language, audio_path, transcript))
    return results


def load_model(checkpoint_path: Path, vocabulary: Vocabulary) -> CNNBiLSTM:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("num_classes") != len(vocabulary):
        raise ValueError(
            f"Incompatible checkpoint {checkpoint_path}: class count "
            f"{checkpoint.get('num_classes')} does not match vocabulary size {len(vocabulary)}"
        )
    if checkpoint.get("vocabulary_tokens") != vocabulary.id_to_char:
        raise ValueError(f"Incompatible checkpoint {checkpoint_path}: vocabulary mapping missing or different")
    model = CNNBiLSTM(num_classes=len(vocabulary))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    vocabulary = Vocabulary.load(DEFAULT_VOCAB_PATH)
    model = load_model(DEFAULT_CHECKPOINT_PATH, vocabulary)

    with torch.no_grad():
        for language, audio_path, expected in test_recordings(PROJECT_ROOT / "data" / "transcripts.csv"):
            features = torch.from_numpy(extract_log_mel(load_audio(PROJECT_ROOT / audio_path))).float()
            logits = model(features.unsqueeze(0))[0]
            token_ids = greedy_ctc_ids(logits)
            predicted = vocabulary.decode(token_ids)
            cer_text = "".join(
                "\ufffd" if token_id == UNKNOWN_ID else vocabulary.id_to_char[token_id]
                for token_id in token_ids
            )
            cer = character_error_rate(expected, cer_text)
            print(f"{language.capitalize()}: {Path(audio_path).name}")
            print(f"  Expected: {expected}")
            print(f"  Predicted: {predicted if predicted else '[empty prediction]'}")
            print(f"  CER: {cer:.4f} ({cer:.2%})")


if __name__ == "__main__":
    main()
