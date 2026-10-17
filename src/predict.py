import argparse
from pathlib import Path

import torch

from .audio_preprocessing import N_MELS, extract_mel_spectrogram, load_audio
from .model import CNNBiLSTM
from .train import BLANK_ID, BLANK_TOKEN


def greedy_ctc_decode(token_ids, id_to_char, blank_id=BLANK_ID):
    decoded = []
    previous_id = None
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id != previous_id and token_id != blank_id:
            if token_id not in id_to_char:
                raise ValueError(f"Unknown token ID in prediction: {token_id}")
            decoded.append(id_to_char[token_id])
        previous_id = token_id
    return "".join(decoded)


def run_synthetic_self_test():
    id_to_char = {1: "a", 2: "b"}
    result = greedy_ctc_decode([1, 1, BLANK_ID, 1, 2, 2], id_to_char)
    if result != "aab":
        raise AssertionError(f"Expected 'aab' from synthetic CTC tokens, got {result!r}")
    print("Synthetic prediction test passed: repeated tokens and blanks decoded correctly.")


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    required_keys = {"model_state_dict", "char_to_id", "n_mels", "blank_id"}
    missing_keys = required_keys - set(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing_keys)}"
        )

    char_to_id = checkpoint["char_to_id"]
    blank_id = checkpoint["blank_id"]
    n_mels = checkpoint["n_mels"]
    if not isinstance(char_to_id, dict) or char_to_id.get(BLANK_TOKEN) != blank_id:
        raise ValueError("Checkpoint has an invalid character vocabulary or blank ID")
    if blank_id != BLANK_ID:
        raise ValueError(
            f"This project expects CTC blank ID {BLANK_ID}, got {blank_id}"
        )
    if n_mels != N_MELS:
        raise ValueError(
            f"Checkpoint expects {n_mels} Mel bins, but preprocessing provides {N_MELS}"
        )

    token_ids = sorted(char_to_id.values())
    if token_ids != list(range(len(token_ids))):
        raise ValueError("Checkpoint character IDs must be contiguous from zero")
    id_to_char = {token_id: character for character, token_id in char_to_id.items()}

    model = CNNBiLSTM(n_mels=n_mels, vocab_size=len(char_to_id))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, id_to_char, blank_id


def predict(audio_path, checkpoint_path):
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model, id_to_char, blank_id = load_model(checkpoint_path)
    waveform = load_audio(str(audio_path))
    features = extract_mel_spectrogram(waveform)
    if features.ndim != 3 or features.shape[0] != 1 or features.shape[1] != N_MELS:
        raise ValueError(
            f"Expected audio features shaped [1, {N_MELS}, time], "
            f"got {tuple(features.shape)}"
        )

    with torch.inference_mode():
        logits = model(features)
        best_token_ids = logits.argmax(dim=-1)[0]
    return greedy_ctc_decode(best_token_ids, id_to_char, blank_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Greedy CTC decoding for a trained CNN-BiLSTM checkpoint."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Test CTC decoding with synthetic token IDs; no files or model needed.",
    )
    mode.add_argument(
        "--audio",
        help="Path to a real audio file to decode using a trained checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/cnn_bilstm_ctc.pt",
        help="Path to the checkpoint created by src.train.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_synthetic_self_test()
        return

    transcript = predict(args.audio, args.checkpoint)
    print(f"Greedy CTC output (unverified): {transcript}")


if __name__ == "__main__":
    main()