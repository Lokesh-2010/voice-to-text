import argparse
from collections import Counter
from pathlib import Path

import torch

if __package__:
    from .audio_preprocessing import extract_mel_spectrogram, load_audio
    from .predict import greedy_ctc_decode, load_model
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.audio_preprocessing import extract_mel_spectrogram, load_audio
    from src.predict import greedy_ctc_decode, load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "diagnostic.pt"
DEFAULT_AUDIO_PATH = PROJECT_ROOT / "data" / "audio" / "telugu_sample_2.wav.m4a"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect frame-level CTC predictions for an audio file."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to a model checkpoint (default: results/checkpoints/diagnostic.pt).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO_PATH,
        help="Path to an audio file (default: data/audio/telugu_sample_2.wav.m4a).",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_absolute():
        args.checkpoint = PROJECT_ROOT / args.checkpoint
    if not args.audio.is_absolute():
        args.audio = PROJECT_ROOT / args.audio
    return args


def main():
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    model, id_to_char, blank_id = load_model(args.checkpoint)
    waveform = load_audio(str(args.audio))
    features = extract_mel_spectrogram(waveform)

    with torch.inference_mode():
        logits = model(features)
        predicted_ids = logits[0].argmax(dim=-1).tolist()

    token_counts = Counter(predicted_ids)
    blank_count = token_counts.get(blank_id, 0)
    nonblank_count = len(predicted_ids) - blank_count
    frequent_tokens = token_counts.most_common(10)
    if blank_id not in {token_id for token_id, _ in frequent_tokens}:
        frequent_tokens.append((blank_id, blank_count))

    print(f"Audio: {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model output shape: {tuple(logits.shape)}")
    print(f"Blank token ID: {blank_id}")
    print(f"Blank-token frames: {blank_count}")
    print(f"Blank-frame percentage: {blank_count / len(predicted_ids) * 100:.2f}%")
    print(f"Non-blank frames: {nonblank_count}")
    print("Most frequent predicted tokens (ID, token, count):")
    for token_id, count in frequent_tokens:
        token = id_to_char.get(token_id, f"<unknown:{token_id}>")
        print(f"  ({token_id}, {token!r}, {count})")
    print(
        "Decoded text: "
        f"{greedy_ctc_decode(predicted_ids, id_to_char, blank_id)!r}"
    )


if __name__ == "__main__":
    main()
