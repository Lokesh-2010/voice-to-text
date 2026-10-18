import argparse
import unicodedata
from pathlib import Path

import torch

if __package__:
    from .predict import greedy_ctc_decode, load_model
    from .train import AudioTranscriptDataset, assign_splits
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.predict import greedy_ctc_decode, load_model
    from src.train import AudioTranscriptDataset, assign_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _normalize(text):
    return unicodedata.normalize("NFC", text).strip()


def error_counts(reference, hypothesis):
    """Return (word_errors, word_total, char_errors, char_total)."""
    reference, hypothesis = _normalize(reference), _normalize(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    return (
        edit_distance(ref_words, hyp_words),
        len(ref_words),
        edit_distance(reference, hypothesis),
        len(reference),
    )


def run_self_test():
    word_errors, word_total, char_errors, char_total = error_counts(
        "the cat sat", "the cat sit"
    )
    assert (word_errors, word_total) == (1, 3), (word_errors, word_total)
    assert (char_errors, char_total) == (1, 11), (char_errors, char_total)
    assert error_counts("abc", "abc")[::2] == (0, 0)
    assert error_counts("abc", "")[2] == 3
    assert error_counts("నా పేరు", "నా పేరు")[::2] == (0, 0)
    assert edit_distance("kitten", "sitting") == 3
    print("Evaluation self-test passed: WER/CER edit distances are correct.")


def evaluate(checkpoint_path, split_names):
    model, id_to_char, blank_id = load_model(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_files = checkpoint.get("train_files")

    dataset = AudioTranscriptDataset(
        PROJECT_ROOT / "data" / "transcripts" / "transcripts.csv",
        PROJECT_ROOT / "data" / "audio",
        require_verified=True,
    )
    splits = assign_splits(dataset.examples)
    transcripts = {
        name: "".join(dataset.vocabulary.id_to_char[i] for i in target.tolist())
        for name, _, _, target in dataset.examples
    }

    if train_files is None:
        print(
            "WARNING: checkpoint does not record its training files; it is "
            "assumed to have seen ALL clips, so no split is independent."
        )

    for split_name in split_names:
        totals = [0, 0, 0, 0]
        print(f"\n== {split_name} split ({len(splits[split_name])} clip(s)) ==")
        for filename, language, features, _ in splits[split_name]:
            with torch.inference_mode():
                ids = model(features.unsqueeze(0)).argmax(dim=-1)[0]
            hypothesis = greedy_ctc_decode(ids, id_to_char, blank_id)
            reference = transcripts[filename]
            counts = error_counts(reference, hypothesis)
            totals = [a + b for a, b in zip(totals, counts)]
            seen = train_files is None or filename in train_files
            print(f"[{language}] {filename}" + ("  (SEEN IN TRAINING)" if seen else ""))
            print(f"  reference : {reference}")
            print(f"  prediction: {hypothesis}")
            print(
                f"  WER {counts[0] / counts[1]:.3f} ({counts[0]}/{counts[1]})  "
                f"CER {counts[2] / counts[3]:.3f} ({counts[2]}/{counts[3]})"
            )
        if totals[1]:
            print(
                f"{split_name} corpus WER {totals[0] / totals[1]:.3f}, "
                f"CER {totals[2] / totals[3]:.3f}"
            )
        leaked = [
            example[0]
            for example in splits[split_name]
            if train_files is None or example[0] in train_files
        ]
        if leaked:
            print(
                f"NOT INDEPENDENT: {len(leaked)}/{len(splits[split_name])} "
                f"{split_name} clip(s) were used to train this checkpoint."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute WER and CER on the deterministic dataset splits."
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    args = parser.parse_args()
    if not args.self_test and args.checkpoint is None:
        parser.error("--checkpoint is required unless --self-test is used")
    return args


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    names = ("train", "validation", "test") if args.split == "all" else (args.split,)
    evaluate(args.checkpoint, names)


if __name__ == "__main__":
    main()
