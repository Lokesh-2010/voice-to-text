import argparse
from pathlib import Path

import torch

from .model import CNNBiLSTM
from .predict import greedy_ctc_decode, load_model, predict
from .train import AudioTranscriptDataset, BLANK_ID
from .audio_preprocessing import N_MELS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = Path("results/checkpoints/overfit_one_te_sample.pt")
DEFAULT_AUDIO_FILE = "enlish_sample.m4a"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train on one verified pair and test exact-sample CTC decoding."
    )
    parser.add_argument("--audio-file", default=DEFAULT_AUDIO_FILE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")
    if not args.checkpoint.is_absolute():
        args.checkpoint = PROJECT_ROOT / args.checkpoint
    return args


def main():
    args = parse_args()
    if args.checkpoint.exists():
        raise FileExistsError(
            f"Checkpoint already exists and will not be overwritten: "
            f"{args.checkpoint}"
        )

    torch.set_num_threads(1)
    torch.manual_seed(42)
    dataset = AudioTranscriptDataset(
        PROJECT_ROOT / "data" / "transcripts" / "transcripts.csv",
        PROJECT_ROOT / "data" / "audio",
        require_verified=True,
    )
    try:
        filename, language, features, target = next(
            row for row in dataset.examples if row[0] == args.audio_file
        )
    except StopIteration as error:
        raise ValueError(
            f"No verified usable dataset example found for {args.audio_file!r}"
        ) from error

    model = CNNBiLSTM(
        n_mels=N_MELS,
        vocab_size=len(dataset.vocabulary.char_to_id),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    features = features.unsqueeze(0)
    input_lengths = torch.tensor([features.shape[-1]], dtype=torch.long)
    target_lengths = torch.tensor([target.numel()], dtype=torch.long)
    output_lengths = model.output_lengths(input_lengths)
    target_text = dataset.vocabulary.decode_ctc(target.tolist())

    best_loss = float("inf")
    best_state = None
    exact_step = None
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(features, input_lengths)
        loss = criterion(
            logits.log_softmax(dim=-1).transpose(0, 1),
            target,
            output_lengths,
            target_lengths,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()

        if step == 1 or step % 25 == 0:
            model.eval()
            with torch.inference_mode():
                eval_logits = model(features)[0]
                token_ids = eval_logits.argmax(dim=-1).tolist()
                decoded = greedy_ctc_decode(
                    token_ids,
                    dataset.vocabulary.id_to_char,
                    BLANK_ID,
                )
                eval_loss = criterion(
                    eval_logits.log_softmax(dim=-1).unsqueeze(1),
                    target,
                    output_lengths,
                    target_lengths,
                ).item()
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if decoded == target_text:
                exact_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                break

    if best_state is None:
        raise RuntimeError("No model state was evaluated")

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists():
        raise FileExistsError(
            f"Checkpoint already exists and will not be overwritten: "
            f"{args.checkpoint}"
        )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_id": dataset.vocabulary.char_to_id,
            "n_mels": N_MELS,
            "blank_id": BLANK_ID,
        },
        args.checkpoint,
    )

    _, id_to_char, blank_id = load_model(args.checkpoint)
    actual_text = predict(
        PROJECT_ROOT / "data" / "audio" / filename,
        args.checkpoint,
    )
    print(f"Audio: {filename} ({language})")
    print(f"Training steps: {step}; exact-match step: {exact_step}")
    print(f"Best CTC loss: {best_loss:.4f}")
    print(f"Target: {target_text!r}")
    print(f"Decoded: {actual_text!r}")
    print(f"Exact match: {actual_text == target_text}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Blank token: {id_to_char[blank_id]!r} (ID {blank_id})")


if __name__ == "__main__":
    main()
