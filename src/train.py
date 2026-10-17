import argparse
import csv
import unicodedata
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .audio_preprocessing import N_MELS, extract_mel_spectrogram, load_audio
    from .model import CNNBiLSTM
    from .verify_dataset import ALTERNATE_AUDIO_FILES
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.audio_preprocessing import N_MELS, extract_mel_spectrogram, load_audio
    from src.model import CNNBiLSTM
    from src.verify_dataset import ALTERNATE_AUDIO_FILES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = ("audio_file", "language", "transcript")
VERIFICATION_COLUMNS = ("audio_verified", "transcript_verified")
ALLOWED_COLUMNS = (*REQUIRED_COLUMNS, *VERIFICATION_COLUMNS)
SUPPORTED_LANGUAGES = {"en", "te"}
BLANK_TOKEN = "<blank>"
BLANK_ID = 0


class CharacterVocabulary:
    def __init__(self, transcripts):
        characters = sorted(
            {
                character
                for text in transcripts
                for character in unicodedata.normalize("NFC", text)
            }
        )
        self.char_to_id = {
            BLANK_TOKEN: BLANK_ID,
            **{character: index for index, character in enumerate(characters, start=1)},
        }
        self.id_to_char = {index: char for char, index in self.char_to_id.items()}

    def encode(self, text):
        normalized = unicodedata.normalize("NFC", text)
        try:
            return [self.char_to_id[character] for character in normalized]
        except KeyError as error:
            raise ValueError(
                f"Transcript contains a character not in the vocabulary: {error.args[0]!r}"
            ) from error

    def decode_ctc(self, token_ids):
        characters = []
        previous_id = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != previous_id and token_id != BLANK_ID:
                if token_id not in self.id_to_char:
                    raise ValueError(f"Unknown token ID: {token_id}")
                characters.append(self.id_to_char[token_id])
            previous_id = token_id
        return "".join(characters)


def _is_verified(value):
    return (value or "").strip().lower() == "true"


class AudioTranscriptDataset(Dataset):
    def __init__(self, csv_path, audio_dir, require_verified=False):
        self.examples = []
        self.language_counts = {}
        self.unverified_audio = []
        self.unverified_transcripts = []
        self.referenced_audio_files = set()
        row_counts = {
            "rows": 0,
            "invalid": 0,
            "missing_audio": 0,
            "unsupported_language": 0,
            "unreadable_audio": 0,
            "ctc_too_short": 0,
            "usable": 0,
        }
        valid_rows = []
        csv_path = Path(csv_path)
        audio_dir = Path(audio_dir)

        with csv_path.open(newline="", encoding="utf-8") as transcript_file:
            reader = csv.DictReader(transcript_file)
            fieldnames = tuple(reader.fieldnames or ())
            missing_columns = set(REQUIRED_COLUMNS) - set(fieldnames)
            unknown_columns = set(fieldnames) - set(ALLOWED_COLUMNS)
            if missing_columns or unknown_columns:
                raise ValueError(
                    "Invalid CSV columns: "
                    f"missing={sorted(missing_columns)}, "
                    f"unsupported={sorted(unknown_columns)}; "
                    f"required={REQUIRED_COLUMNS}, "
                    f"optional={VERIFICATION_COLUMNS}"
                )

            for row_number, row in enumerate(reader, start=2):
                row_counts["rows"] += 1
                if None in row:
                    row_counts["invalid"] += 1
                    print(
                        f"Skipping CSV row {row_number}: too many fields for "
                        "the declared columns"
                    )
                    continue

                audio_filename = (row.get("audio_file") or "").strip()
                language = (row.get("language") or "").strip()
                transcript = unicodedata.normalize(
                    "NFC", (row.get("transcript") or "").strip()
                )
                audio_verified = _is_verified(row.get("audio_verified"))
                transcript_verified = _is_verified(row.get("transcript_verified"))

                if not audio_filename or Path(audio_filename).name != audio_filename:
                    row_counts["invalid"] += 1
                    print(
                        f"Skipping CSV row {row_number}: invalid audio filename "
                        f"{audio_filename!r}"
                    )
                    continue
                if not language or not transcript:
                    row_counts["invalid"] += 1
                    print(
                        f"Skipping CSV row {row_number}: language and transcript "
                        "must be non-empty"
                    )
                    continue
                language = language.lower()
                if language not in SUPPORTED_LANGUAGES:
                    row_counts["unsupported_language"] += 1
                    print(
                        f"Skipping CSV row {row_number}: unsupported language "
                        f"{language!r}; use 'en' or 'te'"
                    )
                    continue

                audio_path = audio_dir / audio_filename
                if not audio_path.is_file():
                    row_counts["missing_audio"] += 1
                    print(
                        f"Skipping CSV row {row_number}: audio file not found: "
                        f"{audio_path}"
                    )
                    continue
                self.referenced_audio_files.add(audio_filename)

                try:
                    waveform = load_audio(str(audio_path))
                    features = extract_mel_spectrogram(waveform)
                except (OSError, RuntimeError, ValueError) as error:
                    row_counts["unreadable_audio"] += 1
                    print(
                        f"Skipping CSV row {row_number}: could not process "
                        f"{audio_path}: {error}"
                    )
                    continue

                if features.ndim != 3 or features.shape[0] != 1:
                    row_counts["invalid"] += 1
                    print(
                        f"Skipping CSV row {row_number}: expected one-channel "
                        f"features shaped [1, mel, time], got {tuple(features.shape)}"
                    )
                    continue
                if features.shape[1] != N_MELS or features.shape[2] == 0:
                    row_counts["invalid"] += 1
                    print(
                        f"Skipping CSV row {row_number}: invalid feature shape "
                        f"{tuple(features.shape)}"
                    )
                    continue

                if not audio_verified:
                    self.unverified_audio.append(audio_filename)
                if not transcript_verified:
                    self.unverified_transcripts.append(audio_filename)

                if require_verified and not (audio_verified and transcript_verified):
                    print(
                        f"Skipping CSV row {row_number}: training requires "
                        "'audio_verified=true' and 'transcript_verified=true'"
                    )
                    continue

                valid_rows.append(
                    (audio_filename, language, transcript, features.squeeze(0))
                )

        if not valid_rows:
            if require_verified and row_counts["rows"]:
                raise ValueError(
                    "No verified training samples. Check the recording and "
                    "transcript, then add audio_verified=true and "
                    "transcript_verified=true columns/values to the CSV."
                )
            raise ValueError(
                f"No usable audio/transcript rows found in {csv_path}. "
                "See the skipped-row messages above."
            )

        self.vocabulary = CharacterVocabulary(
            transcript for _, _, transcript, _ in valid_rows
        )
        for audio_filename, language, transcript, features in valid_rows:
            target = torch.tensor(
                self.vocabulary.encode(transcript), dtype=torch.long
            )
            minimum_frames = target.numel()
            if target.numel() > 1:
                minimum_frames += int((target[1:] == target[:-1]).sum().item())
            if features.shape[1] < minimum_frames:
                print(
                    f"Skipping {audio_filename}: {features.shape[1]} audio frames "
                    f"cannot represent this CTC target (needs at least "
                    f"{minimum_frames})"
                )
                row_counts["ctc_too_short"] += 1
                continue
            self.examples.append((audio_filename, language, features, target))
            self.language_counts[language] = self.language_counts.get(language, 0) + 1
            row_counts["usable"] += 1

        self.row_counts = row_counts

        if not self.examples:
            raise ValueError("No examples remain after CTC target-length validation")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        _, _, features, target = self.examples[index]
        return features, target


def collate_batch(batch):
    feature_sequences, targets = zip(*batch)
    input_lengths = torch.tensor(
        [features.shape[1] for features in feature_sequences], dtype=torch.long
    )
    target_lengths = torch.tensor(
        [target.numel() for target in targets], dtype=torch.long
    )
    padded_features = pad_sequence(
        [features.transpose(0, 1) for features in feature_sequences],
        batch_first=True,
    ).transpose(1, 2)
    concatenated_targets = torch.cat(targets)
    return padded_features, concatenated_targets, input_lengths, target_lengths


def run_synthetic_self_test():
    torch.manual_seed(0)
    unicode_vocabulary = CharacterVocabulary(["Hi నమస్కారం"])
    unicode_ids = unicode_vocabulary.encode("Hi నమస్కారం")
    if unicode_vocabulary.char_to_id[BLANK_TOKEN] != BLANK_ID:
        raise AssertionError("The CTC blank token must have ID 0")
    if not unicode_ids or any(token_id == BLANK_ID for token_id in unicode_ids):
        raise AssertionError("Character encoding must not use the blank ID")

    sequence_lengths = torch.tensor([12, 20], dtype=torch.long)
    vocab_size = 8
    feature_sequences = [
        torch.randn(N_MELS, length) for length in sequence_lengths.tolist()
    ]
    targets_by_sample = [
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([2, 3, 4], dtype=torch.long),
    ]
    features = pad_sequence(
        [sequence.transpose(0, 1) for sequence in feature_sequences],
        batch_first=True,
    ).transpose(1, 2)
    targets = torch.cat(targets_by_sample)
    target_lengths = torch.tensor(
        [target.numel() for target in targets_by_sample], dtype=torch.long
    )

    model = CNNBiLSTM(n_mels=N_MELS, vocab_size=vocab_size)
    model.eval()
    logits = model(features, sequence_lengths)
    output_lengths = model.output_lengths(sequence_lengths)
    if tuple(logits.shape) != (2, int(sequence_lengths.max()), vocab_size):
        raise AssertionError(f"Unexpected logits shape: {tuple(logits.shape)}")
    if not torch.equal(output_lengths, sequence_lengths):
        raise AssertionError(
            f"Unexpected output lengths: {output_lengths.tolist()}"
        )
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    loss = torch.nn.CTCLoss(blank=BLANK_ID)(
        log_probs, targets, output_lengths, target_lengths
    )
    loss.backward()

    if not torch.isfinite(loss):
        raise AssertionError("Synthetic CTC loss is not finite")
    print(
        "Synthetic test passed: "
        f"features={tuple(features.shape)}, "
        f"logits={tuple(logits.shape)}, "
        f"ctc_input={tuple(log_probs.shape)}, "
        f"input_lengths={sequence_lengths.tolist()}, "
        f"output_lengths={output_lengths.tolist()}, "
        f"target_lengths={target_lengths.tolist()}, "
        "CTC loss and backward pass succeeded."
    )


def train(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint already exists and will not be overwritten: "
            f"{checkpoint_path}. Choose a new --checkpoint path."
        )

    dataset = AudioTranscriptDataset(
        PROJECT_ROOT / "data" / "transcripts" / "transcripts.csv",
        PROJECT_ROOT / "data" / "audio",
        require_verified=True,
    )
    missing_languages = SUPPORTED_LANGUAGES - set(dataset.language_counts)
    if missing_languages:
        raise ValueError(
            "Bilingual training requires verified samples for both 'en' and 'te'; "
            f"missing: {', '.join(sorted(missing_languages))}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    vocabulary = dataset.vocabulary
    model = CNNBiLSTM(n_mels=N_MELS, vocab_size=len(vocabulary.char_to_id))
    device = torch.device("cpu")
    model.to(device)

    criterion = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for features, targets, input_lengths, target_lengths in loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features, input_lengths)
            output_lengths = model.output_lengths(input_lengths)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                targets,
                output_lengths,
                target_lengths,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite CTC loss at epoch {epoch}; training stopped"
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} - mean CTC loss: {mean_loss:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_id": vocabulary.char_to_id,
            "n_mels": N_MELS,
            "blank_id": BLANK_ID,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def check_data():
    dataset = AudioTranscriptDataset(
        PROJECT_ROOT / "data" / "transcripts" / "transcripts.csv",
        PROJECT_ROOT / "data" / "audio",
    )
    actual_audio_files = {
        path.name
        for path in (PROJECT_ROOT / "data" / "audio").iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".m4a", ".mp3", ".flac"}
    }
    referenced_files = dataset.referenced_audio_files
    recognized_alternates = {
        alternate
        for alternate, canonical in ALTERNATE_AUDIO_FILES.items()
        if alternate in actual_audio_files and canonical in referenced_files
    }
    unreferenced_audio = sorted(
        actual_audio_files - referenced_files - recognized_alternates
    )
    print(
        f"Dataset validation: {dataset.row_counts['rows']} CSV row(s), "
        f"{len(dataset)} usable sample(s), "
        f"{dataset.row_counts['missing_audio']} missing audio file(s), "
        f"{dataset.row_counts['unreadable_audio']} unreadable audio file(s), "
        f"{dataset.row_counts['invalid']} invalid row(s), "
        f"{dataset.row_counts['unsupported_language']} unsupported language "
        "label(s)."
    )
    print(f"Usable language counts: {dataset.language_counts or 'none'}")
    print(
        "Unverified audio: "
        f"{dataset.unverified_audio or 'none'}"
    )
    print(
        "Unverified transcripts: "
        f"{dataset.unverified_transcripts or 'none'}"
    )
    print(f"Audio files not referenced by CSV: {unreferenced_audio or 'none'}")
    if dataset.unverified_audio or dataset.unverified_transcripts:
        print(
            "For training, review each recording/transcript pair and set "
            "audio_verified=true and transcript_verified=true in the CSV. "
            "Bilingual training also needs verified rows for both en and te."
        )
    else:
        missing_languages = sorted(SUPPORTED_LANGUAGES - set(dataset.language_counts))
        if missing_languages:
            print(
                "All audio/transcript rows are marked verified, but bilingual "
                "training is missing verified samples for: "
                f"{', '.join(missing_languages)}."
            )
        else:
            print(
                "All audio/transcript rows are marked verified, with samples "
                "for both en and te."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CNN-BiLSTM with CTC on the project transcript CSV."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="Run a synthetic shape/CTC test without reading the dataset or training.",
    )
    mode.add_argument(
        "--check-data",
        action="store_true",
        help="Load and validate the CSV/audio without training.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run real training on the CSV/audio dataset.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint",
        default="models/cnn_bilstm_ctc.pt",
        help="Checkpoint output path (created only when real training runs).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    return args


def main():
    args = parse_args()
    if args.self_test:
        run_synthetic_self_test()
    elif args.check_data:
        check_data()
    else:
        train(args)


if __name__ == "__main__":
    main()
