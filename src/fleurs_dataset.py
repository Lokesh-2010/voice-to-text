import argparse
import csv
import hashlib
import io
import json
import string
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio_preprocessing import N_MELS, SAMPLE_RATE, extract_mel_spectrogram, load_audio
from .train import CharacterVocabulary


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "google/fleurs"
DATASET_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PAPER_URL = "https://arxiv.org/abs/2205.12446"
LANGUAGE_CONFIGS = {"en_us": "en", "te_in": "te"}
SOURCE_SPLITS = {"train": "train", "validation": "dev", "test": "test"}
PILOT_SPLIT_COUNTS = {"train": 160, "validation": 20, "test": 20}
MANIFEST_COLUMNS = (
    "dataset",
    "dataset_revision",
    "license",
    "config",
    "language",
    "split",
    "source_split",
    "source_id",
    "source_audio_file",
    "audio_file",
    "audio_sha256",
    "transcript",
    "source_transcription",
    "normalized_transcription",
    "character_transcription",
    "num_samples",
    "gender",
)
TSV_COLUMN_COUNT = 7
VOCABULARY_FILENAME = "vocabulary.json"


def _request(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "bilingual-stt-submission/1.0"},
    )
    return urllib.request.urlopen(request, timeout=120)


def _download_tsv(config, source_split):
    url = (
        f"{DATASET_URL}/resolve/{DATASET_REVISION}/"
        f"data/{config}/{source_split}.tsv"
    )
    with _request(url) as response:
        text = response.read().decode("utf-8")

    rows = []
    seen_audio_files = set()
    for line_number, columns in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        if len(columns) not in {TSV_COLUMN_COUNT, TSV_COLUMN_COUNT - 1}:
            raise ValueError(
                f"Unexpected FLEURS TSV shape in {config}/{source_split}.tsv "
                f"line {line_number}: expected {TSV_COLUMN_COUNT} columns "
                f"(or a recoverable embedded-tab row), "
                f"got {len(columns)}"
            )
        if len(columns) == TSV_COLUMN_COUNT:
            (
                source_id,
                audio_file,
                transcript,
                normalized_transcription,
                character_transcription,
                num_samples,
                gender,
            ) = columns
        else:
            (
                source_id,
                audio_file,
                transcript,
                combined_transcription_fields,
                num_samples,
                gender,
            ) = columns
            normalized_transcription, separator, character_transcription = (
                combined_transcription_fields.partition("\t")
            )
            if not separator:
                raise ValueError(
                    f"Malformed embedded-tab FLEURS TSV row in "
                    f"{config}/{source_split}.tsv line {line_number}"
                )
        audio_path = PurePosixPath(audio_file)
        if (
            not source_id.isdecimal()
            or not audio_file
            or audio_path.name != audio_file
            or audio_path.suffix.lower() != ".wav"
        ):
            raise ValueError(
                f"Invalid FLEURS identifier or audio filename in "
                f"{config}/{source_split}.tsv line {line_number}"
            )
        if audio_file in seen_audio_files:
            raise ValueError(
                f"Duplicate FLEURS audio filename {audio_file} in "
                f"{config}/{source_split}.tsv"
            )
        if not transcript.strip():
            raise ValueError(
                f"Empty FLEURS transcript in {config}/{source_split}.tsv "
                f"line {line_number}"
            )
        if not num_samples.isdecimal():
            raise ValueError(
                f"Invalid FLEURS sample count in {config}/{source_split}.tsv "
                f"line {line_number}"
            )
        seen_audio_files.add(audio_file)
        rows.append(
            {
                "source_id": source_id,
                "source_audio_file": audio_file,
                "transcript": unicodedata.normalize("NFC", transcript.strip()),
                "source_transcription": transcript,
                "normalized_transcription": normalized_transcription,
                "character_transcription": character_transcription,
                "num_samples": int(num_samples),
                "gender": gender,
            }
        )
    return rows


def _select_rows(rows, config, source_split, count):
    unique_rows = {}
    for row in rows:
        source_id = row["source_id"]
        current = unique_rows.get(source_id)
        rank = hashlib.sha256(
            (
                f"{DATASET_REVISION}\t{config}\t{source_split}\t{source_id}\t"
                f"{row['source_audio_file']}"
            ).encode("utf-8")
        ).digest()
        if current is None or rank < current[0]:
            unique_rows[source_id] = (rank, row)
    candidates = [row for _, row in unique_rows.values()]
    if len(candidates) < count:
        raise ValueError(
            f"FLEURS {config}/{source_split} has {len(candidates)} unique "
            f"source IDs; {count} are required"
        )
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(
            (
                f"{DATASET_REVISION}\t{config}\t{source_split}\t"
                f"{row['source_id']}"
            ).encode("utf-8")
        ).digest(),
    )[:count]


def _extract_selected_archive(config, source_split, selected_rows, output_dir):
    url = (
        f"{DATASET_URL}/resolve/{DATASET_REVISION}/"
        f"data/{config}/audio/{source_split}.tar.gz"
    )
    selected_by_name = {
        row["source_audio_file"]: row for row in selected_rows
    }
    extracted = {}
    print(f"Streaming FLEURS archive {config}/{source_split}.tar.gz")
    with _request(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(
                        f"Unsafe member path in FLEURS archive: {member.name!r}"
                    )
                audio_file = member_path.name
                if audio_file not in selected_by_name:
                    continue
                expected_member = PurePosixPath(source_split, audio_file)
                if member_path != expected_member or not member.isfile():
                    raise ValueError(
                        f"Unexpected selected FLEURS archive member: {member.name!r}"
                    )
                if audio_file in extracted:
                    raise ValueError(
                        f"Duplicate audio member in FLEURS archive: {member.name!r}"
                    )

                target = output_dir / audio_file
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise OSError(
                        f"Could not read selected archive member {member.name!r}"
                    )
                digest = hashlib.sha256()
                written = 0
                with source, target.open("xb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != member.size:
                    raise OSError(
                        f"Truncated FLEURS audio {member.name!r}: "
                        f"expected {member.size} bytes, wrote {written}"
                    )
                extracted[audio_file] = digest.hexdigest()

    missing = sorted(set(selected_by_name) - set(extracted))
    if missing:
        raise FileNotFoundError(
            f"FLEURS archive {config}/{source_split} lacked "
            f"{len(missing)} selected recording(s), including {missing[:3]}"
        )
    return extracted


def _write_attribution(path):
    attribution = f"""# FLEURS dataset attribution

- Dataset: [FLEURS]({DATASET_URL})
- Dataset revision: `{DATASET_REVISION}`
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)]({LICENSE_URL})
- Source paper: [FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech]({PAPER_URL})

This local pilot contains 160 training, 20 validation, and 20 test recordings
per language, selected from the official FLEURS `train`, `dev`, and `test`
splits respectively. Repeated numeric source IDs correspond to multiple
recordings of the same sentence; the deterministic selection retains at most
one recording per source ID per split. Audio files are retained in their source WAV format.
The source sentence is NFC-normalized for the project manifest; original,
normalized, and character-level transcript fields, official split names,
source IDs, and audio SHA-256 hashes are retained for traceability. English
`en_us` and Telugu `te_in` are mapped to project language codes `en` and `te`.

When redistributing or publishing adapted material, provide appropriate credit,
link this dataset source and the CC BY 4.0 license, and identify these subset
selection and transcript-normalization changes. This attribution does not
imply FLEURS author endorsement.

Suggested paper citation:

> Conneau, Alexis, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth
> Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. “FLEURS: Few-shot Learning
> Evaluation of Universal Representations of Speech.” arXiv:2205.12446, 2022.
"""
    path.write_text(attribution, encoding="utf-8")


def _write_training_vocabulary(output_dir, training_rows_by_config):
    vocabulary_transcripts = [
        row["transcript"]
        for config in LANGUAGE_CONFIGS
        for row in training_rows_by_config[config]
    ]
    vocabulary = CharacterVocabulary(
        [*vocabulary_transcripts, string.punctuation]
    )
    payload = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": "train",
        "training_rows_by_config": {
            config: len(training_rows_by_config[config])
            for config in LANGUAGE_CONFIGS
        },
        "fixed_character_inventory": "ASCII punctuation (string.punctuation)",
        "characters": sorted(
            character
            for character in vocabulary.char_to_id
            if character != "<blank>"
        ),
        "char_to_id": vocabulary.char_to_id,
    }
    vocabulary_path = output_dir / VOCABULARY_FILENAME
    with vocabulary_path.open("x", encoding="utf-8") as vocabulary_file:
        json.dump(payload, vocabulary_file, ensure_ascii=False, indent=2, sort_keys=True)
        vocabulary_file.write("\n")
    return vocabulary_path


def _read_training_vocabulary(manifest_root):
    vocabulary_path = manifest_root / VOCABULARY_FILENAME
    if not vocabulary_path.is_file():
        raise FileNotFoundError(
            f"FLEURS training vocabulary is missing: {vocabulary_path}. "
            "Build it from the official training splits before validation."
        )
    with vocabulary_path.open(encoding="utf-8") as vocabulary_file:
        payload = json.load(vocabulary_file)
    if (
        payload.get("dataset") != DATASET_ID
        or payload.get("dataset_revision") != DATASET_REVISION
        or payload.get("source_split") != "train"
        or payload.get("fixed_character_inventory")
        != "ASCII punctuation (string.punctuation)"
    ):
        raise ValueError(
            f"FLEURS vocabulary provenance is invalid: {vocabulary_path}"
        )
    rows_by_config = payload.get("training_rows_by_config")
    if not isinstance(rows_by_config, dict) or set(rows_by_config) != set(
        LANGUAGE_CONFIGS
    ):
        raise ValueError(
            f"FLEURS vocabulary must cover both official training configs: "
            f"{vocabulary_path}"
        )
    characters = payload.get("characters")
    if (
        not isinstance(characters, list)
        or any(not isinstance(character, str) or len(character) != 1 for character in characters)
        or len(characters) != len(set(characters))
    ):
        raise ValueError(f"Invalid character inventory in {vocabulary_path}")
    vocabulary = CharacterVocabulary(characters)
    if payload.get("char_to_id") != vocabulary.char_to_id:
        raise ValueError(
            f"Character IDs do not match the saved inventory: {vocabulary_path}"
        )
    return vocabulary


def build_training_vocabulary(manifest_path):
    manifest_path, rows = _read_manifest(manifest_path)
    for row in rows:
        if row["dataset"] != DATASET_ID or row["dataset_revision"] != DATASET_REVISION:
            raise ValueError("Manifest does not match the pinned FLEURS dataset revision")
    training_rows_by_config = {
        config: _download_tsv(config, SOURCE_SPLITS["train"])
        for config in LANGUAGE_CONFIGS
    }
    vocabulary_path = _write_training_vocabulary(
        manifest_path.parent,
        training_rows_by_config,
    )
    print(
        "Built FLEURS vocabulary from official training transcript splits only "
        f"({sum(map(len, training_rows_by_config.values()))} source rows) plus "
        f"fixed ASCII punctuation: {vocabulary_path}"
    )
    return vocabulary_path


def prepare_fleurs(output_dir=None):
    output_dir = Path(output_dir or PROJECT_ROOT / "data" / "fleurs")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    staging_dir = output_dir.with_name(f".{output_dir.name}-preparing")
    if output_dir.exists():
        raise FileExistsError(
            f"FLEURS output already exists and will not be overwritten: {output_dir}"
        )
    if staging_dir.exists():
        raise FileExistsError(
            f"FLEURS preparation staging directory already exists: {staging_dir}"
        )

    staging_dir.mkdir(parents=True)
    try:
        manifest_rows = []
        training_rows_by_config = {}
        for config, language in LANGUAGE_CONFIGS.items():
            for split, source_split in SOURCE_SPLITS.items():
                tsv_rows = _download_tsv(config, source_split)
                if split == "train":
                    training_rows_by_config[config] = tsv_rows
                selected_rows = _select_rows(
                    tsv_rows,
                    config,
                    source_split,
                    PILOT_SPLIT_COUNTS[split],
                )
                audio_dir = (
                    staging_dir / "audio" / config / source_split
                )
                audio_hashes = _extract_selected_archive(
                    config,
                    source_split,
                    selected_rows,
                    audio_dir,
                )
                for row in selected_rows:
                    manifest_rows.append(
                        {
                            "dataset": DATASET_ID,
                            "dataset_revision": DATASET_REVISION,
                            "license": "CC-BY-4.0",
                            "config": config,
                            "language": language,
                            "split": split,
                            "source_split": source_split,
                            "source_id": row["source_id"],
                            "source_audio_file": row["source_audio_file"],
                            "audio_file": str(
                                Path("audio")
                                / config
                                / source_split
                                / row["source_audio_file"]
                            ),
                            "audio_sha256": audio_hashes[row["source_audio_file"]],
                            "transcript": row["transcript"],
                            "source_transcription": row["source_transcription"],
                            "normalized_transcription": row["normalized_transcription"],
                            "character_transcription": row["character_transcription"],
                            "num_samples": row["num_samples"],
                            "gender": row["gender"],
                        }
                    )

        _write_training_vocabulary(staging_dir, training_rows_by_config)
        manifest_path = staging_dir / "manifest.csv"
        with manifest_path.open("x", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest_rows)
        _write_attribution(staging_dir / "ATTRIBUTION.md")
        report = validate_fleurs_manifest(manifest_path)
        if not report["vocabulary_coverage_ok"]:
            raise ValueError(
                "FLEURS pilot has evaluation characters absent from the selected "
                "training transcripts; staging data was not finalized"
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(output_dir)
        print(f"FLEURS pilot prepared and validated: {output_dir}")
        return output_dir / "manifest.csv"
    except Exception:
        print(
            "FLEURS preparation did not complete. Partial files, if any, were "
            f"left for inspection at: {staging_dir}"
        )
        raise


def _read_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = set(MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        extra_columns = set(reader.fieldnames or ()) - set(MANIFEST_COLUMNS)
        if missing_columns or extra_columns:
            raise ValueError(
                f"Invalid FLEURS manifest columns: missing={sorted(missing_columns)}, "
                f"extra={sorted(extra_columns)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"FLEURS manifest is empty: {manifest_path}")
    return manifest_path, rows


def _safe_audio_path(manifest_root, relative_path):
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Audio path must be relative to the manifest: {relative_path!r}")
    resolved_root = manifest_root.resolve()
    resolved_path = (manifest_root / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Audio path escapes the FLEURS data directory: {relative_path!r}"
        ) from error
    return resolved_path


def _minimum_ctc_frames(target):
    required = target.numel()
    if target.numel() > 1:
        required += int((target[1:] == target[:-1]).sum().item())
    return required


def _minimum_ctc_frames_for_text(text):
    normalized = unicodedata.normalize("NFC", text)
    return len(normalized) + sum(
        previous == current for previous, current in zip(normalized, normalized[1:])
    )


def validate_fleurs_manifest(manifest_path):
    manifest_path, rows = _read_manifest(manifest_path)
    expected_counts = {
        (language, split): count
        for language in LANGUAGE_CONFIGS.values()
        for split, count in PILOT_SPLIT_COUNTS.items()
    }
    counts = Counter()
    seen_source_ids = set()
    seen_audio_paths = set()
    training_transcripts = []
    for row in rows:
        if row["dataset"] != DATASET_ID or row["dataset_revision"] != DATASET_REVISION:
            raise ValueError(
                f"Manifest row is not from the pinned FLEURS revision: "
                f"{row['dataset']}@{row['dataset_revision']}"
            )
        if row["license"] != "CC-BY-4.0":
            raise ValueError(f"Unexpected FLEURS license value: {row['license']!r}")
        config = row["config"]
        expected_language = LANGUAGE_CONFIGS.get(config)
        if expected_language is None or row["language"] != expected_language:
            raise ValueError(
                f"Invalid FLEURS language mapping for config {config!r}: "
                f"{row['language']!r}"
            )
        split = row["split"]
        source_split = row["source_split"]
        if split not in SOURCE_SPLITS or SOURCE_SPLITS[split] != source_split:
            raise ValueError(
                f"Manifest split {split!r} does not match official FLEURS "
                f"split {source_split!r}"
            )
        if not row["source_id"].isdecimal() or not row["transcript"].strip():
            raise ValueError("FLEURS rows require a numeric source ID and transcript")
        if row["transcript"] != unicodedata.normalize(
            "NFC", row["source_transcription"].strip()
        ):
            raise ValueError(
                f"Transcript normalization mismatch for {config} "
                f"source ID {row['source_id']}"
            )
        key = (config, source_split, row["source_id"])
        if key in seen_source_ids:
            raise ValueError(f"Duplicate FLEURS source ID: {key}")
        seen_source_ids.add(key)

        relative_audio_path = Path(row["audio_file"])
        if relative_audio_path.as_posix() in seen_audio_paths:
            raise ValueError(f"Duplicate FLEURS audio path: {row['audio_file']}")
        seen_audio_paths.add(relative_audio_path.as_posix())
        expected_path = Path("audio") / config / source_split / row["source_audio_file"]
        if relative_audio_path != expected_path:
            raise ValueError(
                f"Audio path does not match FLEURS source ID for {key}: "
                f"{row['audio_file']!r}"
            )
        if not row["num_samples"].isdecimal():
            raise ValueError(f"Invalid sample count for FLEURS source ID {key}")
        if len(row["audio_sha256"]) != 64 or any(
            character not in "0123456789abcdef" for character in row["audio_sha256"]
        ):
            raise ValueError(f"Invalid audio SHA-256 for FLEURS source ID {key}")

        counts[(row["language"], split)] += 1
        if split == "train":
            training_transcripts.append(row["transcript"])

    for key, expected in expected_counts.items():
        actual = counts[key]
        if actual != expected:
            raise ValueError(
                f"FLEURS pilot count mismatch for {key[0]}/{key[1]}: "
                f"expected {expected}, got {actual}"
            )
    if len(rows) != sum(expected_counts.values()):
        raise ValueError(
            f"FLEURS manifest has {len(rows)} rows, expected "
            f"{sum(expected_counts.values())}"
        )

    vocabulary = _read_training_vocabulary(manifest_path.parent)
    for transcript in training_transcripts:
        vocabulary.encode(transcript)
    uncovered_characters = {}
    for row in rows:
        normalized_transcript = unicodedata.normalize("NFC", row["transcript"])
        missing_characters = set(normalized_transcript) - set(vocabulary.char_to_id)
        if missing_characters:
            key = (row["language"], row["split"])
            uncovered_characters.setdefault(key, set()).update(missing_characters)
        audio_path = _safe_audio_path(manifest_path.parent, row["audio_file"])
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"FLEURS audio not found for {row['config']} source ID "
                f"{row['source_id']}: {audio_path}"
            )
        audio_info = sf.info(str(audio_path))
        sample_rate = audio_info.samplerate
        channels = audio_info.channels
        frames = audio_info.frames
        if sample_rate != SAMPLE_RATE or channels != 1:
            raise ValueError(
                f"Expected source WAV to be mono {SAMPLE_RATE} Hz, got "
                f"{channels} channel(s) at {sample_rate} Hz: {audio_path}"
            )
        if abs(frames - int(row["num_samples"])) > max(2, int(frames * 0.01)):
            raise ValueError(
                f"FLEURS sample count mismatch for {audio_path}: "
                f"manifest={row['num_samples']}, WAV={frames}"
            )
        digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
        if digest != row["audio_sha256"]:
            raise ValueError(f"FLEURS audio checksum mismatch: {audio_path}")

        waveform = load_audio(str(audio_path))
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError(
                f"Expected mono waveform shaped [1, samples], got "
                f"{tuple(waveform.shape)} for {audio_path}"
            )
        if not torch.isfinite(waveform).all():
            raise ValueError(f"Non-finite audio samples in {audio_path}")
        features = extract_mel_spectrogram(waveform)
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[1] != N_MELS
            or features.shape[2] == 0
            or not torch.isfinite(features).all()
        ):
            raise ValueError(
                f"Invalid Mel features {tuple(features.shape)} for {audio_path}"
            )
        required_frames = _minimum_ctc_frames_for_text(normalized_transcript)
        if features.shape[2] < required_frames:
            raise ValueError(
                f"CTC target is too long for {audio_path}: "
                f"{required_frames} required frames, {features.shape[2]} available"
            )

    seconds_by_split_language = Counter()
    for row in rows:
        seconds_by_split_language[(row["language"], row["split"])] += (
            int(row["num_samples"]) / SAMPLE_RATE
        )
    print("FLEURS manifest checks completed")
    print(f"Manifest: {manifest_path}")
    print(
        f"Rows validated: {len(rows)}; vocabulary classes including blank: "
        f"{len(vocabulary.char_to_id)}"
    )
    print(
        "Vocabulary source: full official training transcript splits plus fixed "
        "ASCII punctuation; held-out transcript text was not used to build it."
    )
    for language in sorted(LANGUAGE_CONFIGS.values()):
        print(
            f"{language}: "
            + ", ".join(
                f"{split}={counts[(language, split)]} "
                f"({seconds_by_split_language[(language, split)] / 60:.1f} min)"
                for split in PILOT_SPLIT_COUNTS
            )
        )
    print("Audio checks: readable mono 16 kHz WAV, SHA-256, finite samples, "
          "80-bin Mel features, and CTC minimum frame count passed.")
    print("Transcript checks: non-empty NFC text and training-vocabulary coverage "
          f"{'passed' if not uncovered_characters else 'has gaps'}.")
    if uncovered_characters:
        print("Characters absent from the selected training transcripts:")
        for (language, split), characters in sorted(uncovered_characters.items()):
            rendered = ", ".join(
                f"{character!r} (U+{ord(character):04X})"
                for character in sorted(characters)
            )
            print(f"  {language} {split}: {rendered}")
    return {
        "rows": len(rows),
        "vocabulary_size": len(vocabulary.char_to_id),
        "counts": dict(counts),
        "minutes": dict(seconds_by_split_language),
        "vocabulary_coverage_ok": not uncovered_characters,
        "uncovered_characters": uncovered_characters,
    }


def finalize_staged_fleurs(staging_manifest):
    staging_manifest = Path(staging_manifest)
    staging_dir = staging_manifest.parent
    output_dir = PROJECT_ROOT / "data" / "fleurs"
    if output_dir.exists():
        raise FileExistsError(
            f"FLEURS output already exists and will not be overwritten: {output_dir}"
        )
    if not (staging_dir / "ATTRIBUTION.md").is_file():
        raise FileNotFoundError(
            f"FLEURS attribution record is missing from staging directory: {staging_dir}"
        )
    report = validate_fleurs_manifest(staging_manifest)
    if not report["vocabulary_coverage_ok"]:
        raise ValueError(
            "FLEURS pilot has evaluation characters absent from the selected "
            "training transcripts; staging data was not finalized"
        )
    staging_dir.rename(output_dir)
    print(f"Validated FLEURS staging data finalized at: {output_dir}")
    return output_dir / "manifest.csv"


class FleursManifestDataset(Dataset):
    def __init__(self, manifest_path, split):
        if split not in SOURCE_SPLITS:
            raise ValueError(
                f"Unsupported FLEURS split {split!r}; "
                f"use one of {sorted(SOURCE_SPLITS)}"
            )
        manifest_path, rows = _read_manifest(manifest_path)
        if not any(row["split"] == "train" for row in rows):
            raise ValueError("FLEURS manifest has no training transcripts")
        self.vocabulary = _read_training_vocabulary(manifest_path.parent)
        self.examples = []
        self.records = []
        self.language_counts = Counter()
        for row in rows:
            if row["split"] != split:
                continue
            if row["language"] not in LANGUAGE_CONFIGS.values():
                raise ValueError(f"Unsupported FLEURS language: {row['language']!r}")
            audio_path = _safe_audio_path(manifest_path.parent, row["audio_file"])
            waveform = load_audio(str(audio_path))
            features = extract_mel_spectrogram(waveform).squeeze(0)
            target = torch.tensor(
                self.vocabulary.encode(row["transcript"]), dtype=torch.long
            )
            required_frames = _minimum_ctc_frames(target)
            if features.shape[1] < required_frames:
                raise ValueError(
                    f"CTC target for source ID {row['source_id']} requires "
                    f"{required_frames} frames, got {features.shape[1]}"
                )
            self.examples.append((features, target))
            self.records.append(
                {
                    "audio_file": row["audio_file"],
                    "language": row["language"],
                    "transcript": row["transcript"],
                    "source_id": row["source_id"],
                }
            )
            self.language_counts[row["language"]] += 1
        if not self.examples:
            raise ValueError(f"No examples found for FLEURS split {split!r}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def decode_ctc(self, token_ids):
        return self.vocabulary.decode_ctc(token_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare or validate the separate FLEURS bilingual pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Download the official split archives and prepare 400 selected recordings.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing FLEURS manifest and its audio files.",
    )
    mode.add_argument(
        "--finalize-staged",
        action="store_true",
        help="Validate and finalize a complete staging directory without downloading.",
    )
    mode.add_argument(
        "--build-vocabulary",
        action="store_true",
        help="Build a character inventory from the official FLEURS train transcripts.",
    )
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "data" / "fleurs" / "manifest.csv"),
        help="Manifest path for validation; also sets the output root when preparing.",
    )
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if args.prepare:
        output_dir = manifest_path.parent
        prepare_fleurs(output_dir)
    elif args.build_vocabulary:
        build_training_vocabulary(manifest_path)
    elif args.finalize_staged:
        finalize_staged_fleurs(manifest_path)
    else:
        report = validate_fleurs_manifest(manifest_path)
        if not report["vocabulary_coverage_ok"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
