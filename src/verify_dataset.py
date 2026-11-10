import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "transcripts" / "transcripts.csv"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
REQUIRED_COLUMNS = ("audio_file", "language", "transcript")
VERIFICATION_COLUMNS = ("audio_verified", "transcript_verified")
ALLOWED_COLUMNS = (*REQUIRED_COLUMNS, *VERIFICATION_COLUMNS)
SUPPORTED_LANGUAGES = {"en", "te"}
AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac"}
ALTERNATE_AUDIO_FILES = {
    "Tirmanpalli Bheemgal Road.m4a": "Tirmanpalli Bheemgal Road.wav",
}


def main():
    if not CSV_PATH.is_file():
        print(f"ERROR: transcript CSV not found: {CSV_PATH}")
        return 2
    if not AUDIO_DIR.is_dir():
        print(f"ERROR: audio directory not found: {AUDIO_DIR}")
        return 2

    audio_files = {
        path.name
        for path in AUDIO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    }
    issues = 0
    referenced_files = []
    not_ready_rows = []
    ready_language_counts = {}

    print("Dataset verification report")
    print(f"CSV: {CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Audio directory: {AUDIO_DIR.relative_to(PROJECT_ROOT)}")

    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = tuple(reader.fieldnames or ())
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
        unsupported_columns = [column for column in columns if column not in ALLOWED_COLUMNS]
        if missing_columns:
            issues += len(missing_columns)
            print(f"Missing required CSV columns: {', '.join(missing_columns)}")
        if unsupported_columns:
            issues += len(unsupported_columns)
            print(f"Unsupported CSV columns: {', '.join(unsupported_columns)}")
        if not missing_columns and not unsupported_columns:
            print(f"CSV columns: {', '.join(columns)}")

        verification_columns = set(columns) & set(VERIFICATION_COLUMNS)
        absent_verification_columns = [
            column for column in VERIFICATION_COLUMNS if column not in verification_columns
        ]
        if absent_verification_columns:
            print(
                "Verification columns missing; affected rows cannot be ready "
                f"for training: {', '.join(absent_verification_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            reasons = []
            if None in row:
                issues += 1
                reasons.append("extra value(s) beyond the declared CSV columns")
                print(f"Row {row_number}: extra value(s) beyond the CSV columns")

            audio_file = (row.get("audio_file") or "").strip()
            language = (row.get("language") or "").strip()
            transcript = (row.get("transcript") or "").strip()

            if not audio_file:
                issues += 1
                reasons.append("missing audio_file value")
                print(f"Row {row_number}: missing audio_file value")
            elif Path(audio_file).name != audio_file:
                issues += 1
                reasons.append("audio_file must be a filename, not a path")
                print(
                    f"Row {row_number}: audio_file must be a filename, "
                    f"not a path: {audio_file!r}"
                )
            else:
                referenced_files.append(audio_file)
                if audio_file not in audio_files:
                    issues += 1
                    reasons.append("matching audio file not found in data/audio/")
                    print(
                        f"Row {row_number}: audio file not found in data/audio/: "
                        f"{audio_file!r}"
                    )

            if not transcript:
                issues += 1
                reasons.append("blank transcript")
                print(f"Row {row_number}: blank transcript")

            if not language:
                issues += 1
                reasons.append("missing language value")
                print(f"Row {row_number}: missing language value")
            elif language.lower() not in SUPPORTED_LANGUAGES:
                issues += 1
                reasons.append(f"unsupported language label {language!r}")
                print(
                    f"Row {row_number}: unsupported language code {language!r}; "
                    "expected 'en' or 'te'"
                )

            for column in VERIFICATION_COLUMNS:
                if column not in verification_columns:
                    reasons.append(f"missing {column} column")
                elif (row.get(column) or "").strip().lower() != "true":
                    reasons.append(f"{column} is not true")
                    print(
                        f"Row {row_number}: {column} is not true; "
                        "verification must be performed manually"
                    )

            if reasons:
                not_ready_rows.append((row_number, reasons))
            else:
                normalized_language = language.lower()
                ready_language_counts[normalized_language] = (
                    ready_language_counts.get(normalized_language, 0) + 1
                )

    duplicates = sorted(
        filename
        for filename, count in Counter(referenced_files).items()
        if count > 1
    )
    for filename in duplicates:
        issues += 1
        print(f"Duplicate CSV audio_file entry: {filename!r}")

    referenced_file_set = set(referenced_files)
    recognized_alternates = {
        alternate
        for alternate, canonical in ALTERNATE_AUDIO_FILES.items()
        if alternate in audio_files and canonical in referenced_file_set
    }
    for alternate in sorted(recognized_alternates):
        print(
            f"Recognized alternate audio format: {alternate!r} "
            f"for referenced recording {ALTERNATE_AUDIO_FILES[alternate]!r}"
        )

    unreferenced_files = sorted(
        audio_files - referenced_file_set - recognized_alternates
    )
    issues += len(unreferenced_files)
    for filename in unreferenced_files:
        print(f"Audio file has no CSV row: {filename!r}")

    missing_languages = sorted(SUPPORTED_LANGUAGES - set(ready_language_counts))
    if missing_languages:
        issues += len(missing_languages)
        print(
            "Bilingual training coverage is missing verified "
            f"sample(s) for: {', '.join(missing_languages)}"
        )

    print(
        f"Rows passing verifier checks: {sum(ready_language_counts.values())}; "
        f"rows not ready: {len(not_ready_rows)}"
    )
    for row_number, reasons in not_ready_rows:
        print(f"  Row {row_number}: {'; '.join(reasons)}")
    if not not_ready_rows and not missing_languages:
        print("All CSV rows meet the row-level checks and both languages are covered.")
    elif not not_ready_rows:
        print("All rows meet row-level checks, but bilingual training is not ready.")
    print(
        f"Summary: {len(referenced_files)} audio reference(s), "
        f"{len(duplicates)} duplicate filename(s), "
        f"{len(unreferenced_files)} audio file(s) without a CSV row, "
        f"{issues} issue(s)."
    )
    print(
        "This report checks structure and verification flags only; it does not "
        "verify that transcript text or language labels match the recordings."
    )
    return 1 if issues or not_ready_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
