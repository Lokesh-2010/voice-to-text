"""Transcribe a WAV file locally with the pretrained multilingual Whisper model."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Path to a WAV recording")
    parser.add_argument("--language", required=True, choices=("en", "te"), help="Spoken language")
    args = parser.parse_args()

    if args.audio.suffix.lower() != ".wav":
        parser.error("audio must be a WAV file")
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")

    import whisper

    model = whisper.load_model("small", device="cpu")
    model.eval()
    result = model.transcribe(str(args.audio), language=args.language, task="transcribe", fp16=False)
    print(result["text"].strip())


if __name__ == "__main__":
    main()
