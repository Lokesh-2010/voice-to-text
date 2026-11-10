"""Build and use a deterministic character vocabulary for CTC."""

import csv
import json
from pathlib import Path

BLANK_TOKEN = "<blank>"
UNKNOWN_TOKEN = "<unk>"
BLANK_ID = 0
UNKNOWN_ID = 1


class Vocabulary:
    """Character IDs use Unicode code points, without normalization.

    Decoding converts IDs directly to tokens: it does not collapse CTC
    repetitions or remove blanks. Unknown characters decode as "<unk>".
    """

    def __init__(self, characters: list[str]) -> None:
        if any(len(character) != 1 for character in characters):
            raise ValueError("Vocabulary characters must be single Unicode code points")
        if characters != sorted(set(characters)):
            raise ValueError("Vocabulary characters must be unique and sorted")
        self.id_to_char = [BLANK_TOKEN, UNKNOWN_TOKEN, *characters]
        self.char_to_id = {character: index for index, character in enumerate(self.id_to_char)}

    def __len__(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(character, UNKNOWN_ID) for character in text]

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for index in ids:
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self):
                raise ValueError(f"Invalid vocabulary ID: {index!r}")
            tokens.append(self.id_to_char[index])
        return "".join(tokens)

    def save(self, path: Path) -> None:
        """Create a vocabulary file without overwriting an existing file."""
        with path.open("x", encoding="utf-8") as file:
            json.dump(
                {"char_to_id": self.char_to_id, "id_to_char": self.id_to_char},
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Vocabulary file must contain a JSON object")
        tokens = data.get("id_to_char")
        if (
            not isinstance(tokens, list)
            or len(tokens) < 2
            or tokens[:2] != [BLANK_TOKEN, UNKNOWN_TOKEN]
            or not all(isinstance(token, str) for token in tokens)
        ):
            raise ValueError("Invalid vocabulary token list")
        vocabulary = cls(tokens[2:])
        if data.get("char_to_id") != vocabulary.char_to_id:
            raise ValueError("Vocabulary mappings do not agree")
        return vocabulary


def build_vocabulary(csv_path: Path) -> Vocabulary:
    """Build a vocabulary from non-empty training transcripts only."""
    characters: set[str] = set()
    training_rows = 0
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"split", "transcript"} <= set(reader.fieldnames):
            raise ValueError("CSV must contain split and transcript columns")
        for row in reader:
            if row["split"] == "train":
                transcript = row["transcript"]
                if transcript is None or not transcript.strip():
                    raise ValueError("Training transcript is empty")
                characters.update(transcript)
                training_rows += 1
    if not training_rows:
        raise ValueError("CSV contains no training transcripts")
    return Vocabulary(sorted(characters))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    vocabulary = build_vocabulary(project_root / "data" / "transcripts.csv")
    output = project_root / "models" / "vocab.json"
    vocabulary.save(output)
    print(f"Saved {len(vocabulary)} tokens to {output}")
    print(f"CTC blank ID: {BLANK_ID}; unknown character ID: {UNKNOWN_ID}")
