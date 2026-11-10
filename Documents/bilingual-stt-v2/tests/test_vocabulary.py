import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.vocabulary import BLANK_ID, UNKNOWN_ID, Vocabulary, build_vocabulary


class VocabularyTests(unittest.TestCase):
    def test_training_only_round_trip_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcripts.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["transcript", "split"])
                writer.writerows([("Hello", "train"), ("నేను", "train"), ("Z", "val"), ("Q", "test")])

            vocabulary = build_vocabulary(path)
            self.assertEqual(vocabulary.id_to_char[BLANK_ID], "<blank>")
            self.assertEqual(vocabulary.id_to_char[UNKNOWN_ID], "<unk>")
            self.assertEqual(vocabulary.decode(vocabulary.encode("Hello")), "Hello")
            self.assertEqual(vocabulary.decode(vocabulary.encode("నేను")), "నేను")
            self.assertEqual(vocabulary.encode("ZQ"), [UNKNOWN_ID, UNKNOWN_ID])
            self.assertEqual(vocabulary.decode(vocabulary.encode("Z")), "<unk>")
            self.assertEqual(vocabulary.decode([BLANK_ID]), "<blank>")
            self.assertEqual(vocabulary.id_to_char[2:], sorted(set("Helloనేను")))

            output = Path(directory) / "vocab.json"
            vocabulary.save(output)
            loaded = Vocabulary.load(output)
            self.assertEqual(loaded.char_to_id, vocabulary.char_to_id)
            self.assertEqual(loaded.id_to_char, vocabulary.id_to_char)
            with self.assertRaises(FileExistsError):
                vocabulary.save(output)
            data = json.loads(output.read_text(encoding="utf-8"))
            data["char_to_id"]["<blank>"] = 1
            output.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                Vocabulary.load(output)
            with self.assertRaises(ValueError):
                vocabulary.decode([-1])

    def test_missing_or_empty_training_data_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcripts.csv"
            path.write_text("transcript,split\nHello,val\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_vocabulary(path)
            path.write_text("transcript,split\n,train\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_vocabulary(path)


if __name__ == "__main__":
    unittest.main()
