import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.whisper_inference import main


class WhisperInferenceTests(unittest.TestCase):
    def test_transcribes_each_language_with_multilingual_model(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "sample.wav"
            audio.touch()
            for language, text in (("en", " hello "), ("te", " నమస్కారం ")):
                with self.subTest(language=language):
                    model = Mock()
                    model.transcribe.return_value = {"text": text}
                    whisper = types.ModuleType("whisper")
                    whisper.load_model = Mock(return_value=model)
                    output = io.StringIO()
                    with patch.dict(sys.modules, {"whisper": whisper}), \
                            patch.object(sys, "argv", ["whisper_inference", str(audio), "--language", language]), \
                            contextlib.redirect_stdout(output):
                        main()
                    whisper.load_model.assert_called_once_with("small", device="cpu")
                    model.eval.assert_called_once_with()
                    model.transcribe.assert_called_once_with(
                        str(audio), language=language, task="transcribe", fp16=False
                    )
                    self.assertEqual(output.getvalue(), text.strip() + "\n")

    def test_invalid_audio_and_language_fail_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "sample.wav"
            audio.touch()
            whisper = types.ModuleType("whisper")
            whisper.load_model = Mock()
            for arguments in (
                [str(Path(directory) / "missing.wav"), "--language", "en"],
                [str(Path(directory) / "sample.mp3"), "--language", "en"],
                [str(audio), "--language", "fr"],
            ):
                with self.subTest(arguments=arguments), patch.dict(sys.modules, {"whisper": whisper}), \
                        patch.object(sys, "argv", ["whisper_inference", *arguments]), \
                        contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertNotEqual(error.exception.code, 0)
            whisper.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
