from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 80

def load_audio(audio_file: Path) -> np.ndarray:
    waveform, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_file}")
    return waveform


def extract_log_mel(waveform: np.ndarray) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS
    )
    return librosa.power_to_db(mel_spectrogram)


if __name__ == "__main__":
    audio_file = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "audio"
        / "telugu"
        / "telugu_test.wav"
    )
    waveform = load_audio(audio_file)
    log_mel_spectrogram = extract_log_mel(waveform)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Waveform samples: {waveform.shape[0]}")
    print(f"Log-Mel feature shape: {log_mel_spectrogram.shape}")
