import os
import torch
import torchaudio

from .audio_preprocessing import load_audio

SAMPLE_RATE = 16000
N_MELS = 80


def extract_mel_spectrogram(file_path):
    waveform = load_audio(file_path)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel = mel_transform(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel)

    return log_mel


from pathlib import Path
import librosa

# Find the project directory automatically
project_dir = Path(__file__).resolve().parent.parent

# Locate the actual audio file
audio_path = project_dir / "data" / "audio" / "Timnapalli Bheemangal Road.m4a"

if not audio_path.exists():
    print("Audio file not found:", audio_path)
else:
    print("Audio file found:", audio_path)

    y, sr = librosa.load(str(audio_path), sr=16000)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=80
    )

    print("Audio loaded successfully!")
    print("Sample rate:", sr)
    print("Mel spectrogram shape:", mel.shape)