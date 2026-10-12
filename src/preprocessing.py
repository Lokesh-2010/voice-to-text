

import soundfile as sf
import numpy as np
import torch
import torchaudio

def load_audio(file_path, target_sample_rate=16000):
    """Load audio and convert it to mono at 16 kHz."""

    audio, sample_rate = sf.read(file_path)

    # Convert stereo audio to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert NumPy array to a PyTorch tensor
    audio = torch.tensor(audio, dtype=torch.float32)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )

    return audio

def extract_mel_spectrogram(audio, sample_rate=16000):
    """Convert audio into Mel spectrogram features."""

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        hop_length=160,
        n_mels=80,
    )

    mel = mel_transform(audio)

    # Log scaling helps represent a wide range of energies
    mel = torch.log(mel.clamp(min=1e-9))

    return mel

if __name__ == "__main__":
    print("Audio preprocessing module is ready.")
