import os
import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample audio to 16 kHz
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform


def extract_mel_spectrogram(file_path):
    waveform = load_audio(file_path)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel = mel_transform(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel)

    return log_mel


if __name__ == "__main__":
    print("Audio preprocessing module is ready!")

    audio_path = "data/audio/Trimanpalli Bheemgal Road.m4a"

    if os.path.exists(audio_path):
        mel = extract_mel_spectrogram(audio_path)
        print("Audio loaded successfully!")
        print("Mel spectrogram shape:", mel.shape)
    else:
        print("Audio file not found:", audio_path)