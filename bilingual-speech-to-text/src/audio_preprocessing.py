import torch
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert audio to 16 kHz
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform

def extract_mel_spectrogram(waveform):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )
    mel = mel_transform(waveform)
    return torch.log(mel + 1e-9)

if __name__ == "__main__":
    print("Audio preprocessing module is ready!")