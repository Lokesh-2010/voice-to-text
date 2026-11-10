import torch
import librosa
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80

def load_audio(file_path):
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return torch.from_numpy(waveform).unsqueeze(0)

def extract_mel_spectrogram(waveform):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS
    )
    mel = mel_transform(waveform)
    return torch.log(mel + 1e-9)

if __name__ == "__main__":
    print("Audio preprocessing module is ready!")