from pathlib import Path
import wave


audio_file = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "audio"
    / "telugu"
    / "telugu_test.wav"
)

with wave.open(str(audio_file), "rb") as audio:
    sample_rate = audio.getframerate()
    channels = audio.getnchannels()
    total_frames = audio.getnframes()
    duration = total_frames / sample_rate

print(f"Sample rate: {sample_rate} Hz")
print(f"Channels: {channels}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")
