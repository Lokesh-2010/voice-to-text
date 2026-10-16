# Model architecture and data flow

The inference/training path is:

```text
Audio
  → mono resampling to 16 kHz
  → 80-bin Mel spectrogram
  → log-Mel features
  → CNN
  → BiLSTM
  → per-frame character logits
  → CTC
  → text
```

## Audio preprocessing

`src/audio_preprocessing.py` loads audio as mono at 16,000 samples per second
and returns a waveform tensor. `torchaudio.transforms.MelSpectrogram`
converts it to 80 Mel bins; the implementation then applies
`log(mel + 1e-9)`. A single input has feature shape `[1, 80, time]`.

## CNN feature extraction

`src/model.py` applies two 1D convolution blocks across time:

1. Conv1d, 80 input channels to 128 output channels, kernel size 5, padding 2.
2. BatchNorm1d and ReLU.
3. Conv1d, 128 input/output channels, kernel size 3, padding 1.
4. BatchNorm1d and ReLU.

Both convolutions have stride 1 and padding that preserves the number of
frames. The CNN transforms each frame into a 128-dimensional feature vector.

## Bidirectional sequence model

The CNN output is transposed to `[batch, time, 128]` and passed through a
two-layer bidirectional LSTM with 128 hidden units per direction and 0.2
inter-layer dropout. Packed sequences use the per-audio input lengths so
padding does not enter the recurrent computation. The classifier maps each
256-dimensional output frame to one logit per vocabulary class.

## CTC training and decoding

Transcripts are normalized to Unicode NFC and encoded as individual
characters. `<blank>` uses token ID 0. Training applies log-softmax to the
frame logits, transposes them to `[time, batch, vocabulary]`, and computes
`torch.nn.CTCLoss` using the model's output lengths and target lengths.

CTC sums over possible alignments between frame-level predictions and the
target text, allowing the model to learn without frame-level character
boundaries. At inference, greedy decoding chooses the highest-logit class at
each frame, removes blank tokens, and collapses adjacent repeated token IDs.
This decoder is simple and may perform poorly when blank dominates or when
character predictions are uncertain.

## Data flow summary

For an input audio waveform with `T` Mel frames:

```text
waveform                   [1, samples]       at 16 kHz
log-Mel features            [1, 80, T]
CNN frame features           [1, T, 128]
BiLSTM representations       [1, T, 256]
classifier logits            [1, T, vocabulary_size]
greedy CTC token sequence    [T]
decoded character string     text
```

The CNN-BiLSTM and CTC pipeline is functional enough to overfit one verified
utterance. The current small-data checkpoint remains blank-dominated on a
tested Telugu recording; this architecture description is not a claim of
reliable English or Telugu recognition.
