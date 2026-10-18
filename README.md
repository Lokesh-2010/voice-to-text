# Bilingual English–Telugu Speech-to-Text

## Objective

Build and evaluate a small character-level speech recognizer for English (`en`)
and Telugu (`te`) using Python, PyTorch, a CNN-BiLSTM acoustic model and
Connectionist Temporal Classification (CTC) loss.

> **Honest status:** the model fits the 14 recordings it was trained on
> (training CER 0.004) but **does not generalize to new speech yet**: on two
> independent held-out recordings, WER is 1.00 and CER is 0.83. This is a
> working prototype of the full pipeline (preprocessing, training, decoding,
> evaluation), not a usable recognizer.

## Architecture

`src/model.py` defines `CNNBiLSTM`:

1. Input: log-Mel features of shape `[batch, 80, time]`.
2. Two 1D convolution blocks (Conv1d, GroupNorm, ReLU) with 128 channels.
   GroupNorm replaced BatchNorm because clips are processed one at a time.
   Older BatchNorm checkpoints are detected by `src.predict` and loaded
   through a legacy `normalization="batch"` option (inference only).
3. A two-layer bidirectional LSTM, 128 hidden units per direction.
4. A linear layer over the character vocabulary plus the CTC blank (ID 0).

Training uses `torch.nn.CTCLoss`. Inference uses greedy argmax, collapses
repeated tokens and removes blanks. There is no language model or beam search.

## Audio preprocessing

`src/audio_preprocessing.py` loads audio as **16 kHz mono** and computes an
80-bin Mel spectrogram followed by a natural log. Recordings are stored as
16 kHz mono WAV. On macOS, `.m4a` files can be converted with the built-in
`afconvert` (no FFmpeg needed):

```bash
afconvert data/audio/english_01.wav.m4a data/audio/english_01.wav \
  -f WAVE -d LEI16@16000 -c 1
```

Transcripts are NFC-normalized and split into individual characters.

## Dataset

Local data (recordings and held-out references are not published; see Data privacy):

```text
data/
├── audio/                      # recordings
└── transcripts/
    ├── transcripts.csv         # TRAINING rows (verified)
    └── heldout.csv             # independent test references (never trained on)
```

- **Training: 14 recordings, 7 English and 7 Telugu**, a few seconds each, from
  one speaker. The 70-token character vocabulary is built from these
  transcripts only.
- **Public copy of the data files:** the 14-recording model was trained on the
  full local transcript file. To avoid publishing personal details (the
  speaker's name and place names), the committed `transcripts.csv` lists only
  the 7 non-personal rows, and the checkpoint's stored `train_files` list has
  those filenames replaced by `redacted_clip`. The weights are unchanged; the
  model has still memorized the full training set, including the redacted
  sentences.
- **Independent test: 2 recordings**, `english_05.wav` and `telugu_05.wav`,
  referenced only in `heldout.csv`.
- `transcripts.csv` columns: `audio_file`, `language`, `transcript`,
  `audio_verified`, `transcript_verified`. The verification flags mean a person
  confirmed the audio and the transcript; the tools cannot check that.

## Setup

Python 3.9, macOS (the pinned versions were validated there):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # librosa, torch, torchaudio
```

## Quick start after cloning

The trained checkpoint is included in the repository
(`models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt`, about 3 MB; weights and
vocabulary only), so no download is needed. The training recordings are
**not** published, so you must supply your own 16 kHz mono WAV file:

```bash
python -m src.predict --self-test                 # no audio needed
python -m src.predict --audio path/to/your_clip.wav
```

Expect sensible output only for the original 14 training sentences recorded by
the original speaker. For other speech, accuracy is **not reliable yet** (see
Evaluation). Retraining needs your own recordings plus rows in
`data/transcripts/transcripts.csv`; the commands below show how.

## Validate the dataset

```bash
python -m src.verify_dataset
python -m src.train --check-data
```

## Train

Training refuses to overwrite an existing checkpoint; use a new path each run.
`--use-all-data` trains on every row of `transcripts.csv` (no validation/test
hold-out), which is how the 14-recording checkpoint was produced:

```bash
python -m src.train --train --use-all-data \
  --epochs 700 --batch-size 1 --learning-rate 0.001 \
  --checkpoint models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt
```

Without `--use-all-data`, `assign_splits()` holds out about 40% of each
language for validation and test.

**Result:** 700 epochs, final mean CTC loss **0.0200** (10.77 at epoch 1).
The loss was not monotonic (for example 0.25 at epoch 500), so the final
weights are not perfectly settled.

Saved checkpoint: `models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt`
(vocabulary, 80 Mel bins, blank ID and training file list are stored inside).

## Predict

The default checkpoint is the final 14-clip model; `--checkpoint` overrides it:

```bash
python -m src.predict --audio data/audio/english_02.wav
python -m src.predict \
  --audio data/audio/english_02.wav \
  --checkpoint models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt
```

## Evaluate (WER and CER)

`src/evaluate.py` scores the splits defined by `assign_splits()` and prints
`NOT INDEPENDENT` for any clip the checkpoint was trained on:

```bash
python -m src.evaluate \
  --checkpoint models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt --split all
```

The held-out recordings are scored by running real predictions and comparing
them with `data/transcripts/heldout.csv`:

```python
import csv
from src.predict import predict
from src.evaluate import error_counts

CKPT = "models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt"
for row in csv.DictReader(open("data/transcripts/heldout.csv", encoding="utf-8")):
    hyp = predict("data/audio/" + row["audio_file"], CKPT)
    we, wt, ce, ct = error_counts(row["transcript"], hyp)
    print(row["audio_file"], hyp, f"WER {we}/{wt} CER {ce}/{ct}")
```

## Results

Checkpoint: `cnn_bilstm_ctc_groupnorm_14clips_700ep.pt`.

### Training recordings (seen in training, not independent)

| Metric | Value |
|---|---|
| Exact matches | 12 of 14 |
| WER | 2/83 = 0.024 |
| CER | 2/518 = **0.004** |

Examples (exact matches):

| Clip | Expected = predicted |
|---|---|
| `english_02.wav` | Today I am learning Python programming. |
| `telugu_03.wav` | నాకు తెలుగు భాష అంటే చాలా ఇష్టం. |

The two imperfect clips differ by one character each: `speech-totext` for
`speech-to-text`, and `నివసిస్తన్నాను` for `నివసిస్తున్నాను`.

### Independent held-out recordings (never used for training)

| Clip | Expected | Predicted | WER | CER |
|---|---|---|---|---|
| `english_05.wav` | I enjoy playing cricket with my friends. | `Hel  అsపే amespuetearnui srigeinoith.` | 7/7 | 34/40 = 0.85 |
| `telugu_05.wav` | ఈ రోజు వాతావరణం చాలా బాగుంది. | `నేను హైదర వువటుదాన్నాను.` | 5/5 | 23/29 = 0.79 |
| **Combined** | | | **1.00** | **0.83** |

No word was recognized on either clip. The output mixes scripts and reuses
fragments of training sentences.

## What this means

The model **fits its training recordings but does not generalize to new
speech.** Fourteen short clips from one speaker are enough to memorize the
sentences, not to learn to recognize new words. Only the held-out numbers
above reflect unseen-speech accuracy.

Vocabulary limitation: the vocabulary contains only characters that occur in
the training transcripts, so the model cannot output any other character. The
held-out sentences need two characters it lacks: `j` ("enjoy") and `ణ`
(U+0C23, in "వాతావరణం"). This adds guaranteed errors, but the failure is
nearly total even without it.

## Project structure

```text
src/
├── audio_preprocessing.py    # 16 kHz mono loading, log-Mel features
├── model.py                  # CNNBiLSTM
├── train.py                  # dataset, vocabulary, CTC training
├── predict.py                # checkpoint loading, greedy CTC decoding
├── evaluate.py               # WER/CER
├── verify_dataset.py         # CSV/audio checks
├── diagnose_prediction.py    # frame-level prediction inspection
└── overfit_test.py           # single-clip memorization debugging aid
data/                         # recordings are local and git-ignored; transcripts.csv is included
models/                       # only the final checkpoint is tracked (other *.pt are git-ignored)
results/inference_demo.md     # 2 training + 2 unseen clips (unseen ones fail)
results/english_prediction.log  # raw src.predict CLI output for those 4 clips
requirements.txt
```

Checkpoint location: `models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt`.
Earlier checkpoints in `models/` (`cnn_bilstm_ctc_groupnorm_700ep.pt`,
`cnn_bilstm_ctc_groupnorm_targets_700ep.pt`) were trained on 4 and 6 clips.

Self-tests that need no data: `python -m src.train --self-test`,
`python -m src.predict --self-test`, `python -m src.evaluate --self-test`.

## Data privacy

Voice recordings (`data/audio/`), the held-out references
(`data/transcripts/heldout.csv`) and all model weights except the final
checkpoint are excluded by `.gitignore`. The training transcript file
`data/transcripts/transcripts.csv` and the ~3 MB final checkpoint
`models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt` (weights and vocabulary
only, no audio) are included. Because the audio is not published, the training
data cannot be reproduced without your own recordings. Obtain permission before
sharing voice recordings.

## Future improvements

- **Larger dataset:** many more hours of varied, manually verified speech.
- **More speakers:** different voices, accents, microphones and noise levels.
- **Vocabulary coverage:** train on text covering the full English alphabet and
  Telugu character set so no character is unreachable.
- **Better generalization:** data augmentation, a held-out validation split
  with early stopping, a lower learning rate or scheduler to avoid late loss
  spikes, a larger or pretrained acoustic model, and beam search with a
  language model.
- **Evaluation:** a proper train/validation/test protocol with several
  held-out speakers instead of two clips.
