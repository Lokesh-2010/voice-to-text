# English–Telugu Speech-to-Text

A small character-level speech recognition project for English (`en`) and
Telugu (`te`). It uses a CNN-BiLSTM acoustic model trained with
Connectionist Temporal Classification (CTC), and greedy CTC decoding at
inference time.

> **Current status:** this is a prototype, not a production-ready recognizer.
> The checked-in source does not include model weights or recordings. In the
> local validation performed for this submission, the dataset contained only
> four utterances (two per language). The submission checkpoint predicted blank
> on 99.20% of frames for one Telugu sample and decoded it as `ు?`. A separate
> one-utterance overfit test did learn its training sample exactly. These
> results demonstrate the training/inference path, not generalization or
> transcription accuracy.

## Architecture

`src/model.py` defines `CNNBiLSTM`:

1. Input log-Mel features have shape `[batch, 80, time]`.
2. Two stride-1, padded 1D convolution blocks (Conv1d, BatchNorm1d, ReLU)
   produce 128 channels while preserving the number of time steps.
3. A two-layer, bidirectional LSTM with 128 hidden units per direction models
   the sequence.
4. A linear layer emits per-frame logits over the character vocabulary,
   including the CTC blank token.

`src/audio_preprocessing.py` loads mono audio at 16 kHz and calculates 80-bin
Mel spectrograms, followed by a natural logarithm. Transcripts are NFC Unicode
normalized and represented as individual characters. The blank symbol
`<blank>` has ID 0. Training uses `torch.nn.CTCLoss`; inference uses greedy
argmax followed by CTC blank removal and repeated-token collapse.

## Requirements and setup

The pinned direct dependencies were validated in a Python 3.9 environment on
macOS. PyTorch and torchaudio should use matching versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

The training loader expects local data at:

```text
data/
├── audio/
│   └── <audio files>
└── transcripts/
    └── transcripts.csv
```

The CSV requires the columns `audio_file`, `language`, and `transcript`. The
optional `audio_verified` and `transcript_verified` columns are required for
training: each must be set to `true` only after a person has checked that the
audio is usable and its transcript matches what was spoken. Supported language
codes are `en` and `te`. `audio_file` should be a filename located directly
under `data/audio/`.

The local validation run found four rows, two per language, and all referenced
audio loaded successfully. The dataset verifier checks file existence,
readability, language codes, and verification flags; it cannot confirm that
the spoken content matches the transcript. Add substantially more varied,
manually checked recordings before interpreting model quality.

Local recordings, transcript CSVs, and model weight files are excluded by
`.gitignore` to reduce accidental publication of personal data and generated
artifacts. Obtain permission before sharing any voice recordings or
transcripts. Provide your own local data using the layout above.

## Validate the dataset

From the project root, with the virtual environment active:

```bash
python -m src.verify_dataset
python -m src.train --check-data
```

`verify_dataset` checks the CSV structure, supported labels, file references,
and verification flags. `--check-data` additionally attempts to load and
preprocess audio without starting training.

## Train

Training refuses to overwrite an existing checkpoint. Use a new path for each
run:

```bash
python -m src.train --train \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.001 \
  --checkpoint results/checkpoints/my_run.pt
```

The script prints the mean CTC loss each epoch and saves the trained model
state together with its character vocabulary, Mel-bin count, and blank ID.
The current dataset is extremely small, so lower training loss or exact
training-set transcription should not be interpreted as held-out accuracy.

## Predict

Pass the checkpoint explicitly to avoid accidentally using the default
`models/cnn_bilstm_ctc.pt`:

```bash
python -m src.predict \
  --audio data/audio/telugu_sample_2.wav.m4a \
  --checkpoint results/checkpoints/my_run.pt
```

The prediction printed by the CLI is an unverified greedy decode. Greedy
decoding can return an empty string if blank is the highest-scoring class at
every frame.

## Diagnostics and tests

Inspect frame-level predictions for an audio/checkpoint pair:

```bash
python -m src.diagnose_prediction \
  --checkpoint results/checkpoints/my_run.pt \
  --audio data/audio/telugu_sample_2.wav.m4a
```

Run synthetic CTC/model checks:

```bash
python -m src.train --self-test
python -m src.predict --self-test
```

The training self-test checks output shapes, CTC loss, and backward propagation.
The prediction self-test checks CTC repeated-token and blank handling.

An optional single-utterance overfit test trains only on one verified row,
stops at an exact greedy match (or after the step limit), and writes a new
checkpoint only if its output path does not already exist:

```bash
python -m src.overfit_test \
  --audio-file enlish_sample.m4a \
  --checkpoint results/checkpoints/overfit_test.pt
```

In the local test run, the model exactly reproduced the target
`మీరు ఎలా ఉన్నారు?` from `enlish_sample.m4a` after 625 steps with CTC loss
0.0393. Separately, the submission checkpoint produced 374 blank frames out
of 377 (99.20%) for `telugu_sample_2.wav.m4a`, decoding to `ు?`. These are
local observations on the current four-example dataset, not a benchmark.

## Limitations

- Only four verified rows were present during local checks; this is not enough
  data to train a robust English–Telugu recognizer.
- There is no held-out validation or test split, so generalization has not
  been measured.
- The current submission checkpoint is blank-dominated on the tested Telugu
  utterance.
- The model is character-level and uses greedy decoding only; no language model,
  beam search, normalization beyond NFC, or punctuation post-processing is
  included.
- Verification flags and file checks do not establish transcript accuracy;
  each audio/transcript pair should be reviewed by a fluent speaker.
- Training is CPU-based in the current pipeline and runtime grows with audio
  length and the number of epochs.

## Project layout

```text
src/
├── audio_preprocessing.py
├── diagnose_prediction.py
├── model.py
├── overfit_test.py
├── predict.py
├── train.py
└── verify_dataset.py
data/
├── audio/                 # local, ignored recordings
└── transcripts/           # local, ignored transcript CSV
```
