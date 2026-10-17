# English–Telugu Speech-to-Text

A small character-level speech recognition project for English (`en`) and
Telugu (`te`). It uses a CNN-BiLSTM acoustic model trained with
Connectionist Temporal Classification (CTC), and greedy CTC decoding at
inference time.

> **Current status:** this is a prototype, not a production-ready recognizer.
> The dataset contains only **four verified recordings** (two English, two
> Telugu, about 15 seconds in total). A checkpoint trained on them reproduces
> its own training sentences, but it **does not generalize**: on a new English
> sentence it recognized 0 of 12 words. The results below demonstrate that the
> preprocessing, training, decoding and evaluation pipeline works end to end.
> They are not evidence of transcription accuracy on unseen speech. The source
> repository does not include recordings, transcripts or model weights.

## Architecture

`src/model.py` defines `CNNBiLSTM`:

1. Input log-Mel features have shape `[batch, 80, time]`.
2. Two stride-1, padded 1D convolution blocks (Conv1d, GroupNorm, ReLU)
   produce 128 channels while preserving the number of time steps. GroupNorm
   (one group) replaced BatchNorm because the model runs one clip at a time:
   BatchNorm used per-clip statistics in training but mismatched running
   averages at inference, which broke predictions. GroupNorm behaves
   identically in training and inference. Checkpoints trained with the
   earlier BatchNorm version cannot be loaded.
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

The local dataset has four rows, two per language, and all referenced audio
loads successfully. The dataset verifier checks file existence, readability,
language codes, and verification flags; it cannot confirm that the spoken
content matches the transcript. Add substantially more varied, manually
checked recordings before interpreting model quality.

### Train / validation / test split

`assign_splits()` in `src/train.py` splits the verified rows deterministically
(no randomness) and per language: rows are sorted by filename, about 40% of
each language (at least one row) is held out, and the held-out rows are dealt
alternately to validation and test. With the current four rows the split is:

| Split | Recordings |
|---|---|
| train | `Tirmanpalli Bheemgal Road.wav` (en), `enlish_sample.m4a` (te) |
| validation | `english_sample_2.wav.m4a` (en) |
| test | `telugu_sample_2.wav.m4a` (te) |

`--train` trains only on the train split and stores the list of training
files in the checkpoint. The vocabulary is still built from all verified
transcripts, so characters that occur only in held-out rows are in the output
set even though the model never saw them in training.

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
run. The default of 100 epochs is too few for this dataset (loss stays near the
blank-only plateau); a 700-epoch run on all four clips reached a mean CTC loss
of about 0.04:

```bash
python -m src.train --train \
  --epochs 700 \
  --batch-size 1 \
  --learning-rate 0.001 \
  --checkpoint models/my_run.pt
```

The script prints the train/validation/test split and the mean CTC loss each
epoch, and saves the model state together with its character vocabulary,
Mel-bin count, blank ID, and training-file list. Trained weights (`*.pt`) are
git-ignored. Lower training loss or exact training-set transcription should
not be interpreted as held-out accuracy.

## Predict

Pass the checkpoint explicitly. The default, `models/cnn_bilstm_ctc.pt`, is an
old BatchNorm checkpoint that no longer loads:

```bash
python -m src.predict \
  --audio data/audio/telugu_sample_2.wav.m4a \
  --checkpoint models/my_run.pt
```

The prediction printed by the CLI is an unverified greedy decode. Greedy
decoding can return an empty string if blank is the highest-scoring class at
every frame.

## Evaluate (WER and CER)

`src/evaluate.py` decodes each split with a checkpoint and reports per-clip and
corpus word error rate (WER, word-level edit distance) and character error
rate (CER, character-level edit distance) on NFC-normalized text:

```bash
python -m src.evaluate --checkpoint models/my_run.pt            # all splits
python -m src.evaluate --checkpoint models/my_run.pt --split test
```

Any clip that the checkpoint was trained on is labeled `SEEN IN TRAINING` and
the split is reported as `NOT INDEPENDENT`. Checkpoints that do not record
their training files are treated as having seen every clip.

## Results

### Familiar-sentence demonstrations (not independent accuracy)

These outputs come from `results/inference_demo.md`, generated with
`src.predict` and the 700-epoch GroupNorm checkpoint
`models/cnn_bilstm_ctc_groupnorm_700ep.pt`, which was trained on **all four**
recordings. Every clip below was in its training data, so the matches show
memorization, not recognition of new speech. The reference is the verified
transcript from the CSV; the prediction is model output.

| Language | Audio | Reference | Prediction | Exact match |
|---|---|---|---|---|
| English | `english_sample_2.wav.m4a` | My name is Lokesh. Today I am testing my speech recognition model. | My name is Lokesh. Today I am testing my speech recognition model. | yes |
| English | `Tirmanpalli Bheemgal Road.wav` | Hello, how are you? Today I am learning speech-to-text. | Hello, how are you? Today I am learning speeto-text. | no |
| Telugu | `telugu_sample_2.wav.m4a` | నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను. | నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను. | yes |
| Telugu | `enlish_sample.m4a` | మీరు ఎలా ఉన్నారు? | మీరు ఎలా ఉన్నారు? | yes |

3 of 4 clips match exactly. The one error drops "ch" from "speech".

### Unseen sentence

The same checkpoint was run on a new English recording (not in the dataset,
spoken sentence: "The weather is beautiful today and I want to learn
something new."). The output was
`M లే ోక్ ోడల్్ లోక్ ప్  మక్ మnిస్ిస్నాను.`: 0 of 12 words correct (CER 0.86,
WER 1.00), mostly Telugu characters and fragments of the training sentences.
This is the realistic indication of the current model's ability on new speech.

### WER/CER on the splits

`python -m src.evaluate` was run on the 700-epoch checkpoint above, but that
checkpoint trained on all four clips, so its validation and test scores
(WER 0.000, CER 0.000 on one clip each) are **not independent** and are not
reported as accuracy. No independent WER/CER has been measured. Obtaining it
requires training on the train split only and evaluating on the held-out
clips, and even then each split holds a single recording, which is far too
little for a meaningful estimate.

### `results/english_prediction.log` is outdated

`results/english_prediction.log` contains `M.`. It is the output of an earlier,
under-trained checkpoint (`results/checkpoints/submission.pt`, about 100
epochs) that emitted blank almost everywhere and printed only a first and last
character. It does not reflect the current model, and that checkpoint cannot
be loaded by the current code. Use `results/inference_demo.md` for current
outputs.

## Diagnostics and tests

Inspect frame-level predictions for an audio/checkpoint pair:

```bash
python -m src.diagnose_prediction \
  --checkpoint models/my_run.pt \
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

The overfit test is a debugging aid only; it demonstrates that the model can
memorize one clip, which says nothing about generalization.

## Limitations

- **Only four verified recordings** (two English, two Telugu, about 15 seconds)
  exist. This is not enough data to train a usable English–Telugu recognizer.
- **The model does not generalize.** It reproduces its training sentences but
  failed on a new English sentence (0 of 12 words correct, CER 0.86).
- **No independent accuracy has been measured.** The only retained checkpoint
  that works was trained on all four clips, so its WER/CER on any split is not
  independent. Each split contains a single recording, so even a properly
  held-out score would have no statistical meaning.
- The 3-of-4 exact matches in the demonstration are on training clips.
- Old checkpoints (`results/checkpoints/*.pt`, most of `models/*.pt`) were
  trained with BatchNorm and cannot be loaded by the current GroupNorm model.
- The model is character-level and uses greedy decoding only; no language model,
  beam search, normalization beyond NFC, or punctuation post-processing is
  included.
- Verification flags and file checks do not establish transcript accuracy;
  each audio/transcript pair should be reviewed by a fluent speaker. A
  language-ID check agreed with the labels, but the Telugu sample had only
  about 49% confidence.
- Training is CPU-based and unseeded, so repeated runs give slightly different
  results; runtime grows with audio length and the number of epochs.

## Project layout

```text
src/
├── audio_preprocessing.py
├── diagnose_prediction.py
├── evaluate.py
├── model.py
├── overfit_test.py
├── predict.py
├── train.py
└── verify_dataset.py
data/
├── audio/                 # local, ignored recordings
└── transcripts/           # local, ignored transcript CSV
models/                    # checkpoints (*.pt are git-ignored)
results/
├── inference_demo.md      # current English/Telugu demonstration outputs
└── english_prediction.log # OUTDATED: "M." from an earlier checkpoint
```
