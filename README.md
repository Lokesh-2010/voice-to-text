# English–Telugu Speech-to-Text

A small character-level speech-recognition prototype for English (`en`) and
Telugu (`te`). It uses a CNN-BiLSTM acoustic model, Connectionist Temporal
Classification (CTC) training, and greedy CTC decoding.

> **Status and results:** this is an experimental prototype, not a reliable
> bilingual transcription system. The local dataset contains only four
> verified utterances (two per language), which is insufficient for meaningful
> training or evaluation. The one-sample Telugu overfit checkpoint exactly
> decoded its training recording, but failed on all three other dataset
> recordings (100% WER for each). The diagnostic checkpoint predicted blank
> on all 377 frames of a Telugu recording and decoded it as an empty string.
> These results demonstrate sample memorization, not reliable English or
> Telugu recognition.

## Requirements and installation

Python 3.9 or newer is recommended. PyTorch and torchaudio are pinned to the
same version in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Training expects audio and a transcript CSV in this local layout:

```text
data/
├── audio/
│   └── <audio files>
└── transcripts/
    └── transcripts.csv
```

The CSV needs `audio_file`, `language`, and `transcript` columns. Training also
requires `audio_verified` and `transcript_verified` columns set to `true` only
after manually checking each recording and its transcript. Language labels
must be `en` or `te`; each `audio_file` is a filename relative to
`data/audio/`.

Local recordings, transcripts, and model checkpoints are excluded by
`.gitignore`, so they are not included when this project is submitted or
cloned. Obtain consent before sharing speech recordings or transcripts. The
verification tool checks file structure, readability, language labels, and
verification flags; it cannot determine whether spoken words match the
transcript.

The current local development dataset has four rows: two `en` and two `te`.
For local training or validation, place the reviewed recordings and CSV at the
paths above. The CSV must use the required columns and include
`audio_verified=true` and `transcript_verified=true` only after manually
checking each pair. The four-utterance dataset is too small to train a useful
recognizer or provide a held-out evaluation set.

Check local data before training:

```bash
python -m src.verify_dataset
python -m src.train --check-data
```

## Train

Use a new checkpoint filename for each run; training refuses to overwrite an
existing checkpoint. This command creates
`results/checkpoints/submission_run.pt` locally; no checkpoint is included in
the submission.

```bash
python -m src.train --train \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.001 \
  --checkpoint results/checkpoints/submission_run.pt
```

The training script reports mean CTC loss per epoch and saves the model state,
character vocabulary, Mel-bin count, and CTC blank ID. Lower training loss is
not a substitute for held-out evaluation.

## Predict

After training succeeds, pass the checkpoint created by that run and an audio
file in the canonical local data directory. The example uses the legacy
recording's filename; place the reviewed recording at this path first. Neither
the data nor the checkpoint is bundled.

```bash
python -m src.predict \
  --audio "data/audio/Tirmanpalli Bheemgal Road.wav" \
  --checkpoint results/checkpoints/submission_run.pt
```

The printed output is a greedy, unverified character decode. Blank IDs are
removed and consecutive duplicate IDs collapsed, as required by CTC greedy
decoding. A blank-only argmax sequence produces an empty string.

Inspect frame-level predictions:

```bash
python -m src.diagnose_prediction \
  --checkpoint results/checkpoints/submission_run.pt \
  --audio "data/audio/Tirmanpalli Bheemgal Road.wav"
```

## Tests and observed results

Run the built-in synthetic checks:

```bash
python -m src.train --self-test
python -m src.predict --self-test
```

The training self-test exercises output shapes, CTC loss, and backpropagation;
the prediction self-test checks repeated-token collapse and blank removal.
Both self-tests passed in the local Python 3.9.6 environment with PyTorch and
torchaudio 2.8.0 and librosa 0.11.0. Dataset validation also passed: all four
CSV rows were usable, with two samples per language and no missing or
unreadable audio. The audio loader emitted PySoundFile fallback warnings for
some M4A recordings, then successfully loaded them through audioread.

An optional one-sample overfitting test trains on one selected pair and writes
a new checkpoint; it is not needed for the read-only tests above:

```bash
python -m src.overfit_test \
  --audio-file enlish_sample.m4a \
  --checkpoint results/checkpoints/overfit_test.pt
```

Observed local results (not benchmark results):

| Checkpoint and recording | Expected | Decoded | Blank frames | CER | WER |
|---|---|---|---:|---:|---:|
| `overfit_one_te_sample.pt` on its training audio `enlish_sample.m4a` | `మీరు ఎలా ఉన్నారు?` | `మీరు ఎలా ఉన్నారు?` | 79.49% (124/156) | — | — |
| Same one-sample checkpoint on `Tirmanpalli Bheemgal Road.wav` (English) | `Hello, how are you? Today I am learning speech-to-text.` | `మీరు ఎల ఉనాారు        ` | 69.19% (265/383) | 87.27% | 100.00% |
| Same checkpoint on `english_sample_2.wav.m4a` (English) | `My name is Lokesh. Today I am testing my speech recognition model.` | `మీరు   ` | 93.67% (296/316) | 95.45% | 100.00% |
| Same checkpoint on `telugu_sample_2.wav.m4a` (Telugu) | `నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను.` | `మీరు ా   ` | 91.78% (346/377) | 91.55% | 100.00% |
| `diagnostic.pt` on `telugu_sample_2.wav.m4a` (Telugu) | `నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను.` | empty string | 100.00% (377/377) | 100.00% | 100.00% |

The one-sample checkpoint exactly memorized its training pair; it did not
generalize to the other English and Telugu recordings. CER is Unicode
code-point edit distance divided by reference characters; WER uses
whitespace-separated tokens. Checkpoints are local-only and are not bundled.
No held-out speakers or recordings were available.

## Project structure

```text
.
├── requirements.txt
├── docs/
│   └── architecture.md
└── src/
    ├── audio_preprocessing.py
    ├── diagnose_prediction.py
    ├── model.py
    ├── overfit_test.py
    ├── predict.py
    ├── train.py
    └── verify_dataset.py
```

See [docs/architecture.md](docs/architecture.md) for the model data flow.
Use the `python -m src...` commands above; the legacy root `main.py` is only a
placeholder and is not the application entry point.

## Known limitations

- Four training utterances are insufficient for meaningful training of a
  reliable bilingual recognizer.
- There is no held-out validation or test dataset, so generalization has not
  been measured.
- The submission checkpoint is blank-dominated on the tested Telugu audio.
- Exact overfitting of one sample does not imply transcription of unseen
  recordings.
- The model predicts characters and uses greedy decoding; there is no beam
  search, language model, or language-specific post-processing.
- Training currently runs on CPU and can be slow for longer recordings.
- Audio/transcript verification flags are manual attestations and do not
  replace independent transcript review.
