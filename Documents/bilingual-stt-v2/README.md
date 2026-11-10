> **Submission status:** Sections 12-19 contain historical plans and earlier
> dataset counts. For the current dataset, setup and verified results, see
> [Section 26](#26-current-project-status-and-results) and the
> [five-phase report](PHASE_5_REPORT.md).

## 12. Mel Spectrogram Feature Extraction

Librosa is used to extract Mel spectrogram
features from speech recordings.

Configuration:
- Sample Rate: 16000 Hz
- Mel Bands: 80
- FFT Size: 2048 (Librosa default)
- Hop Length: 512 (Librosa default)

The extracted features represent how
speech frequencies change over time.

These features will be used as input
to the CNN-BiLSTM neural network.
## 13. Transcript Preparation

A CSV file is used to connect audio recordings
with their corresponding text transcripts.

CSV columns:
- audio_path
- language
- transcript
- split

Each recording must have an accurate transcript.

The dataset will be divided into training,
validation, and testing sets.

Recordings used for testing will not be
included in model training.
## 14. New Dataset Collection Plan

The new dataset will contain English and Telugu
speech recordings collected for this project.

Initial target:
- 12 English recordings
- 12 Telugu recordings
- 16 training recordings
- 4 validation recordings
- 4 testing recordings

Each recording will have an accurate transcript.

The dataset will be checked for missing files,
incorrect paths, and duplicate recordings.

This is a small prototype dataset, not a
large-scale speech recognition dataset.
## 15. Dataset Verification

The dataset was checked for:
- Audio format and sample rate
- Mono audio channels
- Matching transcript entries
- Missing and duplicate files
- Training, validation, and testing splits

The dataset must pass verification before
neural network training begins.
## 16. New Audio Transcript Preparation

Four new WAV recordings were added to the
dataset metadata.

Each recording has:
- Audio file path
- Language
- Transcript
- Dataset split

Transcripts are entered manually to ensure
they match the spoken recordings.

Dataset splitting will be completed after
collecting additional recordings.
## 17. Dataset Verification

The English and Telugu audio dataset was checked
before model training. The following checks passed:

- All audio file paths in the dataset metadata exist.
- Language labels are present and are English or Telugu.
- No duplicate audio paths were found.
- WAV format is consistent across the referenced files:
  mono, 16-bit PCM, 16 kHz.

## 18. Dataset Expansion

Additional English and Telugu recordings
are being collected for model development.

Each recording contains one spoken sentence
with a corresponding transcript.

The new recordings will be converted into
16 kHz mono WAV format before training.## 19. Dataset Splitting

The dataset contains 12 recordings:
6 English and 6 Telugu.

The dataset was divided into:
- Training: 8 recordings
- Validation: 2 recordings
- Testing: 2 recordings

A fixed random seed of 42 was used
to make the split reproducible.

The split is intended for initial
pipeline testing, not reliable
real-world accuracy evaluation.

## 20. CNN-BiLSTM Model

The acoustic model in `src/model.py` accepts batched log-Mel
features with shape `(batch, 80, time)` and returns unnormalized
logits with shape `(batch, time, num_classes)`.
Set `num_classes` to the size of the character vocabulary,
including the CTC blank token.

Two small CNN layers extract acoustic patterns and reduce the
frequency dimension from 80 to 20. They preserve the time
dimension, so the output has the same number of time steps as
the input. A bidirectional LSTM considers the sequence in both
directions, and a linear layer produces character-class logits.

Run the shape smoke test with `python3 src/model.py` in an
environment with PyTorch installed. The model does not handle
variable-length padding or compute CTC loss.

## 21. CTC Character Vocabulary

`src/vocabulary.py` builds a vocabulary using only transcripts
whose split is `train`. Characters are Unicode code points sorted
in a deterministic order, preserving case, punctuation, spaces,
and the original text without Unicode normalization.

ID 0 is the CTC blank and ID 1 is the unknown-character token.
Characters not seen in training encode as ID 1 and decode as
`<unk>`, so their original values cannot be recovered. The decode
utility converts IDs directly to tokens; it does not perform CTC
collapse or remove blanks.

Run `python3 src/vocabulary.py` to create `models/vocab.json`.
The command refuses to overwrite an existing vocabulary file.
Use `Vocabulary.load` to reuse the saved mappings and set the
model's `num_classes` to the vocabulary size.

## 22. Dataset and DataLoader

`src/dataset.py` reads the existing `train`, `val`, or `test` rows,
loads the WAV files, extracts 80-band log-Mel features using
`src/audio_preprocessing.py`, and encodes transcripts with the saved
`models/vocab_v3.json` vocabulary by default. Unknown characters use
the existing `<unk>` ID.

The default batch size is 2. The collate function pads features to
`(batch, 80, max_time)`, concatenates the character IDs into a
one-dimensional target tensor, and returns the original feature and
target lengths. Feature lengths exclude padding. The current model
preserves time, so these feature lengths are its CTC input lengths.
Each sample must have at least one frame per target ID, plus one
additional frame for every pair of identical adjacent IDs.

Run `python3 -m src.dataset` from the project root to print a training
batch's shapes and check CTC lengths in all three splits. The current
bidirectional LSTM does not mask padded frames internally: padding can
affect valid outputs even though CTC input lengths exclude it. This
is addressed in the training script by running each sample through the
model without its padded frames.

## 23. CTC Training

Run `python3 -m src.train` from the project root for a one-batch
forward/backward smoke test. It checks all splits for valid CTC targets
first; it does not update model weights or save a checkpoint.

After approving training, run `python3 -m src.train --train`. The
script uses seed 42, Adam at learning rate 0.001, batches of two, and
five epochs by default. `--epochs` may select one through five epochs.
It prints separate training and validation losses and creates
`models/cnn_bilstm_ctc_v3.pt` only after successful training. Existing
checkpoints are never overwritten; `--checkpoint` selects another path.
The checkpoint stores its vocabulary mapping for compatibility checks.

Each sample is passed through the model without padding, then its
logits are padded for CTC. This prevents padding from affecting the
bidirectional LSTM. Validation characters not in the training
vocabulary use `<unk>`, so validation loss is measured on those encoded
targets rather than the original unknown characters.

## 24. Greedy CTC Inference Test

Run `python3 -m src.inference` from the project root to load the
checkpoint and vocabulary and test one recording from each language
in the test split. If there are multiple test recordings for a
language, the first path in sorted order is selected.
The defaults are `models/cnn_bilstm_ctc_v3.pt` and
`models/vocab_v3.json`; the checkpoint must first be created by an
approved training run. Inference rejects mismatched class counts or
missing/different vocabulary mappings, including legacy checkpoints.

Greedy decoding chooses the highest-scoring ID at each time step,
collapses consecutive repeats, then removes blank IDs. The script
prints the original transcript, prediction, and character error rate
(CER). An empty prediction is shown explicitly.

CER is Unicode code-point edit distance divided by the number of
reference code points, without normalizing case, spaces, punctuation,
or Unicode. An unknown predicted token is shown as `<unk>` and counts
as one replacement character for CER. Reference characters absent from
the selected vocabulary cannot be predicted exactly. The two test
samples are too few to establish real-world accuracy.

## 25. Pretrained Whisper Inference

This optional local baseline uses OpenAI's pretrained multilingual
Whisper `small` model. It is separate from the from-scratch
CNN-BiLSTM + CTC model: it uses Whisper's own preprocessing and
tokenizer, not the project's vocabulary or checkpoints. It does not
train or modify the dataset.

From the project root, create a separate environment and install the
optional dependencies. FFmpeg must also be available on `PATH`.

```bash
python3 -m venv .venv-whisper
.venv-whisper/bin/python -m pip install -r requirements-whisper.txt
```

Transcribe a WAV file in the language spoken:

```bash
.venv-whisper/bin/python -m src.whisper_inference path/to/english.wav --language en
.venv-whisper/bin/python -m src.whisper_inference path/to/telugu.wav --language te
```

The script runs on CPU and prints the transcript with leading and
trailing whitespace removed. The first run downloads approximately
484 MB of model weights to Whisper's cache (`~/.cache/whisper` by
default); subsequent runs can use the cached model offline. Audio is
processed locally. Accuracy and runtime depend on the recording and
hardware; the pretrained baseline is not evidence that the
from-scratch model has learned to recognize speech.

## 26. Current Project Status and Results

### Setup

For the from-scratch pipeline, use Python 3.9 or later and install
PyTorch, NumPy and Librosa in a Python environment. FFmpeg and FFprobe
are useful for audio verification. The existing `requirements.txt` is
empty and does not provide a reproducible dependency lockfile.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch numpy librosa
.venv/bin/python -m src.train
```

The last command is a forward/backward smoke test, not training; it
does not update weights or save a checkpoint. Actual training requires
`--train` and an unused `--checkpoint` path, as described in Section 23.
The optional Whisper environment is separate; see Section 25.

### Dataset and architecture

The current `data/transcripts.csv` contains 32 recordings: 28 training
(14 English and 14 Telugu), two validation (one per language), and two
test (one per language). Recordings are 16 kHz, mono, 16-bit PCM WAV.
The small held-out splits cannot establish reliable generalization.

The from-scratch model uses 80-band log-Mel features, two CNN layers,
a bidirectional LSTM, a linear output layer and CTC loss. Its v3
vocabulary has 75 tokens, including blank ID 0 and unknown ID 1.
Training processes only training examples for optimizer updates;
validation examples are evaluated separately.

### From-scratch CNN-BiLSTM + CTC results

The expanded-data experiment trained 10 epochs and continued for
another 10. The saved 20-epoch checkpoint is
`models/cnn_bilstm_ctc_v3_continued_20.pt`. The final training loss
was 3.6366 and validation loss was 3.6660. On the two validation
samples, greedy decoding predicted punctuation only:

| Validation language | Greedy CER | Blank frames |
|---|---:|---:|
| English | 98.00% | 103/104 |
| Telugu | 97.73% | 107/108 |
| Combined | 97.87% | 210/212 (99.06%) |

Combined CER is weighted by reference code points, not averaged across
languages. A diagnostic beam-width-10 evaluation of the same checkpoint
produced seven spaces and a period for each sample, with 85.11%
combined CER; the lower score does not indicate word recognition.

An older test evaluation of the original 60-token, five-epoch
checkpoint, trained before the dataset expansion, had 100.00% CER on
one English and one Telugu sample. Those results are not test results
for the expanded 75-token checkpoint. No test evaluation of that
checkpoint is reported here.

`python -m src.inference` evaluates test recordings using the default
10-epoch v3 checkpoint, not the 20-epoch continuation. Avoid running it
as a substitute for the results above. CER is case-sensitive Unicode
code-point edit distance divided by reference length; the project
function does not normalize Unicode automatically.

### Separate pretrained Whisper validation

The following are individual validation samples, not overall accuracy.
Whisper `small` is pretrained and is not the CNN-BiLSTM model. Telugu
reference and prediction strings were not retained in the public
summary; both were NFC-normalized already, so normalization did not
change the reported CERs.

| Model / language | Split sample count | CER |
|---|---:|---:|
| Whisper small / English | 1 validation sample | 0.00% |
| Whisper small / Telugu, original run | 1 validation sample | 84.09% |
| Whisper small / Telugu, comparison rerun | Same validation sample | 129.55% |

The original Whisper result and comparison both used `small`; they are
not different model sizes. The decoder fell back to sampling, so runs
can vary. CER can exceed 100% when insertions are present. These
observations came from local evaluation. The public-safe aggregate
metrics are recorded in `results/submission_summary.json`; raw
transcripts, predictions and experiment JSON files are excluded from
the proposed submission.

### Limitations

Telugu recognition remains inaccurate. Twenty-eight training
recordings are too few to cover the variety of speakers, accents,
words, environments and acoustic conditions required for robust
recognition. Decreasing loss and single-recording memorization do not
demonstrate held-out accuracy. Validation and test each contain only one
recording per language. The Telugu validation audio is relatively quiet;
its filename maps to the stated reference, but audio intelligibility and
the exact spoken wording require human listening. No general accuracy
claim is supported by these experiments.

### Privacy-safe release scope

The proposed source release excludes `data/`, raw audio, the transcript
CSV, model checkpoints, raw experiment JSON, Whisper weights, local
virtual environments and caches. The character vocabulary contains
character IDs, not complete transcripts. Audio and transcripts should
only be shared after confirming speaker consent and transcript-sharing
permission.