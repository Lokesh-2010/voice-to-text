# FLEURS bilingual pilot

The local FLEURS pilot is separate from the existing personal-recording CSV and
does not change the model or its checkpoints. It selects 200 recordings per
language from the official FLEURS splits:

| Project language | FLEURS config | Official train | Official validation (`dev`) | Official test | Total |
|---|---|---:|---:|---:|---:|
| English (`en`) | `en_us` | 160 | 20 | 20 | 200 |
| Telugu (`te`) | `te_in` | 160 | 20 | 20 | 200 |

Selection is deterministic for the pinned FLEURS repository revision. A
numeric source ID can appear more than once because FLEURS has multiple
recordings of a sentence; the pilot selects at most one recording per source
ID within each official split. The manifest retains the official split,
config, source ID, source transcript fields, source WAV name, and audio
SHA-256. Validation and test examples stay out of the training split. The
available FLEURS metadata does not provide a speaker ID in the source TSV, so
this preparation can preserve the official split boundary but cannot
independently detect speaker overlap between validation and test. FLEURS
describes its training speakers as separate from dev/test speakers.

## License and attribution

The pinned Hugging Face dataset metadata identifies FLEURS as **CC BY 4.0**.
When redistributing or publishing adapted material, provide appropriate
attribution, a link to the [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), identify changes,
and do not imply endorsement. This pilot records the selected subset and NFC
transcript normalization as changes in `data/fleurs/ATTRIBUTION.md`.

The dataset card asks users to cite:

> Conneau, Alexis, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth
> Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. “FLEURS: Few-shot Learning
> Evaluation of Universal Representations of Speech.” arXiv:2205.12446, 2022.

## Prepare and validate locally

The source archives are multi-gigabyte because FLEURS publishes complete
language/split archives. The preparation command streams those archives,
extracts only the 400 selected WAV files, and does not save the full archives.
It refuses to overwrite an existing `data/fleurs/` directory.

```bash
python -m src.fleurs_dataset --prepare
python -m src.fleurs_dataset --validate-only
```

The loader normalizes project transcripts to NFC and uses the prepared
184-class character vocabulary: it is built from **all official FLEURS
training-split transcripts** plus a fixed ASCII punctuation inventory, and
does not use validation or test text to define vocabulary. The audio pilot
still contains only the selected 160 training recordings per language. This
lets held-out transcripts be encoded when a character occurs elsewhere in the
official training split, but a character absent from the selected audio
examples may remain poorly learned. Validation checks the official split
mapping and counts, transcript structure, WAV readability, mono 16 kHz format,
sample counts and checksums, 80-bin Mel features, vocabulary coverage, and the
minimum CTC frame requirement.

The validation command does not establish that each transcript matches the
spoken audio. It verifies structure, audio processing, vocabulary coverage,
and CTC feasibility; transcript correctness requires a separate review.
FLEURS files and manifests are local data and are ignored by Git.

## Training

The FLEURS option is separate from the existing personal-data `--train`
workflow. It creates the CNN-BiLSTM classifier with the prepared vocabulary
size (184 output classes, including CTC blank ID 0), learns only from the
official train rows, and reports CTC loss on the official validation rows
after each epoch. It never loads the official test split. The best validation
loss checkpoint is saved only if the requested path does not already exist;
the default is `results/checkpoints/fleurs_cnn_bilstm_ctc.pt`.

First run the small smoke test. It performs one optimizer update using one
short English and one short Telugu training utterance, computes validation
loss on one utterance from each language, and saves no checkpoint:

```bash
python -m src.train --fleurs-smoke-test \
  --fleurs-manifest data/fleurs/manifest.csv
```

For an explicitly approved full pilot run, choose the epoch count and a fresh
checkpoint path:

```bash
python -m src.train --train-fleurs \
  --epochs 100 \
  --batch-size 1 \
  --learning-rate 0.001 \
  --fleurs-manifest data/fleurs/manifest.csv \
  --checkpoint results/checkpoints/fleurs_cnn_bilstm_ctc.pt
```

### Recorded pilot result

The project ran one CPU epoch on an Apple M3 using all 320 pilot training
recordings (160 per language), monitoring all 40 official validation
recordings (20 per language). Settings were batch size 1 and learning rate
0.001. Training CTC loss was **4.4810** and validation CTC loss was **3.8728**.
The run took approximately 3–5 minutes; exact wall time was not captured. It
saved `results/checkpoints/fleurs_pilot_20261016.pt` locally, with the expected
184 output classes and blank ID 0. This is one epoch only and does not
demonstrate recognition quality. The test split was not loaded, and no
test-set predictions, CER, or WER were calculated. Test-set prediction and
evaluation remain a separate step; do not use test data for epoch selection or
other training decisions.
