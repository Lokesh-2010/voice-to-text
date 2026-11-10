# Five-Phase Submission Summary

This is a provisional five-phase project summary. The manager's
original phase names and acceptance criteria were not available in the
project files or current session, so completion against those specific
requirements cannot be confirmed.

| Phase | Work performed | Status and evidence |
|---|---|---|
| 1. Data preparation | Prepared a 32-recording dataset with 28 train, 2 validation, and 2 test samples. | Dataset details and raw recordings are excluded from this privacy-safe release pending permission confirmation. |
| 2. Features and vocabulary | Implemented 16 kHz log-Mel preprocessing and a train-derived 75-token character vocabulary. | [Preprocessing](src/audio_preprocessing.py), [vocabulary](models/vocab_v3.json); CTC blank ID 0. |
| 3. Model and training | Built a CNN-BiLSTM + CTC model; trained 10 epochs and continued for 10 more on training records. | [Model](src/model.py); weights and raw training histories are excluded from this release. |
| 4. Evaluation | Evaluated the continued model on two validation samples; separately evaluated an older model on two test samples. | Validation greedy CER: English 98.00%, Telugu 97.73%, combined 97.87%. The older model's test CER was 100.00% for each language. Aggregate details: [privacy-safe metrics](results/submission_summary.json). |
| 5. Documentation and baseline | Documented setup, results and limitations, including separate pretrained Whisper inference. | [README](README.md#26-current-project-status-and-results); results do not meet reliable recognition accuracy. |

## Key results and limitations

- The 20-epoch from-scratch model predicted only `.` on both validation
  samples; 99.06% of validation frames chose blank. Its training and
  validation losses were 3.6366 and 3.6660, respectively.
- The saved 100% test CERs belong to the earlier 60-token, five-epoch
  checkpoint, not the 75-token continuation. No saved test evaluation
  of the continued checkpoint is claimed.
- Pretrained Whisper `small` is a separate model. On one validation
  recording per language, English CER was 0.00% and Telugu CER was
  84.09%. A subsequent run of the **same** Whisper small model on that
  Telugu recording yielded 129.55% CER; sampling can vary. These were
  session observations, not overall accuracy or saved result files.
- Telugu recognition remains inaccurate. Training on 28 recordings and
  evaluating on one held-out recording per language cannot establish
  robust generalization. Check the Telugu validation audio against its
  transcript by listening before drawing conclusions about its quality.
- Raw recordings, transcript metadata, checkpoints, and detailed JSON
  logs are excluded pending confirmation of sharing permission for the
  recordings and transcripts. The summary JSON contains aggregate
  metrics only.

No datasets, model weights or checkpoints were changed for this report.
