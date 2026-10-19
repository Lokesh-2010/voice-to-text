# Inference demonstration (14-clip checkpoint)

> **Training examples prove memorization, not recognition.** The two training clips
> below were in the training set. The two UNSEEN clips were excluded from training
> and the model **failed on both**. The model does not generalize to new speech yet.

- Checkpoint: `models/cnn_bilstm_ctc_groupnorm_14clips_700ep.pt` (CNN-BiLSTM-CTC, GroupNorm, 700 epochs, 14 clips: 7 English + 7 Telugu)
- Inference: `src.predict` greedy CTC decoding; predictions are real model output, nothing hardcoded
- Generated: 2026-10-18 09:06

| Clip | Type | Expected | Predicted | Result |
|---|---|---|---|---|
| `english_02.wav` | Training | Today I am learning Python programming. | Today I am learning Python programming. | MATCH |
| `telugu_02.wav` | Training | నేను ఈ రోజు చదువుకుంటున్నాను. | నేను ఈ రోజు చదువుకుంటున్నాను. | MATCH |
| `english_05.wav` | UNSEEN test | I enjoy playing cricket with my friends. | Hel  అsపే amespuetearnui srigeinoith. | **NOT MATCH** |
| `telugu_05.wav` | UNSEEN test | ఈ రోజు వాతావరణం చాలా బాగుంది. | నేను హైదర వువటుదాన్నాను. | **NOT MATCH** |

Held-out scores (from the README): WER 1.00, CER 0.83. Training CER is 0.004.
The unseen sentences also need characters missing from the vocabulary (`j`, `ణ`).
