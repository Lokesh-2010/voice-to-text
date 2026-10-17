# Inference demonstration (familiar sentences)

> **These are NOT independent test results.** Every clip below was used to
> train this checkpoint, so a match shows the model memorized its 4-clip
> training set, not that it recognizes new speech. On an unseen English
> sentence the same checkpoint produced 0/12 correct words.

- Checkpoint: `models/cnn_bilstm_ctc_groupnorm_700ep.pt` (CNN-BiLSTM-CTC, GroupNorm, trained 700 epochs on all 4 clips)
- Inference: `src.predict` (greedy CTC decoding); predictions are model output, references are read from `data/transcripts/transcripts.csv`
- Generated: 2026-10-17 23:35

## English: `Tirmanpalli Bheemgal Road.wav`

- Verified reference: Hello, how are you? Today I am learning speech-to-text.
- Model prediction:   Hello, how are you? Today I am learning speeto-text.
- Exact match: **NO**

## English: `english_sample_2.wav.m4a`

- Verified reference: My name is Lokesh. Today I am testing my speech recognition model.
- Model prediction:   My name is Lokesh. Today I am testing my speech recognition model.
- Exact match: **YES**

## Telugu: `enlish_sample.m4a`

- Verified reference: మీరు ఎలా ఉన్నారు?
- Model prediction:   మీరు ఎలా ఉన్నారు?
- Exact match: **YES**

## Telugu: `telugu_sample_2.wav.m4a`

- Verified reference: నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను.
- Model prediction:   నా పేరు లోకేష్. ఈ రోజు నా స్పీచ్ రికగ్నిషన్ మోడల్‌ను పరీక్షిస్తున్నాను.
- Exact match: **YES**

**Summary: 3/4 exact matches on familiar (training) clips.**
