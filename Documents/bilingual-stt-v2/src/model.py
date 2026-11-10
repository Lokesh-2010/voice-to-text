"""A small CNN-BiLSTM acoustic model for CTC."""

import torch
from torch import nn


class CNNBiLSTM(nn.Module):
    """Map log-Mel features to unnormalized character-class logits.

    Input shape is (batch, 80, time), matching the frequency-first
    log-Mel features produced by audio_preprocessing.py after batching.
    Output shape is (batch, time, num_classes).

    Both CNN layers halve the frequency dimension, from 80 to 40 to 20.
    Their time stride is 1 and their time padding is 1, so the time
    dimension is unchanged. num_classes includes the CTC blank class.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include at least one character and the CTC blank")

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(2, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=32 * 20,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 80 or features.shape[2] == 0:
            raise ValueError("features must have shape (batch, 80, time) with time > 0")

        features = self.cnn(features.unsqueeze(1))
        features = features.permute(0, 3, 1, 2).flatten(start_dim=2)
        features, _ = self.lstm(features)
        return self.classifier(features)


if __name__ == "__main__":
    model = CNNBiLSTM(num_classes=32)
    model.eval()
    with torch.no_grad():
        for batch, time in ((2, 100), (1, 1), (3, 17)):
            logits = model(torch.randn(batch, 80, time))
            expected = (batch, time, 32)
            if tuple(logits.shape) != expected:
                raise RuntimeError(f"Expected shape {expected}, got {tuple(logits.shape)}")
            print(f"PASS: input {(batch, 80, time)} -> output {tuple(logits.shape)}")
