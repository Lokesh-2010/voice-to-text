
import torch
import torch.nn as nn

class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_mels=80,
        cnn_channels=128,
        hidden_size=128,
        num_layers=2,
        vocab_size=100
    ):
        super().__init__()

        # CNN learns patterns from audio features
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mels,
                out_channels=cnn_channels,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )

        # BiLSTM learns sequential speech patterns
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Predict a character at every time step
        self.classifier = nn.Linear(
            hidden_size * 2,
            vocab_size
        )

    def forward(self, x):
        # Input: [batch, 80 mel features, time frames]
        x = self.cnn(x)

        # LSTM input: [batch, time frames, CNN features]
        x = x.transpose(1, 2)

        x, _ = self.bilstm(x)

        # Output: [batch, time frames, vocabulary size]
        logits = self.classifier(x)

        return logits

if __name__ == "__main__":
    model = CNNBiLSTM(vocab_size=100)

    # Simulate 2 audio recordings, each with 200 frames
    dummy_audio = torch.randn(2, 80, 200)

    predictions = model(dummy_audio)

    print("Model created successfully!")
    print("Input shape:", dummy_audio.shape)
    print("Output shape:", predictions.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))