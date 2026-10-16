
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

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

    def output_lengths(self, input_lengths):
        lengths = torch.as_tensor(input_lengths, dtype=torch.long).cpu()
        for layer in self.cnn:
            if isinstance(layer, nn.Conv1d):
                lengths = (
                    lengths
                    + 2 * layer.padding[0]
                    - layer.dilation[0] * (layer.kernel_size[0] - 1)
                    - 1
                ) // layer.stride[0] + 1
        return lengths

    def forward(self, x, input_lengths=None):
        # Input: [batch, 80 mel features, time frames]
        if input_lengths is None:
            input_lengths = torch.full(
                (x.size(0),), x.size(2), dtype=torch.long
            )
        else:
            input_lengths = torch.as_tensor(input_lengths, dtype=torch.long).cpu()

        if input_lengths.numel() != x.size(0):
            raise ValueError("input_lengths must contain one length per sample")
        if torch.any(input_lengths <= 0) or torch.any(input_lengths > x.size(2)):
            raise ValueError("input_lengths must be positive and no longer than the input")

        output_lengths = self.output_lengths(input_lengths)
        cnn_sequences = [
            self.cnn(x[index : index + 1, :, :length]).squeeze(0).transpose(0, 1)
            for index, length in enumerate(input_lengths.tolist())
        ]
        x = pad_sequence(cnn_sequences, batch_first=True)

        # LSTM input: [batch, time frames, CNN features]
        packed = pack_padded_sequence(
            x,
            output_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed, _ = self.bilstm(packed)
        x, _ = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=x.size(1),
        )

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