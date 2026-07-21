"""LSTM-based sequence encoder.

Turns an API-call-sequence (or a TF-IDF vector treated as a length-1
"sequence") into a fixed-size hidden representation that downstream
components (projection head / classifier) consume.
"""
from torch import nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        """x: [batch, seq_len, input_size] -> [batch, hidden_dim]"""
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]
